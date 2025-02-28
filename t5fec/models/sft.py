import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from trl import (
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import wandb  # 添加 wandb 导入

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    # 训练参数设置
    training_args = transformers.TrainingArguments(
        output_dir="../checkpoints/flan-t5-large-sft",
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=True,
        report_to=["wandb"],  # wandb日志配置
        run_name="flan-t5-large-sft-run",  # 设置wandb运行名称
    )

    # 设置随机种子
    set_seed(42)

    # 检查最新的checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # 设置日志级别
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 加载模型和分词器
    model_name = "google/flan-t5-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    dataset = load_dataset(
        'json',
        data_files={'train': '../data/gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl'}
    )

    # 数据预处理函数
    def preprocess_function(examples):
        prompt = """You are an expert in correcting erroneous sentences. Based on the following evidence, identify and correct errors in the original statement. Ensure that the corrected statement maintains the same meaning and structure as the original, only changing the parts that are incorrect.

Evidence: {evidence}

Original statement: {original_statement}

Corrected statement: """

        inputs = prompt.format(evidence=examples['gold_evidence'], original_statement=examples['mutated'])
        print(f"\n\n{inputs}\n\n")
        targets = examples['original']
        print(f"\n\n{targets}\n\n")

        model_inputs = tokenizer(
            inputs,
            max_length=4096,
            truncation=True,
            padding='max_length',  # Ensure padding
            return_tensors='pt'  # Ensure tensors are returned
        )
        
        labels = tokenizer(
            targets,
            max_length=4096,
            truncation=True,
            padding='max_length',  # Ensure padding
            return_tensors='pt'  # Ensure tensors are returned
        )
        
        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    # 处理数据集
    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=dataset['train'].column_names,
        desc="Processing dataset",
    )

    # 设置模型参数
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = model.to(dtype=torch_dtype)
    model.config.use_cache = False if training_args.gradient_checkpointing else True

    # 初始化SFT训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer
    )

    # 开始训练
    logger.info("*** Starting training ***")
    # 启动wandb日志记录
    wandb.init(project="t5fec-sft", name=training_args.run_name)  # 在wandb上创建一个项目和运行
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 保存模型
    logger.info("*** Saving model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # 恢复k,v cache用于快速推理
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)

    # 完成wandb日志
    wandb.finish()  # 完成当前的wandb日志

if __name__ == "__main__":
    main()
