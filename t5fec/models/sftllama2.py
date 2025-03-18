import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        output_dir="../checkpoints/Llama-3.2-1B-Instruct",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,  # 每个GPU的batch size
        gradient_accumulation_steps=8,  # 梯度累积步数
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="Llama-3.2-1B-Instruct-run",
        # 单GPU训练配置
        fp16=False,  # 使用bf16而不是fp16
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
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 加载模型和分词器
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # 设置模型加载参数
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True
    )
    
    # 加载模型实例
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Llama使用右侧填充

    # 加载数据集
    dataset = load_dataset(
        'json',
        data_files={'train': '../data/gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl'}
    )

    # 数据预处理函数
    def preprocess_function(examples):
        prompt = """You are a feature error correction assistant. The user provides an incorrect statement, and you need to correct it. Evidence for correcting this incorrect statement will be provided to you, and you must use the given evidence to revise the incorrect statement. Only correct the erroneous parts of the sentence while keeping the rest intact. The original meaning of the sentence must not be changed.The revised statement should not differ significantly in semantics and format from the original statement.
        You must first think through the reasoning process in your mind before providing the user with the answer. The final answer should be enclosed within the <answer></answer> tags, respectively, i.e.,<answer> answer here </answer>.
        First, output the reasoning process, then output the final answer.The final answer should be enclosed in <answer></answer>.<answer></answer> tags pair can and must appear only once.
        User:'{original_statement}'.Evidence:'{evidence}' Assistant:"""

        inputs = prompt.format(evidence=examples['gold_evidence'], original_statement=examples['mutated'])
        targets = f"<answer>{examples['original']}</answer>"

        model_inputs = tokenizer(
            inputs,
            max_length=4096,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )
        
        labels = tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )
        
        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    # 处理数据集
    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=dataset['train'].column_names,
        desc="Processing dataset",
    )

    # 初始化SFT训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer
    )

    # 开始训练
    logger.info("*** Starting training ***")
    wandb.init(project="Llama-3.2-1B-Instruct-sft", name=training_args.run_name)
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
    wandb.finish()

if __name__ == "__main__":
    main()
