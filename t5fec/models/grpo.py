import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from trl import GRPOTrainer, get_peft_config
import wandb

from sentence_transformers import SentenceTransformer


from fc.program_generator import Reasoning_Program_Generator
from fc.program_execution import Program_Execution

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments:
    """Script arguments for the GRPO training script."""
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'tag_count'"
        },
    )

def main():
    # 训练参数设置
    training_args = transformers.TrainingArguments(
        output_dir="../checkpoints/flan-t5-large-grpo",
        learning_rate=2e-5,
        num_train_epochs=1,
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
        report_to=["wandb"],
        run_name="flan-t5-large-grpo-run"
    )
    # 添加model_init_kwargs参数
    setattr(training_args, 'model_init_kwargs', {})


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
    model_name = "google/flan-t5-large"
    if last_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {last_checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集并过滤
    dataset = load_dataset(
        'json',
        data_files={'train': '../data/train.json'}
    )
    
    # 只保留label为refutes的数据
    dataset = dataset.filter(lambda x: x['label'] == 'refutes')

    # 数据预处理函数
    def preprocess_function(examples):
        prompt = """You are an expert in correcting erroneous sentences. Based on the following evidence, identify and correct errors in the original statement. Ensure that the corrected statement maintains the same meaning and structure as the original, only changing the parts that are incorrect.

        Evidence: {evidence}

        Original statement: {original_statement}

        Corrected statement: """

        inputs = prompt.format(evidence=examples['evidence'], original_statement=examples['claim'])

        model_inputs = tokenizer(
            inputs,
            max_length=4096,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )
        
        return model_inputs

    # 处理数据集
    processed_dataset = dataset.map(
    preprocess_function,
    desc="Processing dataset",
    )

    # 设置模型参数
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    training_args.model_init_kwargs = {
        'torch_dtype': torch_dtype,
        'use_cache': False if training_args.gradient_checkpointing else True
    }

    # 初始化事实验证模块
    program_generator = Reasoning_Program_Generator()
    program_executor = Program_Execution()
    
    # 定义奖励函数
    # 初始化sentence transformer模型
    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    def accuracy_reward(outputs, batch):
        rewards = []
        for output, sample in zip(outputs, batch):
            # 使用sentence transformer计算文本相似度作为奖励
            output_embedding = similarity_model.encode(output, convert_to_tensor=True)
            target_embedding = similarity_model.encode(sample['claim'], convert_to_tensor=True)
            similarity = float(torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=0))
            print(f"\nOutput:{output}\nSimilarity{similarity}\n")
            if similarity < 0.7:
                rewards.append(0.0)
                continue

            # 使用事实验证模块评估生成文本
            programs = program_generator.generate_program(output)
            # 执行推理程序
            sample = [{
                    "idx":0,
                    "id":batch['id'],
                    "claim":output,
                    "gold":"",
                    "predicted_programs":programs,
                    "evidence":batch['evidence']
                }]
            prediction = program_executor.execute_on_dataset(sample)
            print(f"\nPrograms:{programs}\nPrediction:{prediction}\n")
            # 如果预测结果为True，说明生成的文本与证据一致
            if prediction:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            
        return torch.tensor(rewards)

    reward_funcs = [accuracy_reward]

    # 初始化GRPO训练器
    trainer = GRPOTrainer(
        model=model_name if last_checkpoint is None else last_checkpoint,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
    )

    # 开始训练
    logger.info("*** Starting training ***")
    wandb.init(project="t5fec-grpo", name=training_args.run_name)
    
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
    
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
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # 完成wandb日志
    wandb.finish()

if __name__ == "__main__":
    main()
