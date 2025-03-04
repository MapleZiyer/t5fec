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
    level=logging.INFO  # 设置基础日志级别为INFO
)
logger = logging.getLogger(__name__)

# 确保其他日志也设置正确的级别
datasets.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

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
    checkpoint_dir="../checkpoints/flan-t5-large-sft"
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
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="flan-t5-large-grpo-run",
    )
    # 添加reward_weights参数
    setattr(training_args, 'reward_weights', [1.0])
    setattr(training_args, 'reward_scale', 1.0)
    # 添加 model_init_kwargs 参数（这里先设为空字典）
    setattr(training_args, 'model_init_kwargs', {})
    # 添加 max_prompt_length 参数
    setattr(training_args, 'max_prompt_length', 512)
    # 添加 max_completion_length 参数
    setattr(training_args, 'max_completion_length', 256)
    # 添加 num_generations 参数
    setattr(training_args, 'num_generations', 4)  # 设置每个样本生成4个候选答案
    # 添加 use_vllm 参数
    setattr(training_args, 'use_vllm', False)
    # 添加 beta 参数
    setattr(training_args, 'beta', 0.1)
    # 添加 log_completions 参数
    setattr(training_args, 'log_completions', False)
    # 添加 temperature 参数
    setattr(training_args, 'temperature', 0.7)
    # 添加 sync_ref_model 参数
    setattr(training_args, 'sync_ref_model', True)

    # 设置随机种子
    set_seed(42)

    # 检查最新的 checkpoint
    last_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
    if last_checkpoint is not None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # 设置日志级别
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 定义预训练模型名称
    model_pretrained_name = "google/flan-t5-large"
    # 加载模型实例（若有 checkpoint 则从 checkpoint 加载，否则从预训练模型加载）
    if last_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {last_checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_pretrained_name)

    # 加载分词器（使用预训练模型名称）
    tokenizer = AutoTokenizer.from_pretrained(model_pretrained_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集并过滤
    dataset = load_dataset(
        'json',
        data_files={'train': '../data/train.json'}
    )
    # 只保留 label 为 refutes 的数据
    dataset = dataset.filter(lambda x: x['label'] == 'refutes')

    # 数据预处理函数
    # 在preprocess_function中使用更明确的日志格式
    def preprocess_function(examples):
        prompt = """You are an expert in correcting erroneous sentences. Based on the following evidence, identify and correct errors in the original statement. Ensure that the corrected statement maintains the same meaning and structure as the original, only changing the parts that are incorrect.
    
        Evidence: {evidence}
    
        Original statement: {original_statement}
    
        Corrected statement: """

        if not examples['evidence'] or not examples['claim']:
            return {}  # 直接返回空，避免数据异常    
        
        inputs = prompt.format(evidence=examples['evidence'], original_statement=examples['claim'])
    
        model_inputs = tokenizer(
            inputs,
            max_length=512,  # 将最大长度限制在模型支持的范围内
            truncation=True,
            padding='max_length'  # 修改为max_length以确保所有序列长度一致
        )

        model_inputs["prompt"] = inputs

        eos_token_id = tokenizer.eos_token_id
        if model_inputs["input_ids"][:, -1].item() != eos_token_id:
            model_inputs["input_ids"][:, -1] = eos_token_id  # 强制末尾是 EOS

        return model_inputs

    # 处理数据集
    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=[col for col in dataset['train'].column_names if col not in ['id', 'evidence', 'claim']],  # 保留必要的列
        desc="Processing dataset",
        keep_in_memory=True  # 保持数据在内存中
    )

    # 设置模型参数相关
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    # 由于我们传入的是已经加载好的模型实例，所以不需要 model_init_kwargs
    training_args.model_init_kwargs = None

    # 初始化事实验证模块
    program_generator = Reasoning_Program_Generator()
    program_executor = Program_Execution()
    
    # 定义奖励函数，并初始化 sentence transformer 模型
    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    def accuracy_reward(outputs, batch):
        rewards = []
        for output, sample in zip(outputs, batch):
            # 使用 sentence transformer 计算文本相似度作为奖励
            output_embedding = similarity_model.encode(output, convert_to_tensor=True)
            target_embedding = similarity_model.encode(sample['claim'], convert_to_tensor=True)
            similarity = float(torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=0))
            print(f"\nOutput: {output}\nSimilarity: {similarity}\n")
            if similarity < 0.7:
                rewards.append(0.0)
                continue
            # 使用事实验证模块评估生成文本
            programs = program_generator.generate_program(output)
            # 执行推理程序
            sample_data = [{
                "idx": 0,
                "id": sample['id'] if 'id' in sample else None,
                "claim": output,
                "gold": "",
                "predicted_programs": programs,
                "evidence": sample['evidence'] if 'evidence' in sample else None
            }]
            prediction = program_executor.execute_on_dataset(sample_data)
            print(f"\nPrograms: {programs}\nPrediction: {prediction}\n")
            if prediction:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards)

    reward_funcs = [accuracy_reward]

    # 初始化 GRPO 训练器，直接传入模型实例
    trainer = GRPOTrainer(
        model=model,
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

    # 同时保存分词器
    tokenizer.save_pretrained(training_args.output_dir)

    # 完成 wandb 日志
    wandb.finish()

if __name__ == "__main__":
    main()
