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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    checkpoint_dir="../checkpoints/llama-2-7b-chat-sft"
    # 训练参数设置
    training_args = transformers.TrainingArguments(
        output_dir="../checkpoints/llama-2-7b-chat-grpo",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,  # 进一步减小batch size以降低显存占用
        gradient_accumulation_steps=8,  # 相应增加梯度累积步数以保持总批次大小
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="llama-2-7b-chat-grpo-run",
        # 添加DeepSpeed配置
        deepspeed="../configs/ds_config_zero3_llama.json",
        local_rank=-1,  # 分布式训练的本地rank
        ddp_find_unused_parameters=False,  # 优化DDP性能
        fp16=False,  # 使用bf16而不是fp16
    )
    # 添加reward_weights参数
    setattr(training_args, 'reward_weights', [1.0])
    setattr(training_args, 'reward_scale', 1.0)
    # 添加 model_init_kwargs 参数
    setattr(training_args, 'model_init_kwargs', None)
    # 添加 max_prompt_length 参数
    setattr(training_args, 'max_prompt_length', 4096)
    # 添加 max_completion_length 参数
    setattr(training_args, 'max_completion_length', 256)
    # 添加 num_generations 参数
    setattr(training_args, 'num_generations', 2)  # 设置每个样本生成2个候选答案，以匹配全局训练批次大小
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
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    # 设置模型加载参数
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True
    )
    
    # 加载模型实例（若有 checkpoint 则从 checkpoint 加载，否则从预训练模型加载）
    if last_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {last_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # 加载分词器（使用预训练模型名称）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Llama使用右侧填充

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
        inputs = prompt.format(evidence=examples['evidence'], original_statement=examples['claim'])
        # 使用更明确的日志格式

        if not inputs.strip():
            inputs = "No input provided."
            logger.warning("Empty input detected, using default input")
    
        model_inputs = tokenizer(
            inputs,
            max_length=4096,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )

        # 确保所有必要的字段都存在且维度正确
        if 'input_ids' not in model_inputs or len(model_inputs['input_ids']) == 0:
            model_inputs['input_ids'] = tokenizer.encode("Empty input", max_length=4096, padding='max_length', truncation=True)
        if 'attention_mask' not in model_inputs or len(model_inputs['attention_mask']) == 0:
            model_inputs['attention_mask'] = [1] * len(model_inputs['input_ids'])

        # 确保输入数据维度正确
        if isinstance(model_inputs['input_ids'], (list, torch.Tensor)):
            model_inputs['input_ids'] = torch.tensor(model_inputs['input_ids'] if isinstance(model_inputs['input_ids'], list) else model_inputs['input_ids'].tolist())
            # 添加批次维度
            if len(model_inputs['input_ids'].shape) == 1:
                model_inputs['input_ids'] = model_inputs['input_ids'].unsqueeze(0)

        if isinstance(model_inputs['attention_mask'], (list, torch.Tensor)):
            model_inputs['attention_mask'] = torch.tensor(model_inputs['attention_mask'] if isinstance(model_inputs['attention_mask'], list) else model_inputs['attention_mask'].tolist())
            # 添加批次维度
            if len(model_inputs['attention_mask'].shape) == 1:
                model_inputs['attention_mask'] = model_inputs['attention_mask'].unsqueeze(0)

        # 添加prompt字段
        model_inputs['prompt'] = inputs

        return model_inputs

    # 处理数据集
    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=[col for col in dataset['train'].column_names if col not in ['id', 'evidence', 'claim']],  # 保留必要的列
        desc="Processing dataset",
        keep_in_memory=True  # 保持数据在内存中
    )

    # 初始化事实验证模块
    program_generator = Reasoning_Program_Generator()
    program_executor = Program_Execution()
    
    # 定义奖励函数，并初始化 sentence transformer 模型
    similarity_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
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
    wandb.init(project="llama-2-7b-chat-grpo", name=training_args.run_name)
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
