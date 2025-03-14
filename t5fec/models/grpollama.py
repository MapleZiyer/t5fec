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
# 移除LoRA相关导入

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
    checkpoint_dir="../checkpoints/llama-3.2-1b-instruct-grpo"
    # 训练参数设置
    training_args = transformers.TrainingArguments(
        output_dir="../checkpoints/llama-3.2-1b-instruct-grpo",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,  # 由于是较小的模型，可以适当增加batch size
        gradient_accumulation_steps=8,  # 相应减少梯度累积步数
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="llama-3.2-1b-instruct-grpo-run",
        # 添加DeepSpeed配置
        deepspeed="../configs/ds_config_zero3.json",
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
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # 设置模型加载参数
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False  # 始终禁用缓存以配合梯度检查点
    )
    
    # 加载模型实例（若有 checkpoint 则从 checkpoint 加载，否则从预训练模型加载）
    if last_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {last_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    

    
    # 确保模型处于训练模式
    model.train()
    
    # 显式设置requires_grad
    def set_requires_grad(module):
        for param in module.parameters():
            param.requires_grad = True
            if hasattr(param, 'data'):
                param.data.requires_grad_(True)
    
    model.apply(set_requires_grad)

    # 直接使用完整模型训练

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

        # 确保输入张量设置正确的requires_grad属性
        input_ids = torch.tensor(model_inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(model_inputs["attention_mask"], dtype=torch.long)
        
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask
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
    similarity_model.eval()  # 设置为评估模式
    for param in similarity_model.parameters():
        param.requires_grad = False  # 禁用梯度计算
    
    def accuracy_reward(prompts, completions, **kwargs):
        rewards = []
        for output, prompt in zip(completions, prompts):
            print(f"output1:{output}\n")
            output_text = output if isinstance(output, str) else str(output).strip()
            print(f"output2:{output_text}\n")
            # 确保输入文本格式正确
            # 确保输入文本格式正确并提取相关文本
            prompt_text = prompt.split('Original statement: ')[1].split('\n')[0].strip()
            evidence = prompt.split('Evidence: ')[1].split('\n')[0].strip()
            print(f"prompt_text:{prompt_text}\n")
            print(f"evidence:{evidence}\n")
                
            # 编码文本并确保维度正确
            output_embedding = similarity_model.encode(
                output_text, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            ).detach()
            print(f"output_embedding1:{output_embedding}\n")
                
            target_embedding = similarity_model.encode(
                prompt_text, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            ).detach()
            print(f"target_embedding1:{target_embedding}\n")    
                
            # 确保张量类型一致
            output_embedding = output_embedding.to(dtype=torch.float32)
            target_embedding = target_embedding.to(dtype=torch.float32)
            
            print(f"output_embedding2:{output_embedding}\n")
            print(f"target_embedding2:{target_embedding}\n") 
                
            # 如果是二维张量，取平均值得到一维向量
            if output_embedding.dim() > 1:
                output_embedding = output_embedding.mean(dim=0)
            if target_embedding.dim() > 1:
                target_embedding = target_embedding.mean(dim=0)

            print(f"output_embedding3:{output_embedding}\n")
            print(f"target_embedding3:{target_embedding}\n")     
                
            # 计算余弦相似度
            similarity = float(torch.nn.functional.cosine_similarity(
                output_embedding.unsqueeze(0),
                target_embedding.unsqueeze(0),
                dim=1
            ))
            print(f"\nOutput: {output}\nSimilarity: {similarity}\n")
            if similarity < 0.7:
                rewards.append(0.0)
                continue
            # 使用事实验证模块评估生成文本
            programs = program_generator.generate_program(output_text)
            # 执行推理程序
            sample_data = [{
                "idx": 0,
                "id": None,
                "claim": output_text,
                "gold": "",
                "predicted_programs": programs,
                "evidence": evidence
            }]
            prediction = program_executor.execute_on_dataset(sample_data)
            print(f"\nPrograms: {programs}\nPrediction: {prediction}\n")
            if prediction:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards, requires_grad=True)

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
    wandb.init(project="llama-3.2-1b-instruct-grpo", name=training_args.run_name)
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
