import logging
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer
import wandb

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    # 设置随机种子
    set_seed(42)
    
    # 指定预训练模型的路径或名称
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充token为eos token
    tokenizer.padding_side = "right"           # 对于Llama模型，通常使用右侧填充

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft3",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,        # 每个GPU的batch size
        gradient_accumulation_steps=8,          # 梯度累积步数
        gradient_checkpointing=True,            # 开启梯度检查点，节省显存
        bf16=True,                              # 使用bf16训练（需硬件支持）
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        do_train=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name="Llama-3.2-1B-Instruct-sft-run",
        fp16=False,  # 使用bf16而非fp16
    )

    # 检查是否有已存在的checkpoint，便于恢复训练
    last_checkpoint = None

    # 加载训练数据集，这里假设数据集为JSON Lines格式，包含 "mutated"、"original"、"gold_evidence" 字段
    dataset = load_dataset('json', data_files={'train': '../data/sft.jsonl'})

    # 数据预处理函数：构造 prompt，并使用 <answer> 标签包裹目标答案
    def preprocess_function(examples):
        # 构造输入 prompt，可根据需求修改
        data_instance = f"mutation:'{examples['mutated']}'\n\nevidence:'{examples['gold_evidence']}'\n\n"
        targets = f"<answer>{examples['original']}</answer>"
        
        # 使用 tokenizer 编码输入和目标；这里返回列表格式，便于后续 Dataset 处理
        model_inputs = tokenizer(
            data_instance,
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
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

    # 处理数据集，移除原始字段，保留预处理后的数据
    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=dataset['train'].column_names,
        desc="Processing training dataset"
    )

    # 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer
    )

    # 初始化 wandb 日志
    wandb.init(project="Llama-3.2-1B-Instruct-sft", name=training_args.run_name)
    
    # 开始训练
    logger.info("*** Starting SFT training ***")
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

    # 恢复 k,v cache 以便于快速推理（训练后可以开启）
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
