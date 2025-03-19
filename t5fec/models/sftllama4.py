import logging
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, set_seed
from datasets import load_dataset

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # 设置随机种子
    set_seed(42)
    
    # 模型名称
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # 加载模型和分词器（支持 bf16 训练，需硬件支持）
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 设置填充 token 和填充方向
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft4",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=False,  # 使用 bf16 而不是 fp16
        run_name="Llama-3.2-1B-Instruct-sft-run",
    )
    
    # 加载训练数据集（假设数据格式为 JSON Lines 格式）
    dataset = load_dataset("json", data_files={"train": "../data/sft.jsonl"})
    
    # 定义数据预处理函数：构造 prompt 并使用 <answer> 标签包裹目标答案
    def preprocess_function(examples):
        prompt = f"mutation:'{examples['mutated']}'\n\nevidence:'{examples['gold_evidence']}'\n\n"
        target = f"<answer>{examples['original']}</answer>"
        
        # 对输入 prompt 进行编码，设定较长的 max_length
        inputs = tokenizer(prompt, max_length=4096, truncation=True, padding="max_length")
        # 对目标进行编码，目标一般较短
        labels = tokenizer(target, max_length=256, truncation=True, padding="max_length")
        
        # 将 labels 添加到输入字典中
        inputs["labels"] = labels["input_ids"]
        return inputs

    # 对数据集进行预处理，并移除原始字段
    processed_dataset = dataset["train"].map(
        preprocess_function,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset"
    )
    
    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer
    )
    
    # 开始训练
    logger.info("*** Starting SFT training ***")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()
