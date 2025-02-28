import os
import sys
import logging
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed
)
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    # 训练参数设置
    training_args = Seq2SeqTrainingArguments(
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
    )

    # 设置随机种子
    set_seed(42)

    # 加载模型和分词器
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    # 加载数据集
    dataset = load_dataset(
        'json',
        data_files={'train': '../data/gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl'}
    )

    # 数据预处理函数
    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['output']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
        labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # 处理数据集
    processed_dataset = dataset['train'].map(
        preprocess_function,
        remove_columns=dataset['train'].column_names,
        desc="Processing dataset",
    )

    # 初始化训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    logger.info("*** Starting training ***")
    trainer.train()

    # 保存模型
    trainer.save_model()
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()