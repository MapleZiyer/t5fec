from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 指定模型和数据集
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 确保有填充 token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 加载数据集
train_dataset = load_dataset("json", data_files={"train": "../data/sft.jsonl"})

def preprocess_function(examples):
    prompt = f"mutation:'{examples['mutated']}'\n\nevidence:'{examples['gold_evidence']}'"
    response = f"<answer>{examples['original']}</answer>"

    # 拼接完整输入
    full_text = prompt + response

    # Tokenization
    tokenized = tokenizer(full_text, max_length=4096, padding="max_length", truncation=True)
    
    # 计算 prompt 长度
    prompt_tokenized = tokenizer(prompt, max_length=4096, truncation=True)
    prompt_len = len(prompt_tokenized["input_ids"])  # 计算 prompt 的 token 数

    # 构造 labels，忽略 prompt 部分
    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len  # 在 prompt 部分填充 -100

    tokenized["labels"] = labels
    return tokenized

dataset = train_dataset["train"].map(preprocess_function, batched=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    evaluation_strategy="no",
    save_total_limit=2,
    report_to="none",
    do_sample=False
)

# SFT 训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model("/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft3")
tokenizer.save_pretrained("/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft3")
