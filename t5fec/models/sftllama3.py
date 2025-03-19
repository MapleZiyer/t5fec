from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch

# 指定模型和数据集
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 确保有填充token

tokenizer.add_tokens(['__ANSWER__'])
print(f'\n\n{tokenizer.tokenize("__ANSWER__")}\n\n')

# 加载数据集
train_dataset = load_dataset("json", data_files={"train": "../data/sft.jsonl"})

def preprocess_function(examples):
    inputs = f"mutation:'{examples['mutated']}'\n\nevidence:'{examples['gold_evidence']}'\n\n__ANSWER__<answer>{examples['original']}</answer>"
    #labels = f"<answer>{examples['original']}</answer>"

    inputs = tokenizer(inputs, max_length=4096, padding="max_length", truncation=True)

    inputs["labels"] = inputs["input_ids"].copy()

    return inputs

dataset = train_dataset["train"].map(preprocess_function)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 训练参数
training_args = TrainingArguments(
    output_dir="/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft5",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    evaluation_strategy="no",
    save_total_limit=2,
    report_to="none"
)

# 数据整理器
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    mlm=False,
    response_template="__ANSWER__"
)

# SFT 训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model("/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft5")
tokenizer.save_pretrained("/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft5")
