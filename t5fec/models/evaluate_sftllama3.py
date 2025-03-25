from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 训练好的模型路径
model_path = "/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft3"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

# 设置 padding token
tokenizer.pad_token = tokenizer.eos_token 

def generate_response(mutated, gold_evidence, max_length=4096):
    # 构造输入
    prompt = f"mutation:'{mutated}'\n\nevidence:'{gold_evidence}'"
    
    # 进行 tokenization
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 生成输出
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id  # 避免生成时报错
        )

    # 解码输出
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

import json

# 读取测试数据
test_file = "../data/sft.jsonl"
with open(test_file, "r") as f:
    test_samples = [json.loads(line) for line in f]

# 遍历测试集并生成输出
for sample in test_samples[:5]:  # 这里只测试前5个样本
    mutated = sample["mutated"]
    gold_evidence = sample["gold_evidence"]
    
    generated_text = generate_response(mutated, gold_evidence)
    print(f"\nMutation: \n{mutated}\n")
    print(f"Gold Evidence: \n{gold_evidence}\n")
    print(f"Generated Response: \n{generated_text}\n")