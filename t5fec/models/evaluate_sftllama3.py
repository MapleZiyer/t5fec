from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

# 设置CUDA环境变量以便调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '4'

# **1. 加载已训练的模型和 Tokenizer**
model_path = "/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft6"
test_file = "/work/2024/zhulei/t5fec/t5fec/data/sft.jsonl"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 确保填充 token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# **2. 定义推理函数**
def generate_response(mutated_text, evidence_text, max_new_tokens=100):
    """
    使用训练好的模型进行推理
    :param mutated_text: 变异后的文本
    :param evidence_text: 证据文本
    :param max_new_tokens: 生成的最大 token 数
    :return: 生成的修正文本
    """
    # 构造输入格式
    input_text = f"mutation:'{mutated_text}'\n\nevidence:'{evidence_text}'\n\n"

    # 进行 Tokenization
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # 生成输出
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.3,  # 降低temperature以减少随机性
            do_sample=False  # 使用贪婪解码
        )

    # 解码生成的文本
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

# **3. 读取测试集并进行批量推理**

with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line.strip())
        mutated_text = sample["mutated"]
        evidence_text = sample["gold_evidence"]

        # 生成修正文本
        generated_text = generate_response(mutated_text, evidence_text)

        # 存储结果
        results = {
            "mutated": mutated_text,
            "gold_evidence": evidence_text,
            "original": sample["original"],  # 真实修正文本
            "generated": generated_text  # 生成的修正文本
        }

        print(f"Results: {results}")


print(f"测试完成，结果已保存至: {output_file}")
