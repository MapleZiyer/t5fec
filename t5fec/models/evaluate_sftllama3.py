from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

# 设置CUDA环境变量以便调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    inputs = tokenizer(input_text, max_length=4096, padding=True, truncation=True, return_tensors="pt")
    
    # 打印调试信息
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Attention mask shape: {inputs['attention_mask'].shape}")
    
    # 确保输入维度正确
    if len(inputs['input_ids'].shape) == 1:
        inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
    
    # 将输入移动到GPU
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 生成输出
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # 使用贪婪解码
            temperature=None,
            top_p=None,
            num_return_sequences=1,
            use_cache=True
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
