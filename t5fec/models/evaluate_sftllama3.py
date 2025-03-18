import logging
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    # 加载模型和分词器
    model_path = "../checkpoints/Llama-3.2-1B-Instruct4"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 将模型设置为评估模式并移动到GPU
    model.eval()
    model.to('cuda')
    
    # 加载测试数据集
    dataset = load_dataset(
        'json',
        data_files={'test': '../data/sft.jsonl'}
    )
    
    # 对每个样本进行推理
    for example in dataset['test']:
        # 构建输入文本
        input_text = f"mutation:'{example['mutated']}'\n\nevidence:'{example['gold_evidence']}'\n\n"
        
        # 对输入进行编码
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=4096)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # 生成输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\ninput_text:\n{input_text}\n")
        print(f"\ngenerated_text:\n{generated_text}\n")
        
        # 提取<answer>标签中的内容
        start_idx = generated_text.find('<answer>')
        end_idx = generated_text.find('</answer>')
        
        if start_idx != -1 and end_idx != -1:
            answer = generated_text[start_idx + len('<answer>'):end_idx].strip()
            print(f"\nOriginal: {example['mutated']}")
            print(f"Generated: {answer}\n")
            print("-" * 100)

if __name__ == "__main__":
    main()