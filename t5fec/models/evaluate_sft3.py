import logging
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from evaluate import load
from fc.program_generator import Reasoning_Program_Generator
from fc.program_execution import Program_Execution

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def evaluate_model(model_path="../checkpoints/Llama-3.2-1B-Instruct3"):
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    # 从sft.jsonl加载测试用例
    import json
    test_cases = []
    with open('../data/sft.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 只取前5个测试用例
                break
            data = json.loads(line)
            test_case = {
                "evidence": data['gold_evidence'],
                "claim": data['mutated']
            }
            test_cases.append(test_case)
    
    model.eval()
    for test_case in test_cases:
        # 构建输入
        prompt = f"mutation:'{test_case['claim']}'evidence:'{test_case['evidence']}'"
        
        # 生成文本
        inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput:\n{prompt}")
        print(f"\nOutput:\n{output_text}")
        
        # 提取<answer>标签中的内容
        answer_start = output_text.find('<answer>')
        answer_end = output_text.find('</answer>')
        if answer_start != -1 and answer_end != -1:
            output_text = output_text[answer_start + len('<answer>'):answer_end].strip()
        
        # 2. ROUGE分数
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(test_case['claim'], output_text)
        rouge_f1 = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        
        # 3. SARI分数
        sari = load("He-Xingwei/sari_metric")
        results_sari = sari.compute(sources=[test_case['claim']], predictions=[output_text], references=[""])
        sari_score = results_sari['sari']
        
        
        # 输出评估结果
        print(f"\n评估结果:")
        print(f"ROUGE-F1: {rouge_f1:.4f}")
        print(f"SARI: {sari_score:.4f}")

if __name__ == "__main__":
    evaluate_model()