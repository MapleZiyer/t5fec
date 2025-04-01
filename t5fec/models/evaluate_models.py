import logging
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from rouge_score import rouge_scorer

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 模型路径
MODEL1_PATH = "/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-grpo"  # grpollama2.py训练的模型
MODEL2_PATH = "/work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft-grpo"  # grpollama3.py训练的模型

# 测试数据路径
TEST_DATA_PATH = "/Users/bytedance/t5fec/t5fec/data/gold_negate_8-shot_2-retrieved-evidence_dev_gpt-3.5-turbo.jsonl"

# 加载SARI评估指标
sari_metric = load("He-Xingwei/sari_metric")

# 初始化ROUGE评分器
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    logger.info(f"Loading model from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()  # 设置为评估模式
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def generate_correction(model, tokenizer, mutated, evidence, max_length=4096):
    """使用模型生成文本修正"""
    prompt = f"""
    You are a feature error correction assistant. The user provides an incorrect statement, and you need to correct it. Evidence for correcting this incorrect statement will be provided to you, and you must use the given evidence to revise the incorrect statement. Only correct the erroneous parts of the sentence while keeping the rest intact. All the information contained in the original sentence must be retained and cannot be deleted, only modified.The corrected sentence must not be exactly the same as the original sentence.The original meaning of the sentence must not be changed.The revised statement should not differ significantly in semantics and format from the original statement.
    You must first think through the reasoning process in your mind before providing the user with the answer. The reasoning process and the answer should be enclosed within the <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.
    All your outputs must be wrapped in tags. The thought process should be enclosed in <think></think>, and the final result should be enclosed in <answer></answer>. First, output the reasoning process, then output the final answer.The two pairs of tags must both be output.Each tag pair can and must appear only once.Tags cannot be nested.
    User:'{mutated}'.Evidence:'{evidence}' Assistant:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # 确定性生成
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取<answer>标签中的内容
    answer_start = response.find('<answer>')
    answer_end = response.find('</answer>')
    
    if answer_start != -1 and answer_end != -1:
        answer = response[answer_start + len('<answer>'):answer_end].strip()
        return answer
    else:
        # 如果没有找到标签，返回整个响应
        return None

def calculate_metrics(original, prediction, references=None):
    """计算SARI和ROUGE指标"""
    # 计算ROUGE分数
    rouge_scores = rouge_scorer_instance.score(original, prediction)
    
    # 计算SARI分数 (需要源文本、预测文本和参考文本)
    if references is None:
        references = [[""]]  # 如果没有参考文本，使用空字符串
    
    sari_score = sari_metric.compute(sources=[original], predictions=[prediction], references=references)
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'sari': sari_score['sari']
    }

def evaluate_models():
    """评估两个模型并比较结果"""
    # 加载模型
    model1, tokenizer1 = load_model_and_tokenizer(MODEL1_PATH)
    model2, tokenizer2 = load_model_and_tokenizer(MODEL2_PATH)
    
    if model1 is None or model2 is None:
        logger.error("Failed to load models. Exiting.")
        return
    
    # 加载测试数据
    try:
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(test_data)} test samples")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # 评估结果存储
    results = {
        'model1': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'sari': []},
        'model2': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'sari': []}
    }
    
    # 对每个测试样本进行评估
    for i, sample in enumerate(tqdm(test_data[:100], desc="Evaluating")):
        mutated = sample['mutated']
        original = sample['original']
        evidence = sample['gold_evidence']
        
        # 模型1生成
        prediction1 = generate_correction(model1, tokenizer1, mutated, evidence)
        # 模型2生成
        prediction2 = generate_correction(model2, tokenizer2, mutated, evidence)

        if(prediction1 or prediction2) is None:
            logger.warning(f"Failed to generate prediction for sample {i + 1}. Skipping.")
            continue
        
        # 计算指标
        metrics1 = calculate_metrics(original, prediction1)
        metrics2 = calculate_metrics(original, prediction2)
        
        # 存储结果
        for metric in ['rouge1', 'rouge2', 'rougeL', 'sari']:
            results['model1'][metric].append(metrics1[metric])
            results['model2'][metric].append(metrics2[metric])
        
        # 每10个样本打印一次中间结果
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1} samples")
            for model_name in ['model1', 'model2']:
                logger.info(f"{model_name} interim results:")
                for metric in ['rouge1', 'rouge2', 'rougeL', 'sari']:
                    logger.info(f"  {metric}: {np.mean(results[model_name][metric])}")
    
    # 计算平均指标
    avg_results = {
        'model1': {metric: np.mean(values) for metric, values in results['model1'].items()},
        'model2': {metric: np.mean(values) for metric, values in results['model2'].items()}
    }
    
    # 打印结果
    logger.info("\n===== Evaluation Results =====")
    logger.info(f"Model 1 (grpollama2): {MODEL1_PATH}")
    logger.info(f"Model 2 (grpollama3): {MODEL2_PATH}")
    logger.info("\nAverage Metrics:")
    
    for metric in ['rouge1', 'rouge2', 'rougeL', 'sari']:
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Model 1: {avg_results['model1'][metric]:.4f}")
        logger.info(f"  Model 2: {avg_results['model2'][metric]:.4f}")
        diff = avg_results['model2'][metric] - avg_results['model1'][metric]
        logger.info(f"  Difference (Model2 - Model1): {diff:.4f} ({'better' if diff > 0 else 'worse'})")
    
    # 保存详细结果到文件
    output_file = "model_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'average_results': avg_results,
            'detailed_results': results
        }, f, indent=2)
    
    logger.info(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    evaluate_models()