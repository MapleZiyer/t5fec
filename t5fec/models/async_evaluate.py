import asyncio
import aiohttp
import json
import logging
import numpy as np
import random
from tqdm import tqdm
from typing import List, Dict
from rouge_score import rouge_scorer
from evaluate import load

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 配置API参数
API_KEY = "sk-NVz2LEoGeiJ0vMTkt4nwTHestJiEoRcjs8aplkkAEjBPULme"
API_URL = "https://api.bianxie.ai/v1/chat/completions"

# 初始化评估指标
sari_metric = load("He-Xingwei/sari_metric")
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

async def async_gpt_request(session: aiohttp.ClientSession, prompt: str) -> str:
    """异步调用GPT-3.5 API（带速率限制和重试机制）"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 32768
    }

    # 速率控制信号量（最大5个并发请求）
    semaphore = asyncio.Semaphore(5)
    retries = 5
    base_delay = 1.0

    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.post(API_URL, json=payload, headers=headers) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                        logger.warning(f"请求被限速，第{attempt+1}次重试，等待{retry_after}秒")
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    data = await response.json()
                    return data['choices'][0]['message']['content']

            except aiohttp.ClientResponseError as e:
                if e.status == 429 and attempt < retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.warning(f"API请求失败（429），第{attempt+1}次重试，等待{delay:.2f}秒")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"API请求最终失败: {str(e)}")
                    return ""
            except Exception as e:
                logger.error(f"非预期错误: {str(e)}")
                return ""

        logger.error(f"超过最大重试次数{retries}")
        return ""

def calculate_metrics(mutated_source: str, prediction: str, original: str) -> Dict[str, float]:
    """计算SARI和ROUGE指标"""
    # 计算ROUGE
    rouge_scores = rouge_scorer_instance.score(original, prediction)
    
    # 计算SARI（使用mutated作为source，original作为reference）
    sari_score = sari_metric.compute(
        sources=[mutated_source],
        predictions=[prediction],
        references=[[original]]
    )

    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'sari': sari_score['sari']
    }

async def process_sample(session: aiohttp.ClientSession, sample: Dict, fewshot_examples: List[Dict]) -> Dict:
    """处理单个样本的异步验证（带随机延迟）"""
    # 构建包含few-shot的prompt
    prompt = "你是一个特征错误修正助手。以下是8个修正示例：\n"
    for example in fewshot_examples:
        prompt += f"错误语句: {example['mutated']}\n证据: {example['gold_evidence']}\n修正结果: {example['original']}\n\n"
    
    prompt += f"请根据以下新案例进行修正：\n错误语句: {sample['mutated']}\n证据: {sample['gold_evidence']}\n修正结果:"

    # 添加随机延迟（0.1-0.5秒）分散请求
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    # 发送带重试机制的请求
    response = await async_gpt_request(session, prompt)
    
    # 解析响应并计算指标
    metrics = calculate_metrics(sample['mutated'], response, sample['original'])
    return {
        'sample_id': sample.get('id', ''),
        'prediction': response,
        'metrics': metrics
    }

async def main():
    # 加载few-shot示例
    with open('../data/gold_negate_8-shot_2-retrieved-evidence_train_gpt-3.5-turbo.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]
    fewshot_examples = np.random.choice(train_data, size=8, replace=False).tolist()

    # 加载验证数据
    with open('../data/gold_negate_8-shot_2-retrieved-evidence_dev_gpt-3.5-turbo.jsonl', 'r') as f:
        dev_data = [json.loads(line) for line in f]

    # 初始化异步会话
    async with aiohttp.ClientSession() as session:
        tasks = [process_sample(session, sample, fewshot_examples) for sample in dev_data]
        results = []
        
        # 使用tqdm显示进度
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future
            results.append(result)
            
            # 每10个样本记录一次中间结果
            if len(results) % 10 == 0:
                avg_metrics = {
                    'rouge1': np.mean([r['metrics']['rouge1'] for r in results]),
                    'rouge2': np.mean([r['metrics']['rouge2'] for r in results]),
                    'rougeL': np.mean([r['metrics']['rougeL'] for r in results]),
                    'sari': np.mean([r['metrics']['sari'] for r in results])
                }
                logger.info(f"已处理 {len(results)} 样本，平均指标: {avg_metrics}")

    # 保存最终结果
    output_file = "async_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'model': 'gpt-3.5-turbo',
                'fewshot_count': 8,
                'dataset': 'gold_negate_8-shot_2-retrieved-evidence_dev'
            },
            'results': results,
            'average_metrics': {
                'rouge1': np.mean([r['metrics']['rouge1'] for r in results]),
                'rouge2': np.mean([r['metrics']['rouge2'] for r in results]),
                'rougeL': np.mean([r['metrics']['rougeL'] for r in results]),
                'sari': np.mean([r['metrics']['sari'] for r in results])
            }
        }, f, indent=2)

    logger.info(f"评估完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    asyncio.run(main())