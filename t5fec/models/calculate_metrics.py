import json
import numpy as np

# 读取JSON文件
with open('/Users/bytedance/t5fec/t5fec/models/async_evaluation_results.json', 'r') as f:
    data = json.load(f)

# 提取所有样本的指标
rouge1_scores = [sample['metrics']['rouge1'] for sample in data['results']]
rouge2_scores = [sample['metrics']['rouge2'] for sample in data['results']]
rougeL_scores = [sample['metrics']['rougeL'] for sample in data['results']]
sari_scores = [sample['metrics']['sari'] for sample in data['results']]

# 计算平均值
new_averages = {
    'rouge1': np.mean(rouge1_scores),
    'rouge2': np.mean(rouge2_scores),
    'rougeL': np.mean(rougeL_scores),
    'sari': np.mean(sari_scores)
}

# 打印原始平均值和新计算的平均值进行对比
print('原始记录的平均值：')
print(json.dumps(data['average_metrics'], indent=2, ensure_ascii=False))
print('\n新计算的平均值：')
print(json.dumps(new_averages, indent=2, ensure_ascii=False))