#!/bin/bash

# 错误处理
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# 设置环境变量
export CUDA_VISIBLE_DEVICES=4
export WANDB_PROJECT="llamafec-sft"      # 设置wandb项目名称
export WANDB_ENTITY="maplesakura-tianjin-university"   # 设置wandb用户名
export HF_HOME="/work/2024/zhulei/transformers_cache"  # 设置Hugging Face缓存目录

# 禁用GPU间的P2P通信
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

mkdir -p ../checkpoints/Llama-3.2-1B-Instruct5

# 打印训练配置信息
echo "Starting training with following configuration:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"

# 运行训练脚本
echo "Starting training..."
python ../models/sftllama3.py

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi
