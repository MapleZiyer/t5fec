#!/bin/bash

# 错误处理
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# 设置环境变量
export CUDA_VISIBLE_DEVICES=5
export WANDB_PROJECT="llama-3.2-1b-instruct-grpo"      # 设置wandb项目名称
export WANDB_ENTITY="maplesakura-tianjin-university"   # 设置wandb用户名

# 打印训练配置信息
echo "Starting training with following configuration:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"

mkdir -p /work/2024/zhulei/t5fec/t5fec/checkpoints/llama-3.2-1b-instruct-sft-grpo

# 运行训练脚本
echo "Starting training..."
python ../models/grpollama3.py

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi