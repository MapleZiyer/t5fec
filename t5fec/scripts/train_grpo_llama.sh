#!/bin/bash

# 错误处理
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# 设置环境变量
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_PROJECT="llama-2-7b-chat-grpo"      # 设置wandb项目名称
export WANDB_ENTITY="maplesakura-tianjin-university"   # 设置wandb用户名

# 禁用GPU间的P2P通信
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

mkdir -p ../checkpoints/llama-2-7b-chat-grpo

# 打印训练配置信息
echo "Starting training with following configuration:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"

# 运行训练脚本
echo "Starting training..."

# 设置DeepSpeed相关环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export LOCAL_RANK=0

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    --node_rank=0 \
    --master_addr=localhost \
    ../models/grpollama.py \
    --deepspeed ../configs/ds_config_zero3_llama.json \
    --output_dir="../checkpoints/llama-2-7b-chat-grpo" \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --bf16 true

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi