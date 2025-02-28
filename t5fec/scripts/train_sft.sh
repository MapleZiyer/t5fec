#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:/Users/bytedance/t5fec"

# 切换到正确的工作目录
cd /Users/bytedance/t5fec/t5fec

# 运行训练脚本
python models/sft.py