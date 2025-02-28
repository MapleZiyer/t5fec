#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 运行训练脚本
python models/sft.py