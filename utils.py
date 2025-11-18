import os
import time
import torch
from loguru import logger

def create_output_dir(dir_path: str) -> None:
    """创建目录（若不存在则自动创建）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"创建输出目录: {dir_path}")
    else:
        logger.info(f"输出目录已存在: {dir_path}")

def format_time(seconds: float) -> str:
    """将秒数格式化为 时:分:秒"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_gpu_usage() -> str:
    """获取GPU显存使用情况"""
    if not torch.cuda.is_available():
        return "无可用GPU"
    usage = []
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        used = torch.cuda.memory_allocated(i) / 1024**3
        usage.append(f"GPU{i}: {used:.1f}GB/{total:.1f}GB")
    return " | ".join(usage)

def get_available_gpu_memory() -> float:
    """计算所有GPU的可用显存总和（GB）"""
    if not torch.cuda.is_available():
        return 0.0
    total_available = 0.0
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        used = torch.cuda.memory_allocated(i)
        available = (total - used) / 1024**3
        total_available += available
    return total_available