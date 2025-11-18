import json
from loguru import logger
from typing import List, Dict

def load_data(file_path: str) -> List[Dict]:
    """
    加载JSONL格式的数据集
    :param file_path: 数据文件路径（questions.jsonl）
    :return: 样本列表，每个样本包含"sample_id"、"question"、"answer"
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        # 为每个样本添加唯一ID
        for idx, sample in enumerate(data):
            sample["sample_id"] = idx  # 用索引作为样本ID
        
        logger.info(f"成功加载数据集: {len(data)}个样本（来自{file_path}）")
        return data
    except FileNotFoundError:
        logger.error(f"数据集文件不存在: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"数据集格式错误（非JSONL）: {file_path}")
        raise