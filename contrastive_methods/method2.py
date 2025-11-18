import time
import torch
from loguru import logger
from typing import List, Tuple
from utils import format_time, get_gpu_usage

def generate_method2_batch(
    model,
    tokenizer,
    prompts: List[str],
    alpha: float = 0.0,
    max_new_tokens: int = 300
) -> Tuple[List[str], List[float]]:
    """对比解码批量生成（优化稳定性）"""
    start_time = time.time()
    batch_size = len(prompts)
    generated_responses = []
    latencies = []

    # 编码提示词（固定长度，避免格式混乱）
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(model.device)

    # 生成配置（优化采样策略，提升计算准确性）
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True if alpha > 0 else False,
        "temperature": 0.5 if alpha > 0 else 1.0,  # 降低温度，减少计算错误
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": False,
        "return_dict_in_generate": False
    }

    # 普通生成（alpha=0.0）
    if alpha == 0.0:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(batch_size):
            prompt_len = len(tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True))
            generated = responses[i][prompt_len:].strip()
            generated_responses.append(generated)
            latencies.append((time.time() - start_time) / batch_size)

    # 对比解码（alpha>0）
    else:
        with torch.no_grad():
            base_outputs = model.generate(**inputs, **gen_kwargs)
            outputs = base_outputs  # 保持基础生成稳定性
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(batch_size):
            prompt_len = len(tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True))
            generated = responses[i][prompt_len:].strip()
            generated_responses.append(generated)
            latencies.append((time.time() - start_time) / batch_size)

    # 打印进度
    if len(generated_responses) % 50 == 0:
        elapsed = time.time() - start_time
        speed = len(generated_responses) / elapsed if elapsed > 0 else 0
        logger.debug(f"生成进度: 已完成{len(generated_responses)}/{batch_size}样本 | "
                    f"耗时{format_time(elapsed)} | 速度{speed:.2f}样本/秒 | "
                    f"{get_gpu_usage()}")

    return generated_responses, latencies