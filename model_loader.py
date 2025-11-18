from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from loguru import logger
import torch

def load_quantized_model(model_path: str):
    """加载8位量化模型（适配双RTX 5090）"""
    logger.info(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # 8位量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="fp8",
        bnb_8bit_use_double_quant=True
    )

    logger.info(f"Loading model (8-bit quantization): {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="balanced",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model.gradient_checkpointing_enable()
    logger.info(f"Model loaded successfully. Total GPUs used: {torch.cuda.device_count()}")
    return model, tokenizer