import time
import torch
from transformers import GenerationConfig

class CDMethod1:
    """对比解码方法1（批量生成，速度快）"""
    def __init__(self, model, tokenizer, max_new_tokens=100):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.85,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    def generate(self, prompt, alpha):
        """生成回答并返回耗时"""
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=384  # 限制输入长度
        ).to(self.model.device)

        # 计时生成（8bit量化下推理）
        start_time = time.time()
        with torch.no_grad():  # 禁用梯度计算
            outputs = self.model.generate(** inputs, generation_config=self.gen_config)
        elapsed_time = time.time() - start_time

        # 解码结果
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, elapsed_time