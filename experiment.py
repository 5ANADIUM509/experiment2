import os
import time
import json
import torch
import re
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import List, Dict, Tuple
from contrastive_methods.method2 import generate_method2_batch
from utils import format_time, get_gpu_usage, get_available_gpu_memory

def run_experiment(
    model,
    tokenizer,
    data: List[Dict],
    alphas: List[float],
    templates: List[str],
    num_samples: int,
    batch_size: int,  # 强制使用用户输入的batch_size
    max_new_tokens: int,
    output_dir: str,
    samples_dir: str
) -> None:
    # 移除自动调整逻辑，直接使用用户输入的batch_size
    logger.info(f"使用用户指定的batch_size: {batch_size}（可用GPU显存: {get_available_gpu_memory():.1f}GB）")

    # 初始化结果表
    results_df = pd.DataFrame(columns=[
        "sample_id", "alpha", "template", "question", "response",
        "extracted_answer", "reference_answer", "is_correct", "elapsed_time"
    ])
    samples = data[:num_samples]
    total_tasks = len(alphas) * len(templates) * num_samples
    global_pbar = tqdm(total=total_tasks, desc="总进度")

    try:
        for alpha in alphas:
            for template_idx, template in enumerate(templates):
                logger.info(f"处理 Alpha={alpha} | Template={template_idx+1}")
                template_start = time.time()

                for i in range(0, num_samples, batch_size):
                    batch_end = min(i + batch_size, num_samples)
                    batch = samples[i:batch_end]
                    batch_ids = [s["sample_id"] for s in batch]
                    batch_questions = [s["question"] for s in batch]
                    batch_references = [s["answer"] for s in batch]

                    # 生成响应
                    batch_start = time.time()
                    prompts = [template.format(question=q) for q in batch_questions]
                    responses, _ = generate_method2_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens
                    )
                    batch_elapsed = time.time() - batch_start

                    # 提取答案并判断正确性（保持原优化逻辑）
                    extracted_answers = []
                    is_correct_list = []
                    for resp, ref in zip(responses, batch_references):
                        cleaned_resp = re.sub(r"[\n\r]+", " ", resp.strip())
                        cleaned_resp = re.sub(r"[^\w\s:.\-###]", "", cleaned_resp)

                        final_answer_matches = list(re.finditer(r"Final Answer:", cleaned_resp, re.IGNORECASE))
                        if len(final_answer_matches) > 1:
                            last_match = final_answer_matches[-1]
                            cleaned_resp = cleaned_resp[last_match.start():]

                        ans_match = re.search(
                            r"Final Answer:\s*([-+]?\d*\.\d+|\d+)\s*(###END###)?",
                            cleaned_resp,
                            re.IGNORECASE
                        )
                        extracted = ans_match.group(1).strip() if ans_match else "None"
                        extracted_answers.append(extracted)

                        ref_match = re.search(r"####\s*([-+]?\d*\.\d+|\d+)", ref)
                        ref_num = ref_match.group(1).strip() if ref_match else "None"

                        is_correct = False
                        if extracted != "None" and ref_num != "None":
                            try:
                                extracted_float = float(extracted)
                                ref_float = float(ref_num)
                                is_correct = abs(extracted_float - ref_float) < 1e-6
                            except ValueError:
                                is_correct = (extracted == ref_num)

                        is_correct_list.append(is_correct)

                    # 保存单样本结果
                    for sid, q, resp, extracted, ref, correct in zip(
                        batch_ids, batch_questions, responses, extracted_answers, batch_references, is_correct_list
                    ):
                        sample_filename = f"sample_{sid}_alpha{alpha}_template{template_idx+1}.json"
                        sample_path = os.path.join(samples_dir, sample_filename)
                        with open(sample_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "sample_id": sid,
                                "alpha": alpha,
                                "template_idx": template_idx + 1,
                                "template": template,
                                "question": q,
                                "response": resp,
                                "extracted_answer": extracted,
                                "reference_answer": ref,
                                "is_correct": correct,
                                "timestamp": time.time()
                            }, f, ensure_ascii=False, indent=2)

                    # 更新结果表
                    batch_results = [{
                        "sample_id": sid,
                        "alpha": alpha,
                        "template": template_idx + 1,
                        "question": q,
                        "response": resp,
                        "extracted_answer": extracted,
                        "reference_answer": ref,
                        "is_correct": correct,
                        "elapsed_time": batch_elapsed / len(batch)
                    } for sid, q, resp, extracted, ref, correct in zip(
                        batch_ids, batch_questions, responses, extracted_answers, batch_references, is_correct_list
                    )]
                    results_df = pd.concat([results_df, pd.DataFrame(batch_results)], ignore_index=True)
                    global_pbar.update(len(batch))

                logger.info(f"Alpha={alpha} | Template={template_idx+1} 完成 | "
                           f"耗时: {format_time(time.time() - template_start)} | "
                           f"GPU使用: {get_gpu_usage()}")

        # 保存总结果
        results_df.to_csv(os.path.join(output_dir, "total_results.csv"), index=False, encoding="utf-8-sig")
        logger.info(f"总结果保存至: {os.path.join(output_dir, 'total_results.csv')}")

    except Exception as e:
        logger.error(f"实验中断: {str(e)}")
        results_df.to_csv(os.path.join(output_dir, "interrupted_results.csv"), index=False, encoding="utf-8-sig")
    finally:
        global_pbar.close()
        torch.cuda.empty_cache()