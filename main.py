import argparse
import os
import time
from datetime import datetime
from model_loader import load_quantized_model
from data_loader import load_data
from experiment import run_experiment
from utils import create_output_dir

# 最终优化英文模板（强化数学计算逻辑+严格格式）
ENGLISH_TEMPLATES = [
    # Template 1: 详细解题+计算步骤强调
    """You are a math-savvy student. Solve the question step by step, focus on correct calculation, then output the final answer in the EXACT format (MUST follow):
'Final Answer: [number]###END###'

Example:
Question: James runs 3 sprints 3 times a week. He runs 60 meters each sprint. Total meters?
Step 1: Total sprints per week = 3 times/week * 3 sprints/time = 9 sprints
Step 2: Total meters = 9 sprints * 60 meters/sprint = 540 meters
Final Answer: 540###END###

Question: {question}""",
    
    # Template 2: 简洁计算+格式强制
    """Calculate the answer correctly and output in this REQUIRED format (NO extra text, MUST end with ###END###):
'Final Answer: [number]###END###'

Question: {question}""",
    
    # Template 3: 逻辑推理+计算验证
    """Act as a logical problem-solver. Explain reasoning briefly, verify calculation twice, then give final answer in:
'Final Answer: [number]###END###'

Example:
Question: A robe needs 2 blue bolts, half that for white. Total bolts?
Reasoning: White bolts = 2/2 = 1; Total = 2+1=3
Final Answer: 3###END###

Question: {question}""",
    
    # Template 4: 计算优先+格式约束
    """Solve the problem with accurate calculation, then output ONLY the final answer in this format:
'Final Answer: [number]###END###'

Question: {question}""",
    
    # Template 5: 直接输出+格式示例
    """Compute the result accurately and output in the EXACT specified format:
'Final Answer: [number]###END###'

Example: 2+3=? → Final Answer: 5###END###

Question: {question}"""
]

def main():
    parser = argparse.ArgumentParser(description="Math Problem Solving Experiment (Final Optimized)")
    parser.add_argument("--data_path", default="./data/questions.jsonl", help="Path to dataset (JSONL)")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/LLM-Research/Llama-3-8B-Instruct", help="Path to LLM model")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.2, 0.4], help="Alpha values for contrastive decoding")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (8-bit quantization)")
    args = parser.parse_args()

    # 创建输出目录
    output_root = "experiment_results"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"run_{run_id}")
    create_output_dir(output_dir)
    samples_dir = os.path.join(output_dir, "individual_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # 加载模型和数据
    model, tokenizer = load_quantized_model(args.model_path)
    data = load_data(args.data_path)
    print(f"Successfully loaded dataset with {len(data)} samples")

    # 运行实验
    start_time = time.time()
    run_experiment(
        model=model,
        tokenizer=tokenizer,
        data=data,
        alphas=args.alphas,
        templates=ENGLISH_TEMPLATES,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        samples_dir=samples_dir
    )

    total_time = time.time() - start_time
    print(f"Experiment completed! Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()