import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from glob import glob

plt.rcParams["axes.unicode_minus"] = False

def load_total_results(results_path: str) -> pd.DataFrame:
    """加载总结果CSV文件"""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"总结果文件不存在: {results_path}")
    return pd.read_csv(results_path)

def analyze_single_sample(samples_dir: str, sample_id: str = None) -> None:
    """分析单个样本"""
    sample_files = glob(os.path.join(samples_dir, "*.json"))
    if not sample_files:
        print("未找到单样本文件")
        return

    if sample_id:
        target_files = [f for f in sample_files if f"sample_{sample_id}_" in f]
        if not target_files:
            print(f"未找到样本ID {sample_id} 的文件")
            return
        sample_file = target_files[0]
    else:
        sample_file = sample_files[0]

    with open(sample_file, "r", encoding="utf-8") as f:
        sample = json.load(f)
    print("\n===== 单样本详情 =====")
    print(f"样本ID: {sample['sample_id']}")
    print(f"Alpha值: {sample['alpha']}")
    print(f"模板编号: {sample['template_idx']}")
    print(f"问题: {sample['question']}")
    print(f"模型响应: {sample['response'][:300]}...")
    print(f"提取答案: {sample['extracted_answer']}")
    print(f"参考答案: {sample['reference_answer']}")
    print(f"是否正确: {'是' if sample['is_correct'] else '否'}")

def plot_accuracy_by_alpha(results_df: pd.DataFrame, output_dir: str) -> None:
    """按alpha值绘制准确率图"""
    alpha_acc = results_df.groupby("alpha")["is_correct"].mean().reset_index()
    alpha_acc["accuracy"] = alpha_acc["is_correct"] * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(x="alpha", y="accuracy", data=alpha_acc, palette="viridis")
    plt.title("Accuracy by Alpha Value", fontsize=14)
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    for i, row in enumerate(alpha_acc.itertuples()):
        plt.text(i, row.accuracy + 1, f"{row.accuracy:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_alpha.png"), dpi=300)
    print(f"Alpha值准确率图保存至: {os.path.join(output_dir, 'accuracy_by_alpha.png')}")

def plot_accuracy_by_template(results_df: pd.DataFrame, output_dir: str) -> None:
    """按模板绘制准确率图"""
    template_acc = results_df.groupby("template")["is_correct"].mean().reset_index()
    template_acc["accuracy"] = template_acc["is_correct"] * 100

    plt.figure(figsize=(12, 6))
    sns.barplot(x="template", y="accuracy", data=template_acc, palette="magma")
    plt.title("Accuracy by Template", fontsize=14)
    plt.xlabel("Template Index", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    for i, row in enumerate(template_acc.itertuples()):
        plt.text(i, row.accuracy + 1, f"{row.accuracy:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_template.png"), dpi=300)
    print(f"模板准确率图保存至: {os.path.join(output_dir, 'accuracy_by_template.png')}")

def plot_alpha_vs_template(results_df: pd.DataFrame, output_dir: str) -> None:
    """绘制热力图"""
    pivot = results_df.pivot_table(
        index="template",
        columns="alpha",
        values="is_correct",
        aggfunc="mean"
    )
    pivot = pivot * 100

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Accuracy (%)"})
    plt.title("Accuracy Heatmap: Alpha vs Template", fontsize=14)
    plt.xlabel("Alpha Value", fontsize=12)
    plt.ylabel("Template Index", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alpha_vs_template_heatmap.png"), dpi=300)
    print(f"热力图保存至: {os.path.join(output_dir, 'alpha_vs_template_heatmap.png')}")

def main():
    parser = argparse.ArgumentParser(description="实验结果分析与可视化")
    parser.add_argument("--experiment_dir", required=True, help="实验结果目录")
    parser.add_argument("--sample_id", help="可选：指定样本ID")
    args = parser.parse_args()

    total_results_path = os.path.join(args.experiment_dir, "total_results.csv")
    samples_dir = os.path.join(args.experiment_dir, "individual_samples")
    if not os.path.exists(samples_dir):
        raise NotADirectoryError(f"单样本目录不存在: {samples_dir}")

    analysis_dir = os.path.join(args.experiment_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 加载结果并打印概览
    print("===== 实验概览 =====")
    results_df = load_total_results(total_results_path)
    overall_acc = results_df["is_correct"].mean() * 100
    print(f"总样本数: {len(results_df)}")
    print(f"整体准确率: {overall_acc:.2f}%")
    print(f"测试的Alpha值: {sorted(results_df['alpha'].unique())}")
    print(f"测试的模板数量: {len(results_df['template'].unique())}")

    # 分析单个样本
    analyze_single_sample(samples_dir, args.sample_id)

    # 生成图表
    plot_accuracy_by_alpha(results_df, analysis_dir)
    plot_accuracy_by_template(results_df, analysis_dir)
    plot_alpha_vs_template(results_df, analysis_dir)

    print("\n===== 分析完成 =====")
    print(f"分析结果和图表保存至: {analysis_dir}")

if __name__ == "__main__":
    main()