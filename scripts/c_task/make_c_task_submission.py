from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
C_ROOT = ROOT / "results" / "c_task"
OUT = C_ROOT / "submission_for_D"
FIG = OUT / "figures"
TABLE = OUT / "tables"


def pick(df: pd.DataFrame, name: str) -> pd.Series:
    rows = df[df["experiment_name"] == name]
    if rows.empty:
        raise ValueError(f"Missing experiment row: {name}")
    return rows.iloc[-1]


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


def save_bar(path: Path, data: pd.DataFrame, x: str, y_cols, title: str, ylabel: str) -> None:
    ax = data.set_index(x)[y_cols].plot(kind="bar", figsize=(10, 5), width=0.78)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_single_bar(path: Path, data: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
    ax = data.plot(kind="bar", x=x, y=y, figsize=(10, 5), width=0.72, legend=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_log_bar(path: Path, data: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
    ax = data.plot(kind="bar", x=x, y=y, figsize=(10, 5), width=0.72, legend=False)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25, which="both")
    values = data[y].tolist()
    labels = [f"{v:.2e}" if v < 0.01 else f"{v:.3f}" for v in values]
    ax.bar_label(ax.containers[0], labels=labels, padding=3, fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(C_ROOT / "experiment_registry.csv")

    rows = [
        ("E4 official full dev", pick(registry, "E4_official_doge_eval_causal_dev")),
        ("E4 local 500 full dev", pick(registry, "E4_local_doge_eval_causal_dev_fixed")),
        ("E5 train500 eval500", pick(registry, "E5_small_sample_train_causal_500_fixed")),
        ("E5 train1000 eval1000", pick(registry, "E5_small_sample_train_causal_1000_fixed")),
        ("E6 no contrastive", registry[registry["loss_profile"] == "no_dichotomy_contrastive"].iloc[-1]),
        ("E6 + dichotomy loss", registry[registry["loss_profile"] == "with_dichotomy_loss"].iloc[-1]),
        ("E6 avg pooling", pick(registry, "E6_pooling_ablation_avg_fixed")),
    ]

    summary_records = []
    for label, row in rows:
        summary_records.append({
            "label": label,
            "dataset": row["dataset"],
            "split": row["split"],
            "train_size": row.get("train_size", ""),
            "eval_size": row.get("eval_size", ""),
            "pooling": row.get("pooling_strategy", ""),
            "loss_profile": row.get("loss_profile", ""),
            "DCF": float(row["DCF"]),
            "DCF_percent": pct(row["DCF"]),
            "DCF_positive": float(row["DCF-positive"]),
            "DCF_positive_percent": pct(row["DCF-positive"]),
            "DCF_negative": float(row["DCF-negative"]),
            "DCF_negative_percent": pct(row["DCF-negative"]),
            "pos_neg_degree": float(row["pos_neg_degree"]),
            "checkpoint": row.get("checkpoint", ""),
        })

    summary = pd.DataFrame(summary_records)
    summary.to_csv(TABLE / "c_task_summary_for_D.csv", index=False, encoding="utf-8-sig")

    paper_comparison = pd.DataFrame([
        {
            "label": "Paper BERT baseline",
            "source": "Original paper / README leaderboard",
            "model": "BERT",
            "dataset": "causal_reasoning",
            "split": "leaderboard setting",
            "DCF_percent": 27.17,
            "angle_percent_or_reported": 0.26,
            "comparison_note": "Paper baseline; not a DoGE-trained model.",
        },
        {
            "label": "Paper Ours (BERT)",
            "source": "Original paper / README leaderboard",
            "model": "Ours (BERT)",
            "dataset": "causal_reasoning",
            "split": "leaderboard setting",
            "DCF_percent": 67.59,
            "angle_percent_or_reported": 20.69,
            "comparison_note": "Main paper BERT result; closest reference for the official BERT checkpoint.",
        },
        {
            "label": "Paper Ours (RoBERTa)",
            "source": "Original paper / README leaderboard",
            "model": "Ours (RoBERTa)",
            "dataset": "causal_reasoning",
            "split": "leaderboard setting",
            "DCF_percent": 76.55,
            "angle_percent_or_reported": 5.06,
            "comparison_note": "Paper's strongest causal DCF row; architecture differs from this C-task BERT run.",
        },
        {
            "label": "Our official checkpoint eval",
            "source": "This reproduction",
            "model": "official opposite-score-causal-reasoning-bert",
            "dataset": "causal_reasoning",
            "split": "dev full, 17,944 samples",
            "DCF_percent": 69.77,
            "angle_percent_or_reported": 23.96,
            "comparison_note": "DCF is directly comparable to Paper Ours (BERT); angle is a local evaluator proxy from pos_neg_degree * 100.",
        },
        {
            "label": "Our local 500-sample checkpoint",
            "source": "This reproduction",
            "model": "BERT, 500-sample small training",
            "dataset": "causal_reasoning",
            "split": "dev full, 17,944 samples",
            "DCF_percent": 31.15,
            "angle_percent_or_reported": 0.00013,
            "comparison_note": "Undertrained small-sample control; useful as B's baseline, not as a replacement for the paper model.",
        },
    ])
    paper_comparison.to_csv(TABLE / "paper_causal_comparison_for_D.csv", index=False, encoding="utf-8-sig")

    e6 = summary[summary["label"].isin([
        "E5 train500 eval500",
        "E6 no contrastive",
        "E6 + dichotomy loss",
        "E6 avg pooling",
    ])].copy()
    e6.to_csv(TABLE / "e6_ablation_for_D.csv", index=False, encoding="utf-8-sig")

    e5 = summary[summary["label"].isin([
        "E5 train500 eval500",
        "E5 train1000 eval1000",
    ])].copy()
    e5.to_csv(TABLE / "e5_training_scale_for_D.csv", index=False, encoding="utf-8-sig")

    plot_summary = summary.copy()
    for col in ["DCF", "DCF_positive", "DCF_negative"]:
        plot_summary[col] = plot_summary[col] * 100.0

    save_bar(
        FIG / "c_task_dcf_overview.png",
        plot_summary[plot_summary["label"].isin([
            "E4 official full dev",
            "E4 local 500 full dev",
            "E5 train500 eval500",
            "E5 train1000 eval1000",
        ])],
        "label",
        ["DCF"],
        "C Task DCF Overview",
        "DCF (%)",
    )

    save_bar(
        FIG / "e6_ablation_dcf_pos_neg.png",
        plot_summary[plot_summary["label"].isin([
            "E5 train500 eval500",
            "E6 no contrastive",
            "E6 + dichotomy loss",
            "E6 avg pooling",
        ])],
        "label",
        ["DCF", "DCF_positive", "DCF_negative"],
        "E6 Ablation Metrics",
        "Metric (%)",
    )

    pos_degree = summary[summary["label"].isin([
        "E4 official full dev",
        "E4 local 500 full dev",
        "E5 train500 eval500",
        "E5 train1000 eval1000",
        "E6 no contrastive",
        "E6 + dichotomy loss",
        "E6 avg pooling",
    ])].copy()
    pos_degree.to_csv(TABLE / "pos_neg_degree_values_for_D.csv", index=False, encoding="utf-8-sig")
    save_bar(
        FIG / "pos_neg_degree_comparison_linear.png",
        pos_degree,
        "label",
        ["pos_neg_degree"],
        "Positive-Negative Degree Comparison",
        "pos_neg_degree",
    )
    save_log_bar(
        FIG / "pos_neg_degree_comparison.png",
        pos_degree,
        "label",
        "pos_neg_degree",
        "Positive-Negative Degree Comparison (log scale)",
        "pos_neg_degree (log scale)",
    )

    save_single_bar(
        FIG / "paper_causal_dcf_comparison.png",
        paper_comparison,
        "label",
        "DCF_percent",
        "Causal DCF: Paper Leaderboard vs This Reproduction",
        "DCF (%)",
    )

    angles_path = C_ROOT / "e4_official_eval" / "causal_reasoning" / "dev" / "angles_data.csv"
    if angles_path.exists():
        angles = pd.read_csv(angles_path)
        ax = angles.plot(
            kind="hist",
            bins=60,
            alpha=0.55,
            figsize=(10, 5),
            density=True,
        )
        ax.set_title("Official DoGE Angle-Distance Distributions")
        ax.set_xlabel("cosine distance")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIG / "official_angle_distributions.png", dpi=220)
        plt.close()
        angles.describe().to_csv(TABLE / "official_angle_distribution_stats.csv", encoding="utf-8-sig")

    key_files = {
        C_ROOT / "experiment_registry.csv": "experiment_registry.csv",
        C_ROOT / "c_task_report_zh.md": "c_task_report_zh.md",
        C_ROOT / "run_commands.md": "run_commands.md",
        C_ROOT / "e4_official_eval" / "causal_reasoning" / "dev" / "metrics.csv": "official_full_dev_metrics.csv",
        C_ROOT / "e4_local_eval" / "causal_reasoning" / "dev" / "metrics.csv": "local_500_full_dev_metrics.csv",
    }
    stale_ambiguous_metrics = OUT / "metrics.csv"
    if stale_ambiguous_metrics.exists():
        stale_ambiguous_metrics.unlink()
    for src, dst in key_files.items():
        if src.exists():
            shutil.copy2(src, OUT / dst)

    checklist = """# C 任务完成度清单

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| E0 环境 smoke test | 已完成 | `../e0_smoke/environment_check.md` |
| E4 官方 DoGE 评测 | 已完成 | `official_full_dev_metrics.csv` |
| E5 小样本训练 | 已完成 | `../e5_small_train/causal_reasoning_original_doge_cls_train500_eval500_seed42/checkpoint` |
| E5 训练规模补强 | 已完成 | `tables/e5_training_scale_for_D.csv` |
| E6 loss 消融 | 已完成 | `tables/e6_ablation_for_D.csv` |
| E6 pooling 消融 | 已完成 | `tables/e6_ablation_for_D.csv` |
| E7 原始 DoGE 对照 | 已完成 | `../e5_small_train/causal_reasoning_original_doge_cls_train500_eval500_seed42/checkpoint` |
| 给 D 的图表 | 已完成 | `figures/*.png` |

结论：除 A/B 协作依赖项外，C 的主复现任务已完成。后续仅建议在时间宽裕时继续做 2000 样本或跨数据集扩展。
"""
    (OUT / "completion_checklist.md").write_text(checklist, encoding="utf-8")

    readme = f"""# 给 D 的 C 任务提交包

## 完成度结论

C 的非协作主任务已经完成：E0 smoke test、E4 官方 DoGE 评测、E5 小样本训练、E6 消融、E7 原始 DoGE 对照均已有可复跑结果。

## 最重要结果

- 官方 causal checkpoint 全量 dev：DCF = 69.77%，DCF-positive = 87.20%，DCF-negative = 77.71%。
- 本地 500 样本 checkpoint 全量 dev：DCF = 31.15%。
- 500 样本训练链路可用：DCF = 33.00%。
- E6 消融显示：当前小样本设置下原始 DoGE-style contrastive 配置优于关闭 contrastive、额外打开 dichotomy loss、avg pooling。

## 这些指标怎么解释

- `DCF`：核心指标。它统计在同一 context 下，模型是否把 positive 和 negative 表示得比 positive-neutral、negative-neutral 更远。数值越高，说明模型越能捕捉“条件化的对立关系”。
- `DCF-positive`：只看 positive-neutral 是否比 positive-negative 更近。它反映 positive 这一侧是否被正确拉回到 neutral 附近。
- `DCF-negative`：只看 negative-neutral 是否比 positive-negative 更近。它反映 negative 这一侧是否被正确拉回到 neutral 附近。
- `pos_neg_degree`：positive 和 negative 的平均几何分离程度。官方 checkpoint 的该值更有解释意义；小样本 checkpoint 因训练规模小、epoch 少，该值偏小，汇报时优先看 DCF 趋势。
- `pos_neg_degree_comparison.png` 使用对数纵轴。原因是官方 checkpoint 的 `pos_neg_degree=0.2396`，而小样本实验多在 `1e-6` 到 `1e-4` 量级；如果用普通线性纵轴，小样本柱子会被压到几乎看不见。这不是实验缺失，而是数量级差异。

## 结果说明

| 结果 | 解释 | 可写进 PPT 的结论 |
| --- | --- | --- |
| 官方 checkpoint DCF = 69.77% | 官方模型在 17,944 条 causal dev 样本上能稳定把正反观点分开，DCF-positive 和 DCF-negative 也都较高。 | 官方 Opposite-Score / DoGE 模型在 causal reasoning 场景上复现成功。 |
| 官方角距离分布 | `pos-neg` 平均距离最大，`pos-neutral` 和 `neg-neutral` 更小，符合“正反观点互相远离、中性项相对靠近”的几何假设。 | 几何分布支持论文方法的核心直觉。 |
| 本地 500 样本 checkpoint 全量 dev DCF = 31.15% | 只用 500 条样本训练 1 epoch 后，模型已有一定区分能力，但明显低于官方完整 checkpoint。 | 本地训练链路跑通，但小样本模型不能等同于官方充分训练模型。 |
| E5 500 样本 DCF = 33.00%，1000 样本 DCF = 29.30% | 简单扩大到 1000 样本没有直接提升，说明该设置下性能受训练轮数、采样、loss 权重和超参共同影响。 | 小样本复现实验可用，但进一步提升需要系统调参，不是单纯加样本。 |
| E6 原始 DoGE-style 配置最好 | 原始 contrastive 设置高于关闭 contrastive、额外打开 dichotomy loss、avg pooling。 | 当前 causal 小样本设置下，DoGE-style contrastive 是最关键的有效组件。 |
| avg pooling 低于 cls pooling | 平均池化没有带来提升，说明当前 BERT 小样本设置更依赖 `[CLS]` 表示。 | 后续对照实验建议继续使用 `cls` pooling。 |
| pos_neg_degree 图中官方柱子远高于小样本 | 官方模型经过充分训练，正反样本几何分离明显；小样本模型只训练 1 epoch，虽然 DCF 有一定效果，但几何分离强度还很弱。 | 这张图说明官方模型学到了更强的正反几何结构，小样本实验主要用于流程复现和 baseline。 |

## 和原论文结果对照

原论文 / 官方 README leaderboard 中，Causal Reasoning 相关结果如下：

| 项目 | Causal DCF | Causal Angle | 说明 |
| --- | ---: | ---: | --- |
| Paper BERT baseline | 27.17% | 0.26 | 普通 BERT baseline。 |
| Paper Ours (BERT) | 67.59% | 20.69 | 与本次官方 BERT checkpoint 最接近的论文主结果。 |
| Paper Ours (RoBERTa) | 76.55% | 5.06 | 论文表中 Causal DCF 最高的设置，但 backbone 不同。 |
| This reproduction: official BERT checkpoint | 69.77% | 23.96 proxy | 本次在 causal dev 全量 17,944 条上复现，DCF 比 Paper Ours (BERT) 高 2.18 个百分点。 |
| This reproduction: local 500-sample checkpoint | 31.15% | 0.00013 proxy | 只训练 500 样本 1 epoch 的小样本对照，不能代表论文充分训练结果。 |

对照结论：本次官方 checkpoint 复现结果与论文 `Ours (BERT)` 同量级，并且 DCF 略高，说明 E4 官方 DoGE 评测复现成功；与 `Ours (RoBERTa)` 仍有 6.78 个百分点差距，主要因为本次 C 任务使用的是 causal BERT checkpoint，而不是 RoBERTa 版本。小样本 checkpoint 的作用是给 B 的创新方法提供公平 baseline，不应和论文完整训练结果直接竞争。

注意：本次 DCF 可以和论文表格直接对照；Angle/degree 在当前仓库 evaluator 中以 `pos_neg_degree` 派生记录，和 README 中 `Angle` 的展示口径不完全一致，因此汇报时建议把 Angle 作为几何趋势说明，不作为硬性复现误差。

## 汇报时建议强调

本次 C 任务的核心价值不是把 500 样本小模型训练到官方水平，而是完成了 Windows + RTX 5070 Ti 环境下的完整可复跑链路：官方评测、小样本训练、消融对比、checkpoint 交付和可视化整理。官方 checkpoint 给出主复现结果，小样本实验给 B 的创新方法提供公平对照。

## 图表

- `figures/c_task_dcf_overview.png`：E4 官方、E4 本地、E5 训练规模对比。
- `figures/e6_ablation_dcf_pos_neg.png`：E6 消融 DCF / DCF-positive / DCF-negative 对比。
- `figures/pos_neg_degree_comparison.png`：pos_neg_degree 对数坐标对比，用于避免小样本柱子被官方结果压扁。
- `figures/pos_neg_degree_comparison_linear.png`：pos_neg_degree 线性坐标备份，用于展示官方模型和小样本模型的绝对差距。
- `figures/official_angle_distributions.png`：官方 checkpoint 的三类角距离分布。
- `figures/paper_causal_dcf_comparison.png`：论文 Causal DCF 与本次复现结果对照。

## 表格

- `tables/c_task_summary_for_D.csv`：C 任务核心结果总表。
- `tables/e6_ablation_for_D.csv`：E6 消融表。
- `tables/e5_training_scale_for_D.csv`：E5 训练规模补强表。
- `tables/paper_causal_comparison_for_D.csv`：论文 Causal 结果与本次复现对照表。
- `tables/pos_neg_degree_values_for_D.csv`：pos_neg_degree 原始数值表。
- `tables/official_angle_distribution_stats.csv`：官方角距离分布统计。
- `official_full_dev_metrics.csv`：官方 checkpoint 全量 dev 指标。
- `local_500_full_dev_metrics.csv`：本地 500 样本 checkpoint 全量 dev 备用指标。

## 建议 PPT 使用

1. 主结果页：使用 `c_task_dcf_overview.png`。
2. 消融页：使用 `e6_ablation_dcf_pos_neg.png`。
3. 论文对照页：使用 `paper_causal_dcf_comparison.png`。
4. 几何解释页：使用 `official_angle_distributions.png`。
5. 细节备份：引用 `tables/c_task_summary_for_D.csv` 和 `tables/paper_causal_comparison_for_D.csv`。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
