# C 任务完成度清单

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
