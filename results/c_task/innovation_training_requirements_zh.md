# E7 创新训练需求评估

## 1. 结论

README 中的 E7 创新方案在计划之内，但它不要求一开始训练论文级别模型。

建议路线是：先做小样本公平对照，再根据结果决定是否扩大训练规模。

| 问题 | 结论 |
| --- | --- |
| 是否在计划内 | 是。E7 明确要求至少一个新增算法组件，并与原始 DoGE 小样本训练对照。 |
| 是否需要本地训练 | 需要。Dichotomy Axis Loss、Hard Negative Mining、Adaptive Loss Reweighting 都会改变训练目标或训练 batch 逻辑，不能只靠官方 checkpoint 评测完成。 |
| 是否需要本地训到论文级别 | 不需要。论文级结果已经由官方 checkpoint 复现，E7 需要的是同设置下创新方法是否优于原始 DoGE baseline。 |
| C 的职责 | 提供原始 DoGE 对照结果，并协助 B 用同样数据、seed、batch size、epoch 跑创新训练。 |
| B 的职责 | 实现创新 loss / hard mining，并登记创新实验结果。 |

## 2. 各创新方案的训练需求

| 方案 | 是否必须重新训练 | 代码改动位置 | 训练成本 | 推荐优先级 |
| --- | --- | --- | --- | --- |
| Dichotomy Axis Loss | 必须 | `oppositescore/trainer/loss.py` 的 DatasetFormats.D 分支 | 低到中 | 最高 |
| Geometry-aware Hard Negative Mining | 必须 | loss 计算或 batch 内 pair mining 逻辑 | 中 | 第二 |
| Geometry-Adaptive Loss Reweighting | 必须 | loss 权重动态调节 | 低 | 备选 |
| Context Dropout | 必须 | 数据 tokenization 或 collator 阶段 | 低到中 | 备选 |

## 3. 推荐实验规模

第一阶段只需要小样本，不建议直接全量训练。

| 阶段 | 目的 | 建议配置 | 是否必须 |
| --- | --- | --- | --- |
| P0 sanity | 检查创新 loss 不报错、loss 有梯度 | train 50 到 100，eval 100 | 必须 |
| P1 主创新对照 | 判断创新是否优于原始 DoGE | train 500，eval 500，seed 42，epoch 1 | 必须 |
| P2 稳定性补强 | 看趋势是否稳定 | train 1000，eval 1000，seed 42，epoch 1 | 建议 |
| P3 hard subset | 检验创新是否改善困难样本 | 等 A 给 hard subset 后跑 | 建议 |
| P4 更大规模 | 争取更强结果 | train 2000 或更多，epoch 1 到 3 | 可选 |
| P5 论文级本地训练 | 从零接近官方 checkpoint | 全量训练、多 epoch、多超参 | 不建议作为当前目标 |

## 4. 公平对照设置

创新实验必须和原始 DoGE 对照保持以下条件一致：

- dataset：`causal_reasoning`
- train size：优先 500，其次 1000
- eval size：优先 500，其次 1000
- seed：42
- backbone：`google-bert/bert-base-uncased`
- pooling：`cls`
- batch size：8
- gradient accumulation：4
- epoch：1
- 评测指标：DCF、DCF-positive、DCF-negative、pos_neg_degree

当前已经有原始 DoGE 对照：

- train/eval 500：DCF = 0.3300
- train/eval 1000：DCF = 0.2930

因此 E7 的最低成功标准是：创新方法在相同 500 样本设置下 DCF 高于 0.3300，或者在 hard subset / context perturbation 上表现出更清晰优势。

## 5. 为什么不建议现在训练论文级模型

本地训练论文级模型理论上可行，但不是当前 E7 的必要条件。

原因：

- 官方 checkpoint 已经复现到 DCF = 69.77%，已覆盖论文级主复现。
- 当前 500/1000 小样本训练主要用于验证训练链路和创新对照。
- 从零训练到论文级别需要全量数据、多 epoch、多超参搜索，时间风险较高。
- 创新实验更重要的是公平比较，而不是先把 baseline 训练到官方 checkpoint 水平。

## 6. 建议给组内的表述

可以这样和 B/D 对齐：

> E7 创新需要本地训练，因为创新组件会改变 loss 或 batch 内样本关系。当前不需要从零训练论文级模型；论文级主结果已经由官方 checkpoint 完成复现。创新实验建议先在 train/eval 500 的原始 DoGE baseline 上做公平对照，若 DCF 或 hard subset 表现有提升，再扩大到 1000/2000 样本。

