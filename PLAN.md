# Conditional Dichotomy Quantification 复现与创新协作文档

本文档面向 4 人小组协作，目标是在 2026-06-07 前完成论文复现、结果验证、创新实验和最终汇报材料。

## 1. 项目理解

论文：Conditional Dichotomy Quantification via Geometric Embedding, ACL 2025 oral.

核心问题不是普通语义相似度，而是条件二分量化。给定同一个上下文 `Z`，衡量两个输出 `X` 和 `Y` 在该上下文下有多“对立”。传统 STS 模型常把“不相似”当作“对立”，但论文的动机例子说明这不可靠：两个支持者之间的语义相似度可能低于支持者与反对者之间的相似度。

论文提出的任务是 ConDQ，典型数据结构为：

- `context`: 共同上下文，例如辩题、前提-假设、原因-结果。
- `positive/supporter`: 支持该上下文关系的文本。
- `negative/defeater`: 削弱或反对该上下文关系的文本。
- `neutral`: 与正反都不形成强对立的中性文本。

论文方法 DoGE 的核心是用几何嵌入表达正反关系，重点让 `positive` 与 `negative` 在嵌入空间中形成更强角距离，同时让 `neutral` 在几何关系上更接近中间参照。评测主要看：

- `DCF`: Dichotomy Consistency Frequency，判断 `pos-neutral` 和 `neg-neutral` 是否都比 `pos-neg` 更近。
- `DCF-positive` 与 `DCF-negative`: 分别检查正侧和负侧的一致性。
- `Oppo-Angle` 或代码中的 `pos_neg_degree`: 正反文本之间的几何对立强度。

## 2. 当前代码理解

关键入口：

- `motivation_code_main.py`: 复现论文动机，展示普通 embedding/STS 在正反判断上的失败案例。
- `training_dichotomybert_train_main.py`: 训练 DichotomyE/DoGE 风格模型的主脚本。
- `test_dichotomy_test_main.py`: 加载模型并在数据集上计算 DCF 等指标。
- `simulation_main.py`: 随机角度模拟，用来理解 DCF 的几何概率和随机基线。
- `oppositescore/model/dichotomye.py`: DichotomyE 模型封装，负责加载 backbone、tokenizer、pooling、训练与推理。
- `oppositescore/trainer/dichotomy_dataset.py`: 将 `context/positive/negative/neutral` 拼成模型输入。
- `oppositescore/trainer/loss.py`: 训练损失，尤其是 DatasetFormats.D 下的 dichotomy loss 和 dichotomy contrastive loss。
- `oppositescore/evaluation/evaluation.py`: 评测 DCF、DCF-positive、DCF-negative 与角度类指标。

数据目录：

- `data/debate`: Perspectrum debate 数据。
- `data/defeasible_nli`: Defeasible NLI 数据。
- `data/causal_reasoning`: Delta causal reasoning 数据。

已发现的复现风险：

- 训练和测试脚本里的目录名是 `data/perspectrum/`、`data/defeasible_snli/`、`data/delta_causal/`，但仓库实际目录是 `data/debate/`、`data/defeasible_nli/`、`data/causal_reasoning/`。
- Hugging Face 缓存路径写死为 Linux 路径 `/mnt/lia/scratch/scui/cached_files/hf`，在当前 Windows 环境会失败，需要改为可配置参数或本地目录。
- Defeasible NLI 的 test 文件 README 说明没有完整上传，短期复现应优先使用 dev 集或已上传文件。
- README 和部分源码注释存在编码乱码，不影响核心实验，但最终报告截图和文档中要自行修正。
- `evaluation.py` 会把 `angles_data.csv` 写到当前目录，批量实验时需要改成可配置输出路径，避免覆盖。

## 3. 实验清单

本项目按实验推进，而不是按抽象模块推进。

### E0：环境与数据 Smoke Test

目的：确认每个人的机器都能跑最小流程，避免 B 跑通后 C/D 在自己机器上重新踩坑。

内容：

- 安装依赖并能 import `oppositescore`。
- 读取三类数据中至少一个 jsonl 文件。
- 用 10 条样本跑通 tokenizer、collator、evaluation。
- 记录每台机器的 GPU、CUDA、PyTorch、可用显存和推荐 batch size。

最低产出：

- `environment_check.md`
- 每人一条 smoke test 运行记录
- 常见报错和解决方法

截止：2026-05-27

### E1：论文动机复现实验

目的：证明普通 STS/embedding 相似度不能直接等价于“条件对立”。

内容：

- 跑通 `motivation_code_main.py` 或等价脚本。
- 用 3 到 5 个样例展示：两个支持文本可能相似度较低，支持与反对文本反而可能相似度较高。
- 写清楚这个实验为什么引出 ConDQ 和 DoGE。

最低产出：

- 一张动机案例表
- 一段报告文字
- 可放入 PPT 的 1 页图示

截止：2026-05-28

### E2：数据与指标验证实验

目的：保证后续训练和评测使用的数据字段、split、DCF 指标都是正确的。

内容：

- 统计 `data/debate`、`data/defeasible_nli`、`data/causal_reasoning` 的可用文件。
- 抽样检查 `context_text/supporter_text/defeater_text/neutral_text` 到 `context/positive/negative/neutral` 的映射。
- 人工检查至少 20 条样本，确认 positive、negative、neutral 的语义角色。
- 对 `DichotomyEvaluator` 的 DCF 逻辑写一页解释。

最低产出：

- `data_report.md`
- `metric_report.md`
- 20 条样本检查记录
- 训练/评测可用 split 建议

截止：2026-05-29

### E2.5：数据质量与困难样本子集构造

目的：让数据负责人不仅做说明，还产出可直接服务消融和创新实验的数据子集。

内容：

- 从每个可用数据集中抽取固定小样本子集，例如 500、1000、2000 条，供 E5/E6/E7 使用。
- 构造一个 clean subset：选择字段完整、语义角色清晰、长度不过长的样本，作为 smoke test 和稳定训练子集。
- 构造一个 hard subset：选择普通 embedding 难以区分、positive/negative/neutral 语义边界模糊或人工判断困难的样本。
- 给每个子集写清楚来源、筛选规则、样本数和适用实验。

最低产出：

- `data_splits_report.md`
- clean subset 清单
- hard subset 清单
- 每个子集的字段检查和样例说明

截止：2026-05-30

### E3：普通 Embedding Baseline 实验

目的：建立复现对照组，说明普通 embedding 模型在条件二分任务上不足。

内容：

- 至少选择 1 到 2 个 baseline：BERT、SBERT、UAE/AnglE、SimCSE 中任选。
- 在至少一个数据集 dev/test 上计算 `DCF`、`DCF-positive`、`DCF-negative`、`pos_neg_degree`。
- 如果显存允许，三个数据集都跑同一 baseline，形成跨领域 baseline 表。

最低产出：

- baseline 结果表
- 运行命令
- 每个 baseline 的一句话结论

截止：2026-05-31

### E3.5：统一评测脚本与实验结果登记

目的：把各成员的实验输出统一成同一种格式，避免最后无法合表。

内容：

- 设计统一结果字段：实验名、负责人、数据集、split、模型、checkpoint、seed、batch size、显存、DCF、DCF-positive、DCF-negative、pos_neg_degree、备注。
- 给 E3/E4/E6/E7/E8/E9 都提供同一份结果登记模板。
- 整理所有实验命令，保证报告中可以复现。
- 检查不同成员提交的结果是否缺少 seed、数据 split 或模型来源。

最低产出：

- `experiment_registry.csv`
- `run_commands.md`
- 结果字段规范
- 每日实验登记检查

截止：2026-06-04

### E4：官方 Opposite-Score / DoGE 评测复现

目的：复现论文主方法相对 baseline 的优势。

内容：

- 使用官方 checkpoint 或本仓库 `DichotomyE` 加载预训练模型。
- 在至少一个数据集上跑完整评测。
- 优先复现 README leaderboard 中一个场景的相对结论。
- 若 Defeasible NLI test 缺失，则说明改用 dev 或抽样评测。

最低产出：

- DoGE/Opposite-Score 结果表
- 与 E3 baseline 的对比表
- 一段复现结论

截止：2026-06-01

### E5：小样本训练复现实验

目的：证明小组能跑通训练链路，而不是只调用预训练模型。

内容：

- 从一个数据集抽样 500、1000 或 2000 条训练样本。
- 跑通 `DichotomyE.fit()`。
- 保存 checkpoint、训练日志和 dev 指标。
- 记录不同机器上的可行 batch size、gradient accumulation 和训练时间。

最低产出：

- 小样本训练结果
- 训练命令和配置
- checkpoint 路径说明

截止：2026-06-02

### E6：主复现消融实验

目的：说明论文方法里的关键损失项或设置确实有作用。

内容：

- 在小样本或可承受数据规模上做 2 到 3 个消融。
- 推荐消融项：`dichotomy_contrastive_w`、`dichotomy_w`、`ibn_w`、pooling strategy。
- 每次只改一个变量。

最低产出：

- 消融表格
- 对指标变化的解释
- 失败或异常结果记录

截止：2026-06-04

### E7：算法创新实验

目的：形成真正能写进报告的方法创新，而不是只做外围分析。

主方案：Axis-aware Hard Dichotomy Learning

- Dichotomy Axis Loss：显式构建 `positive-negative` 条件对立轴，让 neutral 在该轴上的投影更靠近中间区域。
- Geometry-aware Hard Negative Mining：从 batch 内挖掘最破坏 DCF 几何关系的困难负例。

备选方案：

- Geometry-Adaptive Loss Reweighting：根据当前 batch 的几何 gap 动态调整 contrastive/dichotomy loss 权重。
- Context Dropout：训练时随机删除或 mask 部分 context，测试弱 context 鲁棒性。

最低产出：

- 一个新增算法组件
- 与原始 DoGE 小样本训练的对照结果
- 至少一个消融：无创新组件 vs 有创新组件

截止：2026-06-04

### E8：上下文敏感性与错误分析实验

目的：增强报告解释力，证明模型是否真的依赖 context。

内容：

- 正常 context 评测。
- 随机打乱 context 评测。
- 可选：去掉 context 或保留一半 context 评测。
- 分析至少 20 个错误样例。

最低产出：

- context sensitivity 结果表
- 角度分布图或 embedding 可视化
- 错误案例分析

截止：2026-06-05

### E9：跨数据集泛化与稳健性实验

目的：让 baseline 负责人也产出一组可讨论的实验结论，并检查方法是否只在单一领域有效。

内容：

- 使用同一个 baseline 模型在 debate、defeasible NLI、causal reasoning 上统一评测。
- 使用同一个 DoGE/Opposite-Score checkpoint 在多个数据集上评测，若模型 checkpoint 是领域专用的，需要明确标注来源领域。
- 对比不同领域的 `DCF`、`DCF-positive`、`DCF-negative` 和 `pos_neg_degree`。
- 分析哪些领域最难、哪些指标下降最明显。

最低产出：

- 跨数据集结果表
- 领域差异分析
- 对后续创新实验选择数据集的建议

截止：2026-06-04

## 4. 按实验分工

### 成员 A：E1 + E2 + E2.5 + E8 的数据质量与错误分析负责人

主责实验：

- E1：论文动机复现实验
- E2：数据与指标验证实验
- E2.5：数据质量与困难样本子集构造
- E8：上下文敏感性与错误分析实验

具体任务：

- 写清楚论文任务、数据字段和 DCF 指标。
- 检查样本语义角色是否正确。
- 为 C/D 产出 clean subset 和 hard subset，避免训练和创新实验临时抽样不一致。
- 设计 E8 的 context 扰动方式：正常 context、随机 context、无 context 或弱 context。
- 复用 B/C 的评测命令，独立跑出 E8 的结果表。
- 做至少 20 条错误样例分析，并解释哪些错误可能被创新方法改善。

不主责：

- 不负责训练脚本实现。
- 不负责主模型 checkpoint 训练，但需要使用 C 产出的模型做 E8 评测。
- 不负责维护全组实验登记表，这属于 B。
- 不负责最终图表设计，这属于 D。

最终交付：

- `data_report.md`
- `metric_report.md`
- `data_splits_report.md`
- clean subset 与 hard subset
- 动机案例表
- context sensitivity 结果表
- 错误样例分析

### 成员 B：E0 + E3 + E3.5 + E7 + E9 的流水线与算法创新负责人

主责实验：

- E0：环境与数据 smoke test
- E3：普通 embedding baseline
- E3.5：统一评测脚本与实验结果登记
- E7：算法创新实验的实现与实验登记
- E9：跨数据集泛化与稳健性实验

具体任务：

- 把路径、缓存、数据读取和输出目录做成可配置流程。
- 提供统一运行命令，让不同机器都能复用。
- 维护实验日志模板。
- 跑 baseline，并整理 baseline 结果表。
- 维护 `experiment_registry.csv`，每天检查所有实验结果字段是否完整。
- 维护 `run_commands.md`，保证 A/C/D 的评测和训练命令能被复用。
- 实现 E7 的算法创新组件，优先实现 Dichotomy Axis Loss，其次尝试 Geometry-aware Hard Negative Mining。
- 与 C 对接训练配置，保证创新方法与原始 DoGE 使用同数据、同 seed、同训练规模。
- 统一跑至少一个 baseline 在三个数据集上的结果。
- 汇总 DoGE/Opposite-Score 的多数据集评测结果，形成跨领域对比表；领域语义解释由 A 协助，图表呈现由 D 协助。

不主责：

- B 跑通不代表其他人不用跑。B 负责流程可迁移，C/D 仍然要在自己机器上跑 smoke test。
- B 不负责主模型大规模训练，训练资源和原始对照由 C 协助。
- B 不负责解释所有错误样例，领域语义解释交给 A。
- B 不负责制作 clean/hard 数据子集，这属于 A。
- B 不负责最终算法图示和 PPT 包装，这属于 D。

最终交付：

- 环境检查说明
- baseline 结果表
- 跨数据集泛化结果表
- 创新 loss / hard mining 实现说明
- 创新实验运行命令和结果登记
- `experiment_registry.csv`
- `run_commands.md`
- 统一命令清单
- 常见报错表

### 成员 C：E4 + E5 + E6 + E7 训练对照的主复现负责人

主责实验：

- E4：官方 Opposite-Score / DoGE 评测复现
- E5：小样本训练复现实验
- E6：主复现消融实验
- E7：原始 DoGE 对照训练和创新训练协助

具体任务：

- 在自己的机器上复用 B 的流程跑 smoke test。
- 跑官方模型评测，形成与 baseline 的主对比。
- 跑小样本训练，确认训练链路可用。
- 做 loss 和 pooling 消融。
- 为 B 的创新实验提供同数据、同 seed、同训练规模的原始 DoGE 对照，并协助跑创新训练。

不主责：

- 不需要替所有人解决环境问题，但要把自己机器上的训练配置反馈给 B。
- 不负责最终 PPT 视觉整理。

最终交付：

- 主复现结果表
- 小样本训练记录
- 消融实验表
- 创新实验原始对照结果
- checkpoint 和日志说明

### 成员 D：全部结果可视化 + 最终汇报负责人

主责实验：

- E3/E4/E6/E8/E9 的图表整理和最终汇报
- E7：算法创新结果的可视化与报告表达

具体任务：

- 不主责实现创新 loss，但需要把 B/C 的创新实验结果整理成清晰的方法图、对比表和消融图。
- 画角度分布、embedding 可视化、context sensitivity 图和跨数据集对比图。
- 整合 PPT 和最终报告。

不主责：

- 不负责创新方法代码实现，B 负责实现，C 负责训练对照。
- 不负责 baseline 大表的维护。
- 不负责 E8 的错误样例语义解释，A 负责解释，D 负责展示。

最终交付：

- 创新方法图和创新实验可视化
- 可视化图
- PPT 初稿和演示稿
- 创新方法说明

## 5. 时间排期

### 2026-05-25 至 2026-05-27：E0/E1/E2 启动

目标：全员完成环境 smoke test，明确数据、指标和论文动机。

- A：完成 E1 初稿和 E2 数据字段检查。
- B：完成 E0 流程和统一命令草案。
- C：在自己机器上跑 E0，并阅读 loss/trainer。
- D：在自己机器上跑 E0，并准备最终可视化模板。

检查点：2026-05-27 晚上，每人必须有一条可复现运行记录。

### 2026-05-28 至 2026-05-31：E3/E4 基础复现

目标：至少一个数据集完成 baseline 与 DoGE/Opposite-Score 对比。

- A：完成 E2 报告和 E1 动机案例。
- A：完成 E2.5 clean/hard subset。
- B：完成 E3 baseline，并建立 E3.5 结果登记表。
- B：启动 E9 跨数据集评测。
- C：完成 E4 官方模型评测。
- B：确定 E7 创新实现路线。
- D：准备 E7/E8 可视化模板。

检查点：2026-05-31 晚上，必须有 baseline vs DoGE 的第一张结果表。

### 2026-06-01 至 2026-06-04：E5/E6/E7 核心实验

目标：完成小样本训练、主消融和至少一个算法创新实验。

- A：完成 E8 上下文扰动评测和错误样例分析。
- B：完成 E9 跨数据集泛化实验，并维护 E3.5 最终实验日志。
- B：完成 E7 算法创新实现，并提交与 C 对接后的创新实验记录。
- C：完成 E5 和 E6。
- D：完成 E7/E8/E9 的可视化初稿，并和 B/C 核对结果表达。

检查点：2026-06-04 晚上，定稿所有核心实验结果。

### 2026-06-05 至 2026-06-07：E8 与最终交付

目标：完成上下文敏感性分析、可视化、报告和 PPT。

- 2026-06-05：完成 E8、报告文字和表格。
- 2026-06-06：完成 PPT 初稿、演示脚本和结果核对。
- 2026-06-07：最终提交，保留半天处理环境或结果异常。

## 6. 每日协作规则

- 每个实验都必须记录命令、模型、数据、seed、输出文件。
- 所有图表都保留原始数据，不只保存截图。
- 遇到 GPU 或网络问题时，优先缩小数据集跑通流程。
- 不在最后两天新增高风险创新，只做整理和修正。

## 7. 最终交付物

- 复现报告：论文理解、代码说明、实验设置、结果表格、消融、创新、错误分析。
- PPT：10 到 15 页，突出任务动机、方法直觉、复现结果和创新亮点。
- 实验结果目录：保存 csv、log、图表和必要 checkpoint 说明。
- 创新实验说明：说明创新点、实现方式、结果和局限。

## 8. 参考资料

- ACL Anthology 论文页：https://aclanthology.org/2025.acl-long.383/
- 论文 PDF：https://aclanthology.org/2025.acl-long.383.pdf
- 官方代码仓库：https://github.com/cui-shaobo/conditional-dichotomy-quantification
