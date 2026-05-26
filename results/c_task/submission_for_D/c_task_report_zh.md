# 成员 C 复现与训练任务中文报告

## 1. 任务范围

本次工作面向成员 C 负责的内容，主要包括：

- E0：在本机复用流程完成环境与数据 smoke test。
- E4：官方 Opposite-Score / DoGE 评测复现。
- E5：小样本训练复现实验。
- E6：主复现消融实验。
- E7：为后续创新实验提供原始 DoGE 对照训练结果。

按照要求，本次暂时不做创新算法协作内容，只完成 C 需要的复现、训练、消融和可复跑交付。

## 2. 本机环境配置结果

最终新建并使用的 conda 环境为：

```powershell
conda activate condq-c
```

环境关键信息如下：

| 项目 | 结果 |
| --- | --- |
| 操作系统 | Windows 10 / 10.0.22631 |
| Python | 3.10.20 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| 显存 | 约 15.9 GB |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Hugging Face 缓存目录 | `D:\nlp\conditional-dichotomy-quantification\hf_cache` |

普通 Python 依赖按照要求主要使用清华源：

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-c-windows.txt
```

但清华 PyPI 主源安装到的 `torch` 是 CPU-only 版本，无法使用 RTX 5070 Ti。因此 GPU 版 PyTorch 必须改用 PyTorch 官方 CUDA 12.8 wheel 源：

```powershell
python -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

## 3. 遇到的问题与解决方式

### 3.1 初始工作区只有协作文档

一开始 `D:\nlp` 下只有 `README.md`，没有官方代码、数据或 git 仓库。因此先从官方仓库克隆：

```powershell
git clone https://github.com/cui-shaobo/conditional-dichotomy-quantification.git conditional-dichotomy-quantification
```

克隆后确认官方数据文件已包含在仓库中。

### 3.2 README 在 PowerShell 中显示乱码

PowerShell 默认编码显示导致中文 README 出现乱码。实际文件是 UTF-8，使用显式编码读取即可：

```powershell
Get-Content -Encoding UTF8 README.md
```

该问题不影响代码运行。

### 3.3 官方脚本存在 Linux 硬编码缓存路径

官方脚本中写死了：

```python
/mnt/lia/scratch/scui/cached_files/hf
```

该路径在 Windows 下不可用。本次新增统一脚本，将 Hugging Face 缓存改为项目本地：

```text
D:\nlp\conditional-dichotomy-quantification\hf_cache
```

同时设置：

```python
HF_HOME
TRANSFORMERS_CACHE
HF_HUB_DISABLE_SYMLINKS_WARNING
```

### 3.4 官方训练/测试脚本数据路径与实际目录不一致

官方脚本使用旧目录名：

- `data/perspectrum/`
- `data/defeasible_snli/`
- `data/delta_causal/`

仓库实际目录为：

- `data/debate/`
- `data/defeasible_nli/`
- `data/causal_reasoning/`

本次新增 `scripts/condq_common.py` 统一管理数据集映射，避免继续手工改路径。

### 3.5 Windows 下 bitsandbytes 不是必要依赖但会导致 import 风险

官方 `dichotomye.py` 和 `angle.py` 顶部直接 import `bitsandbytes`。Windows 上如果不做 4/8-bit LLM 量化，其实不需要它。

已修复为可选导入：

- 普通 BERT / RoBERTa 训练不需要 bitsandbytes。
- 只有使用 4/8-bit LLM 量化时才会显式报错要求安装 bitsandbytes。

### 3.6 新版 Transformers 调用 Trainer.compute_loss 参数变化

当前安装的 `transformers 4.57.6` 会向 `compute_loss` 传入额外参数，例如 `num_items_in_batch`。

官方 `DichotomyTrainer.compute_loss` 原签名不兼容，已改为：

```python
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
```

### 3.7 PyTorch 不允许修改非叶子变量的 requires_grad

官方 trainer 中有：

```python
loss.requires_grad = True
```

新版 PyTorch 下会报错：

```text
RuntimeError: you can only change requires_grad flags of leaf variables.
```

该行不是必要逻辑，已删除，不改变损失本身。

### 3.8 消融配置可能导致 loss 为 Python float

最初尝试完全关闭 D 格式下有效损失项时，loss 返回 Python float，训练时报错：

```text
AttributeError: 'float' object has no attribute 'backward'
```

处理方式：

- 在 `DichotomyLoss` 中增加零张量兜底，避免无损失配置直接崩溃。
- 将 E6 消融设计改为可训练配置：保留 `dichotomy_w=1.0`，关闭 `dichotomy_contrastive_w=0.0`。

### 3.9 evaluation.py 默认覆盖 angles_data.csv

官方 evaluator 会把角度数据固定写到当前目录：

```text
angles_data.csv
```

批量实验时会互相覆盖。已改为可配置 `angles_output_path`，每次实验写入自己的结果目录。

### 3.10 Hugging Face 官方 checkpoint 下载先失败，后续已解决

最初尝试下载官方模型：

- `shaobocui/opposite-score-causal-reasoning-bert`
- `shaobocui/opposite-score-debate-bert`

均遇到：

```text
429 Client Error: Too Many Requests
```

也尝试过：

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
```

但镜像站出现 SSL / connection reset 错误。因此第一次自动下载官方 checkpoint 时 E4 未能完成。

后续你已成功将官方模型下载到本地目录：

```text
hf_cache\official\opposite-score-causal-reasoning-bert
```

并使用本地 checkpoint 路径完成 E4 官方模型评测：

```powershell
conda run -n condq-c python scripts\run_doge_eval.py --scenario causal_reasoning --split dev --model hf_cache\official\opposite-score-causal-reasoning-bert --batch-size 128 --experiment-name E4_official_doge_eval_causal_dev
```

### 3.11 基础 BERT 下载中断

下载 `google-bert/bert-base-uncased` 时一度出现：

```text
IncompleteRead
```

处理方式：

- 安装 `hf_xet`。
- 使用 Hugging Face 缓存续传。
- 最终 `bert-base-uncased` 成功下载并用于训练。

## 4. 已新增或修改的主要文件

### 4.1 新增脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/condq_common.py` | 统一数据路径、缓存、加载、评测和实验登记 |
| `scripts/run_e0_smoke.py` | E0 环境和数据 smoke test |
| `scripts/run_doge_eval.py` | E4 DoGE / Opposite-Score 评测入口 |
| `scripts/run_small_train.py` | E5 小样本训练入口 |
| `scripts/run_ablation.py` | E6 消融实验入口 |

### 4.2 修改官方代码

| 文件 | 修改内容 |
| --- | --- |
| `oppositescore/model/dichotomye.py` | bitsandbytes 可选导入；fp16 逻辑修正 |
| `oppositescore/model/angle.py` | bitsandbytes 可选导入 |
| `oppositescore/trainer/dichotomye_trainer.py` | 兼容新版 Trainer；删除非法 `loss.requires_grad = True` |
| `oppositescore/trainer/loss.py` | loss 为 float 时返回可反传零张量兜底 |
| `oppositescore/evaluation/evaluation.py` | angles CSV 输出路径可配置 |

### 4.3 新增环境依赖文件

```text
requirements-c-windows.txt
```

该文件用于 Windows 下 C 任务复现，避免原始 `requirements.txt` 在 Windows 上强依赖 bitsandbytes。

## 5. 实验完成情况

### 5.1 E0 环境与数据 Smoke Test

已完成。

输出文件：

```text
results/c_task/e0_smoke/environment_check.md
results/c_task/e0_smoke/smoke_result.json
```

Smoke test 使用 10 条 causal reasoning dev 样本，跑通了：

- `oppositescore` import
- 数据读取
- tokenizer
- collator
- evaluator
- GPU 环境识别

Smoke test 中 tiny random BERT 的 DCF 为：

```text
DCF = 0.1000
```

该值只用于验证流程，不作为模型结论。

### 5.2 E5 小样本训练复现实验

已完成。

基础设置：

| 项目 | 值 |
| --- | --- |
| 数据集 | causal reasoning |
| 训练 split | train |
| 评测 split | dev |
| 训练样本数 | 500；补强实验 1000 |
| 评测样本数 | 500；补强实验 1000 |
| 基础模型 | `google-bert/bert-base-uncased` |
| pooling | `cls` |
| batch size | 8 |
| gradient accumulation | 4 |
| epoch | 1 |
| seed | 42 |

结果：

| 训练规模 | 评测规模 | DCF | DCF-positive | DCF-negative | pos_neg_degree |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 500 | 0.3300 | 0.5820 | 0.4720 | 0.00000136 |
| 1000 | 1000 | 0.2930 | 0.5390 | 0.4430 | 0.00000055 |

最终 checkpoint 保存位置：

```text
results/c_task/e5_small_train/causal_reasoning_original_doge_cls_train500_eval500_seed42/checkpoint
results/c_task/e5_small_train/causal_reasoning_original_doge_cls_train1000_eval1000_seed42/checkpoint
```

### 5.3 E6 主复现消融实验

已完成三组消融，均使用 causal reasoning 500 训练样本、500 dev 评测样本、seed 42。

结果如下：

| 实验 | DCF | DCF-positive | DCF-negative | pos_neg_degree |
| --- | ---: | ---: | ---: | ---: |
| 原始 DoGE-style 配置 | 0.3300 | 0.5820 | 0.4720 | 0.00000136 |
| 关闭 dichotomy contrastive，仅保留 dichotomy loss | 0.2220 | 0.5140 | 0.3500 | 0.00039345 |
| 同时使用 dichotomy loss 与 dichotomy contrastive | 0.1900 | 0.4860 | 0.3220 | 0.00037964 |
| pooling 从 `cls` 改为 `avg` | 0.2560 | 0.5640 | 0.4160 | 0.00001228 |

观察：

- 在当前小样本设置下，原始 DoGE-style 配置的 DCF 最高。
- 去掉 dichotomy contrastive 后 DCF 明显下降，说明该项在当前复现实验中是有帮助的。
- 额外打开 `dichotomy_w` 后没有提升，反而继续下降，说明该项在小样本设置下可能需要重新调权重或训练更久。
- pooling 从 `cls` 改为 `avg` 后 DCF 下降到 0.2560，说明当前 BERT 小样本设置下 `cls` pooling 更适合作为原始 DoGE 对照。

### 5.4 E4 官方 Opposite-Score / DoGE 评测复现

官方 checkpoint 已成功完成评测。模型路径为：

```text
hf_cache\official\opposite-score-causal-reasoning-bert
```

评测数据为 causal reasoning dev 全量 17,944 条样本。

结果：

| 指标 | 值 |
| --- | ---: |
| DCF | 0.6977 |
| DCF-positive | 0.8720 |
| DCF-negative | 0.7771 |
| pos_neg_degree | 0.2396 |
| pos_neutral_degree | 0.0908 |
| neg_neutral_degree | 0.1562 |

该结果明显高于此前临时使用的本地 500 样本 checkpoint 评测结果：

| 模型 | Eval size | DCF | DCF-positive | DCF-negative | pos_neg_degree |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方 causal checkpoint | 17,944 | 0.6977 | 0.8720 | 0.7771 | 0.2396 |
| 本地 500 样本 checkpoint | 17,944 | 0.3115 | 0.5545 | 0.4807 | 0.00000127 |

结果文件：

```text
results/c_task/e4_official_eval/causal_reasoning/dev/metrics.csv
```

官方 checkpoint 曾经的网络阻塞与解决记录：

```text
results/c_task/e4_official_checkpoint_blocker.md
```

## 6. 最终交付物

| 文件或目录 | 内容 |
| --- | --- |
| `results/c_task/c_task_report_zh.md` | 本中文报告 |
| `results/c_task/c_task_report.md` | 英文简版报告 |
| `results/c_task/run_commands.md` | 所有复跑命令 |
| `results/c_task/experiment_registry.csv` | 实验登记表 |
| `results/c_task/e0_smoke/` | E0 smoke test 输出 |
| `results/c_task/e5_small_train/` | E5 小样本训练 checkpoint 和指标 |
| `results/c_task/e6_ablation/` | E6 消融 checkpoint 和指标 |
| `results/c_task/e4_official_eval/` | E4 官方 checkpoint 全量 dev 评测 |
| `results/c_task/e4_local_eval/` | E4 本地 500 样本 checkpoint 备用评测 |
| `results/c_task/e4_official_checkpoint_blocker.md` | 官方 checkpoint 网络问题与解决记录 |
| `results/c_task/submission_for_D/` | 给 D 的结果表和可视化提交包 |

## 7. 结论

本机 Windows 10 + RTX 5070 Ti 环境可以完成 C 任务所需的主要训练与评测链路，不需要 WSL。

已经完成：

- 环境配置。
- GPU 可用性验证。
- 数据读取与 tokenizer/collator/evaluator smoke test。
- 500 与 1000 样本 DoGE-style 小样本训练。
- 三组主复现消融，包括 loss 与 pooling。
- 官方 Opposite-Score / DoGE checkpoint 在 causal dev 全量数据上的评测。
- 本地 DoGE-style checkpoint 在 causal dev 全量数据上的备用评测。
- 面向 D 的图表与提交包整理。
- 统一实验登记与复跑命令整理。

当前 C 任务的主复现链路已经完成。后续主要是扩展样本规模、与 B 的创新实验和 A/D 的数据/可视化协作对接。

## 8. 接下来你还需要做什么

### 8.1 已完成：官方 checkpoint 的 E4 评测

你已经成功用本地官方 checkpoint 跑通 E4：

```powershell
conda run -n condq-c python scripts\run_doge_eval.py --scenario causal_reasoning --split dev --model hf_cache\official\opposite-score-causal-reasoning-bert --batch-size 128 --experiment-name E4_official_doge_eval_causal_dev
```

本轮已更新：

- `results/c_task/experiment_registry.csv`
- `results/c_task/c_task_report_zh.md`
- `results/c_task/e4_official_checkpoint_blocker.md`

接下来如果时间允许，可以继续补 debate 或 defeasible NLI 的官方 checkpoint 跨数据集评测，但这已经不再是 C 主复现的阻塞项。

### 8.2 已补强：小样本规模从 500 扩到 1000

已补跑 1000 样本训练：

```powershell
conda run -n condq-c python scripts\run_small_train.py --scenario causal_reasoning --train-size 1000 --eval-size 1000 --batch-size 8 --gradient-accumulation-steps 4 --epochs 1 --loss-profile original_doge --experiment-name E5_small_sample_train_causal_1000_fixed
```

结果为 DCF 0.2930。它没有超过 500 样本版本，但证明更大一点的训练规模也能稳定跑通。若后续时间充足，可选继续补 2000：

```powershell
conda run -n condq-c python scripts\run_small_train.py --scenario causal_reasoning --train-size 2000 --eval-size 1000 --batch-size 8 --gradient-accumulation-steps 4 --epochs 1 --loss-profile original_doge --experiment-name E5_small_sample_train_causal_2000
```

2000 样本不是当前 C 主任务阻塞项。

### 8.3 和 B 对接：提供原始 DoGE 对照结果

B 负责 E7 创新方法实现。你需要把当前原始 DoGE 对照结果发给 B，确保后续创新实验使用同样设置：

- 数据集：`causal_reasoning`
- 训练样本：500
- 评测样本：500
- seed：42
- batch size：8
- gradient accumulation：4
- epoch：1
- pooling：`cls`
- 原始对照 checkpoint：`results/c_task/e5_small_train/causal_reasoning_original_doge_cls_train500_eval500_seed42/checkpoint`
- 原始对照 DCF：0.3300

如果 B 的创新方法代码合入后，你只需要把创新训练配置替换成 B 提供的 loss/profile，再用同样数据规模和 seed 跑一次即可。

### 8.4 和 A 对接：等待 clean/hard subset

A 负责 clean subset 和 hard subset。拿到 A 的子集后，你需要用相同训练脚本在子集上补跑一组训练或评测。

建议优先顺序：

1. clean subset 上跑 E5 原始 DoGE。
2. hard subset 上跑评测，观察 DCF 是否下降。
3. 如果 B 的创新方法已完成，再在 hard subset 上跑原始 DoGE vs 创新方法对比。

目前脚本已经支持固定 jsonl 路径的基础逻辑，但如果 A 给的是新文件名，可能需要在 `scripts/condq_common.py` 里补一个数据路径映射。

### 8.5 和 D 对接：给图表负责人提供可视化输入

D 需要图表和 PPT。已经整理好专门提交包：

```text
results/c_task/submission_for_D/
```

其中包含：

- `figures/c_task_dcf_overview.png`
- `figures/e6_ablation_dcf_pos_neg.png`
- `figures/pos_neg_degree_comparison.png`
- `figures/official_angle_distributions.png`
- `tables/c_task_summary_for_D.csv`
- `tables/e6_ablation_for_D.csv`
- `tables/e5_training_scale_for_D.csv`
- `README.md`

D 可以基于这些文件画：

- E5/E6 DCF 对比柱状图。
- DCF-positive / DCF-negative 对比图。
- `angles_data.csv` 的角度分布图。

### 8.6 汇报时需要说明的限制

最终汇报时不要回避以下限制，但要讲清楚它们不影响“训练链路已跑通”的结论：

- 官方 checkpoint 最初因 Hugging Face 429 / 连接超时无法在线加载，后来通过预下载到本地目录解决。
- 当前 E5/E6 是小样本实验，主要证明训练流程和损失消融趋势。
- 本地 500 样本 checkpoint 的 `pos_neg_degree` 数值很小，是因为训练规模很小且 evaluator 当前使用的是 `1 - cos(cosine_distance)`，不是论文表格里直观的角度度数；官方 checkpoint 的 `pos_neg_degree=0.2396` 明显更合理。
- 当前最可靠的结论是：在 500 样本 causal setting 下，原始 DoGE-style contrastive 配置优于两个已跑消融。

### 8.7 你可以直接提交的材料

如果老师或组长现在就要检查进度，可以提交：

- 本报告：`results/c_task/c_task_report_zh.md`
- 实验登记：`results/c_task/experiment_registry.csv`
- D 的提交包：`results/c_task/submission_for_D/`
- 复跑命令：`results/c_task/run_commands.md`
- 官方 E4 结果：`results/c_task/e4_official_eval/causal_reasoning/dev/metrics.csv`
- 环境检查：`results/c_task/e0_smoke/environment_check.md`
- checkpoint 说明：`results/c_task/e5_small_train/causal_reasoning_original_doge_cls_train500_eval500_seed42/checkpoint`

当前状态可以概括为：

> C 负责的环境配置、官方 DoGE 评测、小样本训练、消融实验和原始 DoGE 对照链路均已完成；后续重点是扩大样本规模，以及和 A/B/D 对接 clean/hard subset、创新实验和可视化。
