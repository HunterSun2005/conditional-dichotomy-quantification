# 成员 A 交付说明

## 已完成工作

成员 A 负责 E1、E2、E2.5，以及 E8 中的数据质量与错误分析部分。当前已生成的产物覆盖：

- E1 动机案例：展示为什么原始相似度不能直接等价于条件对立。
- E2 数据字段映射、split 统计，以及 24 条样本语义角色检查记录。
- E2 指标说明：解释 DCF、DCF-positive、DCF-negative 和 pos_neg_degree。
- E2.5 固定随机种子的 small、clean、hard 子集，用于后续训练、消融和创新实验。
- E8 上下文敏感性输入文件，以及 normal、random、empty、weak context 四种设置下的轻量 proxy 结果表。
- E8 错误样例和定性错误分析。

## 文件结构

- `scripts/a_task/generate_member_a_artifacts.py`：主入口脚本。
- `scripts/a_task/config.py`：数据路径和原始字段映射。
- `scripts/a_task/data_io.py`：JSONL/CSV 读写，以及标准化样本结构。
- `scripts/a_task/lexical_metrics.py`：基于 Python 标准库的 bag-of-words DCF proxy。
- `scripts/a_task/artifact_builder.py`：子集构造、动机案例、context 变体和错误样例抽取。
- `scripts/a_task/report_text.py`：Markdown 报告文本生成逻辑。
- `results/a_task/`：生成的报告和 CSV 输出。
- `data/member_a_splits/`：生成的 small、clean、hard 和 context-variant JSONL 文件。

## 如何运行

成员 A 的产物生成脚本只依赖 Python 标准库，因此不需要创建虚拟环境。

```bash
python scripts/a_task/generate_member_a_artifacts.py
```

可选参数：

```bash
python scripts/a_task/generate_member_a_artifacts.py --seed 123 --max-eval 1000
```

脚本写入生成文件的路径均为相对路径，因此整个项目目录迁移后，不需要重写报告或 JSONL 元数据。

## 输出文件

- `data_report.md`：数据文件、样本数量、空字段检查和字段角色映射。
- `metric_report.md`：DCF 指标定义与解释。
- `data_splits_report.md`：生成子集的路径和筛选规则。
- `motivation_cases.csv`：5 条用于 E1 的词面相似度失败案例。
- `sample_role_check.csv`：跨数据集和 split 的 24 条已检查样例。
- `context_sensitivity_results.csv`：E8 context 扰动下的 proxy 指标。
- `error_cases.csv` 和 `error_analysis.md`：24 条 DCF 失败样例及错误类型分析。
- `run_commands.md`：简洁命令参考。

## 如何解读结果

当前 E8 结果表使用的是轻量 bag-of-words proxy，不是最终的 DoGE/Opposite-Score 模型。它适合用来筛选 hard examples，并检查 context 扰动文件是否构造正确。最终模型结论需要在 B/C 的统一评测脚本中，使用相同 context variants 和指定 checkpoint 重新评测后再给出。

DCF 越高，表示 neutral 在几何关系上越经常比 positive-negative 两端更接近两侧端点。`pos_neg_degree` 表示 positive 和 negative 的端点分离强度，但不能单独解读：模型可能把 positive 和 negative 分得很开，却仍然把 neutral 放在错误位置。

clean subset 适合 smoke test 和稳定的小规模实验。hard subset 适合消融实验、创新方法压力测试和定性错误分析。
