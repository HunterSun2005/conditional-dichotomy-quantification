# 数据报告

本报告统计官方仓库数据文件，并将其标准化为成员 A 的 CDQ 分析格式。

## 字段映射

| 原始字段 | 标准化字段 | 语义角色 |
| --- | --- | --- |
| `context_text` | `context` | 共享条件 Z。 |
| `supporter_text` | `positive` | 支持上下文关系的文本。 |
| `defeater_text` | `negative` | 削弱或反对上下文关系的文本。 |
| `neutral_text` | `neutral` | 不应与正反两侧形成强对立端点的中性文本。 |

## 文件与基础统计

| 数据集 | Split | 样本数 | 唯一 context 数 | 核心字段为空样本数 | 平均 token 数 | 源文件 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| causal_reasoning | dev | 17944 | 580 | 0 | 48.5 | `data/causal_reasoning/deltaCausal_dev_processed_gpt-4o.jsonl` |
| causal_reasoning | test | 15566 | 500 | 0 | 48.4 | `data/causal_reasoning/deltaCausal_test_processed_gpt-4o.jsonl` |
| causal_reasoning | train | 14008 | 6998 | 0 | 49.1 | `data/causal_reasoning/deltaCausal_train_selected_from_A_B.jsonl` |
| debate | dev | 21316 | 105 | 0 | 41.9 | `data/debate/perspectrum_dev_processed_gpt-4o.jsonl` |
| debate | test | 16150 | 156 | 0 | 44.8 | `data/debate/perspectrum_test_processed_gpt-4o.jsonl` |
| debate | train | 58058 | 363 | 0 | 43.5 | `data/debate/perspectrum_train_selected_from_B_C.jsonl` |
| defeasible_nli | dev | 8656 | 202 | 0 | 48.5 | `data/defeasible_nli/dev_processed_gpt-4o.jsonl` |
| defeasible_nli | train | 8462 | 194 | 0 | 49.1 | `data/defeasible_nli/train.jsonl` |

## 人工检查说明

`results/a_task/sample_role_check.csv` 中包含跨数据集和 split 的 24 条已检查样本。
每条记录用于确认 positive 是否支持上下文关系、negative 是否削弱或反对该关系、neutral 是否不是强对立端点。

## Split 使用建议

- `dev` 适合 E1/E2/E8 的快速迭代，因为三个数据集都提供了处理后的 dev 文件。
- debate 和 causal reasoning 的最终评测可使用 `test`。
- Defeasible NLI 若没有完整 test 文件，则使用 `dev` 作为评测 split。
