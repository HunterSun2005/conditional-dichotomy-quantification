# 数据子集报告

所有子集都使用标准化字段 `context`、`positive`、`negative` 和 `neutral`，并保留来源元数据。

## 筛选规则

- `small_500`、`small_1000`、`small_2000`：固定随机种子 42 的确定性随机抽样；如运行时指定其他 seed，则以指定值为准。
- `clean`：四个核心字段完整、四段文本互不相同，且总 token 长度不超过该 split 的 85 分位数。
- `hard`：bag-of-words proxy 下 DCF 失败，或 DCF margin 很小的样本。

## 生成文件

| 数据集 | Split | 子集 | 数量 | 路径 |
| --- | --- | --- | ---: | --- |
| debate | train | small_500 | 500 | `data/member_a_splits/debate/train/small_500.jsonl` |
| debate | train | small_1000 | 1000 | `data/member_a_splits/debate/train/small_1000.jsonl` |
| debate | train | small_2000 | 2000 | `data/member_a_splits/debate/train/small_2000.jsonl` |
| debate | train | clean | 50137 | `data/member_a_splits/debate/train/clean.jsonl` |
| debate | train | hard | 500 | `data/member_a_splits/debate/train/hard.jsonl` |
| debate | dev | small_500 | 500 | `data/member_a_splits/debate/dev/small_500.jsonl` |
| debate | dev | small_1000 | 1000 | `data/member_a_splits/debate/dev/small_1000.jsonl` |
| debate | dev | small_2000 | 2000 | `data/member_a_splits/debate/dev/small_2000.jsonl` |
| debate | dev | clean | 18121 | `data/member_a_splits/debate/dev/clean.jsonl` |
| debate | dev | hard | 500 | `data/member_a_splits/debate/dev/hard.jsonl` |
| debate | test | small_500 | 500 | `data/member_a_splits/debate/test/small_500.jsonl` |
| debate | test | small_1000 | 1000 | `data/member_a_splits/debate/test/small_1000.jsonl` |
| debate | test | small_2000 | 2000 | `data/member_a_splits/debate/test/small_2000.jsonl` |
| debate | test | clean | 13870 | `data/member_a_splits/debate/test/clean.jsonl` |
| debate | test | hard | 500 | `data/member_a_splits/debate/test/hard.jsonl` |
| defeasible_nli | train | small_500 | 500 | `data/member_a_splits/defeasible_nli/train/small_500.jsonl` |
| defeasible_nli | train | small_1000 | 1000 | `data/member_a_splits/defeasible_nli/train/small_1000.jsonl` |
| defeasible_nli | train | small_2000 | 2000 | `data/member_a_splits/defeasible_nli/train/small_2000.jsonl` |
| defeasible_nli | train | clean | 7319 | `data/member_a_splits/defeasible_nli/train/clean.jsonl` |
| defeasible_nli | train | hard | 500 | `data/member_a_splits/defeasible_nli/train/hard.jsonl` |
| defeasible_nli | dev | small_500 | 500 | `data/member_a_splits/defeasible_nli/dev/small_500.jsonl` |
| defeasible_nli | dev | small_1000 | 1000 | `data/member_a_splits/defeasible_nli/dev/small_1000.jsonl` |
| defeasible_nli | dev | small_2000 | 2000 | `data/member_a_splits/defeasible_nli/dev/small_2000.jsonl` |
| defeasible_nli | dev | clean | 7366 | `data/member_a_splits/defeasible_nli/dev/clean.jsonl` |
| defeasible_nli | dev | hard | 500 | `data/member_a_splits/defeasible_nli/dev/hard.jsonl` |
| causal_reasoning | train | small_500 | 500 | `data/member_a_splits/causal_reasoning/train/small_500.jsonl` |
| causal_reasoning | train | small_1000 | 1000 | `data/member_a_splits/causal_reasoning/train/small_1000.jsonl` |
| causal_reasoning | train | small_2000 | 2000 | `data/member_a_splits/causal_reasoning/train/small_2000.jsonl` |
| causal_reasoning | train | clean | 11987 | `data/member_a_splits/causal_reasoning/train/clean.jsonl` |
| causal_reasoning | train | hard | 500 | `data/member_a_splits/causal_reasoning/train/hard.jsonl` |
| causal_reasoning | dev | small_500 | 500 | `data/member_a_splits/causal_reasoning/dev/small_500.jsonl` |
| causal_reasoning | dev | small_1000 | 1000 | `data/member_a_splits/causal_reasoning/dev/small_1000.jsonl` |
| causal_reasoning | dev | small_2000 | 2000 | `data/member_a_splits/causal_reasoning/dev/small_2000.jsonl` |
| causal_reasoning | dev | clean | 15188 | `data/member_a_splits/causal_reasoning/dev/clean.jsonl` |
| causal_reasoning | dev | hard | 500 | `data/member_a_splits/causal_reasoning/dev/hard.jsonl` |
| causal_reasoning | test | small_500 | 500 | `data/member_a_splits/causal_reasoning/test/small_500.jsonl` |
| causal_reasoning | test | small_1000 | 1000 | `data/member_a_splits/causal_reasoning/test/small_1000.jsonl` |
| causal_reasoning | test | small_2000 | 2000 | `data/member_a_splits/causal_reasoning/test/small_2000.jsonl` |
| causal_reasoning | test | clean | 13196 | `data/member_a_splits/causal_reasoning/test/clean.jsonl` |
| causal_reasoning | test | hard | 500 | `data/member_a_splits/causal_reasoning/test/hard.jsonl` |

## 推荐用途

- clean subset 适合 smoke test 和稳定的小样本训练。
- hard subset 适合 E6/E7 消融压力测试和 E8 人工错误分析。
- small subset 适合 B/C 需要固定规模、可对齐训练或评测输入的场景。
