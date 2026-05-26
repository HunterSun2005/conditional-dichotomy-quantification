# C Task Reproduction Report

## Scope

Member C is responsible for E4, E5, E6, and the original DoGE control side of E7. Innovation-specific collaboration is left for later, as requested.

## Environment

- OS: Windows 10 / 10.0.22631
- Conda env: `condq-c`
- Python: 3.10.20
- GPU: NVIDIA GeForce RTX 5070 Ti, 15.9 GB
- PyTorch: 2.10.0+cu128
- CUDA runtime used by PyTorch: 12.8
- HF cache: `D:\nlp\conditional-dichotomy-quantification\hf_cache`

Note: general Python packages were installed from Tsinghua PyPI. The Tsinghua PyPI `torch` wheel was CPU-only, so GPU PyTorch had to be installed from the official PyTorch CUDA 12.8 wheel index.

## Completed Results

| Experiment | Dataset / split | Train size | Eval size | Model / checkpoint | DCF | DCF+ | DCF- | pos_neg_degree |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| E5 small training | causal/dev | 500 | 500 | `google-bert/bert-base-uncased` -> local DoGE | 0.3300 | 0.5820 | 0.4720 | 0.00000136 |
| E5 scale check | causal/dev | 1000 | 1000 | `google-bert/bert-base-uncased` -> local DoGE | 0.2930 | 0.5390 | 0.4430 | 0.00000055 |
| E6 ablation: no contrastive | causal/dev | 500 | 500 | local DoGE | 0.2220 | 0.5140 | 0.3500 | 0.00039345 |
| E6 ablation: with dichotomy loss | causal/dev | 500 | 500 | local DoGE | 0.1900 | 0.4860 | 0.3220 | 0.00037964 |
| E6 ablation: avg pooling | causal/dev | 500 | 500 | local DoGE | 0.2560 | 0.5640 | 0.4160 | 0.00001228 |
| E4 official checkpoint eval | causal/dev | - | 17,944 | `hf_cache/official/opposite-score-causal-reasoning-bert` | 0.6977 | 0.8720 | 0.7771 | 0.23955676 |
| E4 local checkpoint eval | causal/dev | - | 17,944 | E5 local checkpoint | 0.3115 | 0.5545 | 0.4807 | 0.00000127 |

## Interpretation

The small-sample training chain is working end to end on Windows GPU: dataset loading, tokenization, training, checkpoint saving, and DCF evaluation all complete successfully.

For the 500-sample causal setting, the original DoGE-style contrastive setup performs best among the completed small ablations. The two loss ablations both reduce DCF, which supports keeping the dichotomy contrastive component in the main reproduction setting. The absolute `pos_neg_degree` values are small because the official evaluator computes `1 - cos(cosine_distance)` rather than a raw angle in degrees; results should be compared within the same evaluator.

## E4 Official Checkpoint Status

The official causal checkpoint was initially blocked by Hugging Face 429/timeout errors, but it was later predownloaded to `hf_cache/official/opposite-score-causal-reasoning-bert` and evaluated successfully on the full causal dev split. The official result is DCF 0.6977.

## Outputs

- Smoke test: `results/c_task/e0_smoke/environment_check.md`
- Registry: `results/c_task/experiment_registry.csv`
- E4 metrics: `results/c_task/e4_official_eval/causal_reasoning/dev/metrics.csv`
- E5 checkpoint: `results/c_task/e5_small_train/causal_reasoning_original_doge_cls_seed42/checkpoint`
- E6 checkpoints: `results/c_task/e6_ablation/*/checkpoint`
- Repro commands: `results/c_task/run_commands.md`
- D-facing package: `results/c_task/submission_for_D/`
