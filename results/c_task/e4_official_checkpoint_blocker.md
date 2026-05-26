# E4 Official Checkpoint Network Note

The official Hugging Face checkpoints under `shaobocui/*` could not be downloaded during the first run, but the causal checkpoint was later downloaded successfully and E4 has now been completed.

Observed failures:

- `shaobocui/opposite-score-causal-reasoning-bert`: `429 Client Error: Too Many Requests` on `config.json`.
- `shaobocui/opposite-score-debate-bert`: `429 Client Error: Too Many Requests` on `config.json`.
- `HF_ENDPOINT=https://hf-mirror.com` was also tried, but the mirror returned SSL/connection-reset failures for the same model files.

Initial fallback:

- Trained a local DoGE-style BERT checkpoint on causal reasoning 500 samples.
- Evaluated that local checkpoint on the full causal dev split, 17,944 examples.
- Fallback result: DCF 0.3115.

Completed official run:

- Official local checkpoint: `hf_cache\official\opposite-score-causal-reasoning-bert`
- Dataset: causal reasoning dev, 17,944 examples.
- Result: DCF 0.6977, DCF-positive 0.8720, DCF-negative 0.7771, pos_neg_degree 0.2396.
- Results are in `results/c_task/e4_official_eval/causal_reasoning/dev/metrics.csv`.

Rerun command:

```powershell
conda run -n condq-c python scripts\run_doge_eval.py --scenario causal_reasoning --split dev --model hf_cache\official\opposite-score-causal-reasoning-bert --batch-size 128 --experiment-name E4_official_doge_eval_causal_dev
```
