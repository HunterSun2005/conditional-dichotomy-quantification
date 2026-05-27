# B Task Run Commands

## Environment

Current base environment is missing `peft`, so B-task baseline scripts are designed to avoid `DichotomyE` and run directly on `transformers`.

Recommended install command:

```bash
pip install -r requirements.txt
```

If you only want the E3/E3.5 baseline flow, the minimum packages are:

```bash
pip install torch transformers datasets pandas numpy scikit-learn tqdm boltons huggingface-hub
```

## E3: Single-Dataset Baseline

Run BERT on causal reasoning dev:

```bash
python scripts/b_task/run_embedding_baseline.py \
  --scenario causal_reasoning \
  --split dev \
  --model google-bert/bert-base-uncased \
  --pooling-strategy cls \
  --batch-size 32
```

Run RoBERTa on debate dev:

```bash
python scripts/b_task/run_embedding_baseline.py \
  --scenario debate \
  --split dev \
  --model FacebookAI/roberta-base \
  --pooling-strategy cls \
  --batch-size 32
```

## E3: Cross-Dataset Baseline

Run one baseline across all three datasets:

```bash
python scripts/b_task/run_embedding_baseline.py \
  --scenario all \
  --split dev \
  --model google-bert/bert-base-uncased \
  --pooling-strategy cls \
  --batch-size 32
```

## E3.5: Output Locations

- Registry: `results/b_task/experiment_registry.csv`
- Per-run metrics: `results/b_task/e3_embedding_baselines/<run_name>/metrics.json`
- Per-run CSV: `results/b_task/e3_embedding_baselines/<run_name>/metrics.csv`

## Naming Convention

Experiment names are written as:

```text
E3_baseline_<dataset>_<split>_<model_short_name>_<pooling>_seed<seed>
```

Examples:

- `E3_baseline_debate_dev_bert-base-uncased_cls_seed42`
- `E3_baseline_causal_reasoning_dev_roberta-base_cls_seed42`
