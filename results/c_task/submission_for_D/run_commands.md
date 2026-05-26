# C Task Run Commands

Environment:

```powershell
conda create -y -n condq-c python=3.10 pip
conda activate condq-c
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-c-windows.txt
python -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fsspec==2026.2.0 hf_xet
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . --no-deps
```

E0 smoke test:

```powershell
conda run -n condq-c python scripts\run_e0_smoke.py --scenario causal_reasoning --split dev --sample-size 10 --batch-size 4
```

E4 official checkpoint evaluation, completed with the predownloaded official local checkpoint:

```powershell
conda run -n condq-c python scripts\run_doge_eval.py --scenario causal_reasoning --split dev --model hf_cache\official\opposite-score-causal-reasoning-bert --batch-size 128 --experiment-name E4_official_doge_eval_causal_dev
```

E4 fallback/local DoGE checkpoint evaluation, completed in this run:

```powershell
conda run -n condq-c python scripts\run_doge_eval.py --scenario causal_reasoning --split dev --model results\c_task\e5_small_train\causal_reasoning_original_doge_cls_seed42\checkpoint --batch-size 128 --experiment-name E4_local_doge_eval_causal_dev
```

E5 small-sample training:

```powershell
conda run -n condq-c python scripts\run_small_train.py --scenario causal_reasoning --train-size 500 --eval-size 500 --batch-size 8 --gradient-accumulation-steps 4 --epochs 1 --loss-profile original_doge --experiment-name E5_small_sample_train_causal_500

conda run -n condq-c python scripts\run_small_train.py --scenario causal_reasoning --train-size 1000 --eval-size 1000 --batch-size 8 --gradient-accumulation-steps 4 --epochs 1 --loss-profile original_doge --experiment-name E5_small_sample_train_causal_1000_fixed
```

E6 ablation:

```powershell
conda run -n condq-c python scripts\run_ablation.py --scenario causal_reasoning --profiles no_dichotomy_contrastive with_dichotomy_loss --train-size 500 --eval-size 500 --batch-size 8 --gradient-accumulation-steps 4 --epochs 1 --seed 42

conda run -n condq-c python scripts\run_small_train.py --scenario causal_reasoning --train-size 500 --eval-size 500 --batch-size 8 --gradient-accumulation-steps 4 --epochs 1 --pooling-strategy avg --loss-profile original_doge --output-root results\c_task\e6_ablation --experiment-name E6_pooling_ablation_avg_fixed
```

Build D-facing submission package:

```powershell
conda run -n condq-c python scripts\make_c_task_submission.py
```
