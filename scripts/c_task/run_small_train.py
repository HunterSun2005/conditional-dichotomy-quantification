import argparse
from pathlib import Path
from typing import Dict

from condq_common import (
    DEFAULT_OUTPUT_ROOT,
    append_registry,
    configure_hf_cache,
    ensure_dir,
    evaluate_dataset,
    load_condq_dataset,
    resolve_scenario,
    scenario_file,
    set_seed,
    timestamp,
    tokenize_dataset,
    torch_environment,
    write_json,
    write_metrics_csv,
)


LOSS_PRESETS: Dict[str, Dict[str, float]] = {
    "original_doge": {
        "cosine_w": 1.0,
        "ibn_w": 20.0,
        "angle_w": 1.0,
        "dichotomy_w": 0.0,
        "dichotomy_contrastive_w": 1.0,
        "cosine_tau": 20.0,
        "ibn_tau": 20.0,
        "angle_tau": 20.0,
        "dichotomy_tau": 20.0,
        "dichotomy_contrastive_tau": 1.0,
    },
    "no_dichotomy_contrastive": {
        "cosine_w": 1.0,
        "ibn_w": 20.0,
        "angle_w": 1.0,
        "dichotomy_w": 1.0,
        "dichotomy_contrastive_w": 0.0,
        "cosine_tau": 20.0,
        "ibn_tau": 20.0,
        "angle_tau": 20.0,
        "dichotomy_tau": 20.0,
        "dichotomy_contrastive_tau": 1.0,
    },
    "with_dichotomy_loss": {
        "cosine_w": 1.0,
        "ibn_w": 20.0,
        "angle_w": 1.0,
        "dichotomy_w": 1.0,
        "dichotomy_contrastive_w": 1.0,
        "cosine_tau": 20.0,
        "ibn_tau": 20.0,
        "angle_tau": 20.0,
        "dichotomy_tau": 20.0,
        "dichotomy_contrastive_tau": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E5 small-sample DoGE training.")
    parser.add_argument("--scenario", default="causal_reasoning")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "e5_small_train"))
    parser.add_argument("--registry", default=str(DEFAULT_OUTPUT_ROOT / "experiment_registry.csv"))
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pooling-strategy", default="cls")
    parser.add_argument("--loss-profile", choices=sorted(LOSS_PRESETS), default="original_doge")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--owner", default="C")
    parser.add_argument("--experiment-name", default="E5_small_sample_train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = resolve_scenario(args.scenario)
    set_seed(args.seed)
    cache_dir = configure_hf_cache(args.cache_dir)
    run_name = (
        f"{scenario}_{args.loss_profile}_{args.pooling_strategy}"
        f"_train{args.train_size}_eval{args.eval_size}_seed{args.seed}"
    )
    output_dir = ensure_dir(Path(args.output_root) / run_name)
    checkpoint_dir = ensure_dir(output_dir / "checkpoint")

    import torch
    from oppositescore.model.dichotomye import DichotomyE
    from oppositescore.trainer.dichotomy_dataset import DichotomyEDataTokenizer

    train_path = scenario_file(scenario, args.train_split)
    eval_path = scenario_file(scenario, args.eval_split)
    train_ds = load_condq_dataset(train_path, sample_size=args.train_size, seed=args.seed).shuffle(seed=args.seed)
    eval_ds = load_condq_dataset(eval_path, sample_size=args.eval_size, seed=args.seed)

    model = DichotomyE.from_pretrained(
        args.model,
        cached_hf_dir=str(cache_dir),
        max_length=args.max_length,
        pooling_strategy=args.pooling_strategy,
        train_mode=True,
    )
    if torch.cuda.is_available():
        model.cuda()

    tokenizer_fn = DichotomyEDataTokenizer(model.tokenizer, model.max_length)
    tokenized_train = tokenize_dataset(train_ds, tokenizer_fn, "Tokenizing train")
    loss_kwargs = LOSS_PRESETS[args.loss_profile]
    write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "scenario": scenario,
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "loss_kwargs": loss_kwargs,
            "cache_dir": str(cache_dir),
        },
    )

    model.fit(
        train_ds=tokenized_train,
        valid_ds=None,
        output_dir=str(checkpoint_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=1000,
        eval_steps=None,
        warmup_steps=0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_kwargs=loss_kwargs,
        fp16=args.fp16,
        logging_steps=10,
        argument_kwargs={"report_to": [], "save_safetensors": True},
    )

    metrics = evaluate_dataset(model, eval_ds, output_dir=output_dir, batch_size=max(args.batch_size, 16))
    env = torch_environment()
    write_json(output_dir / "metrics.json", {"metrics": metrics, "environment": env})
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    append_registry(
        {
            **metrics,
            "timestamp": timestamp(),
            "experiment_name": args.experiment_name,
            "owner": args.owner,
            "dataset": scenario,
            "split": args.eval_split,
            "model": args.model,
            "checkpoint": str(checkpoint_dir),
            "seed": args.seed,
            "train_size": len(train_ds),
            "eval_size": len(eval_ds),
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "pooling_strategy": args.pooling_strategy,
            "loss_profile": args.loss_profile,
            "gpu": env["gpu"],
            "cuda": env["cuda"],
            "torch": env["torch"],
            "output_dir": str(output_dir),
            "notes": "Small-sample DoGE training run.",
        },
        Path(args.registry),
    )
    print(metrics)


if __name__ == "__main__":
    main()
