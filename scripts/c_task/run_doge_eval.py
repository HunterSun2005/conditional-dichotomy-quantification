import argparse
from pathlib import Path

from condq_common import (
    DEFAULT_OUTPUT_ROOT,
    append_registry,
    configure_hf_cache,
    ensure_dir,
    evaluate_dataset,
    load_condq_dataset,
    official_model_for,
    resolve_scenario,
    scenario_file,
    set_seed,
    timestamp,
    torch_environment,
    write_json,
    write_metrics_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official Opposite-Score/DoGE evaluation.")
    parser.add_argument("--scenario", default="causal_reasoning")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "e4_official_eval"))
    parser.add_argument("--registry", default=str(DEFAULT_OUTPUT_ROOT / "experiment_registry.csv"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pooling-strategy", default="cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--owner", default="C")
    parser.add_argument("--experiment-name", default="E4_official_doge_eval")
    args = parser.parse_args()

    scenario = resolve_scenario(args.scenario)
    model_name = args.model or official_model_for(scenario)
    set_seed(args.seed)
    cache_dir = configure_hf_cache(args.cache_dir)
    output_dir = ensure_dir(Path(args.output_dir) / scenario / args.split)

    import torch
    from oppositescore.model.dichotomye import DichotomyE

    data_path = scenario_file(scenario, args.split)
    dataset = load_condq_dataset(data_path, sample_size=args.sample_size, seed=args.seed)
    checkpoint_type = "official_checkpoint"
    if args.model:
        model_path = Path(model_name)
        if model_path.exists() and "opposite-score" in model_path.name:
            checkpoint_type = "official_local_checkpoint"
        elif model_path.exists():
            checkpoint_type = "local_checkpoint"
        else:
            checkpoint_type = "external_checkpoint"
    model = DichotomyE.from_pretrained(
        model_name,
        cached_hf_dir=str(cache_dir),
        max_length=args.max_length,
        pooling_strategy=args.pooling_strategy,
        train_mode=False,
    )
    if torch.cuda.is_available():
        model.cuda()

    metrics = evaluate_dataset(model, dataset, output_dir=output_dir, batch_size=args.batch_size)
    env = torch_environment()
    run_info = {
        "experiment_name": args.experiment_name,
        "scenario": scenario,
        "split": args.split,
        "model": model_name,
        "sample_size": len(dataset),
        "batch_size": args.batch_size,
        "pooling_strategy": args.pooling_strategy,
        "seed": args.seed,
        "data_path": str(data_path),
        "cache_dir": str(cache_dir),
        "metrics": metrics,
        "environment": env,
    }
    write_json(output_dir / "metrics.json", run_info)
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    append_registry(
        {
            **metrics,
            "timestamp": timestamp(),
            "experiment_name": args.experiment_name,
            "owner": args.owner,
            "dataset": scenario,
            "split": args.split,
            "model": model_name,
            "checkpoint": model_name,
            "seed": args.seed,
            "eval_size": len(dataset),
            "batch_size": args.batch_size,
            "pooling_strategy": args.pooling_strategy,
            "loss_profile": checkpoint_type,
            "gpu": env["gpu"],
            "cuda": env["cuda"],
            "torch": env["torch"],
            "output_dir": str(output_dir),
            "notes": f"{checkpoint_type} evaluation.",
        },
        Path(args.registry),
    )
    print(metrics)


if __name__ == "__main__":
    main()
