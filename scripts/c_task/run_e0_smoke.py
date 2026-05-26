import argparse
import platform
from pathlib import Path

from condq_common import (
    DEFAULT_OUTPUT_ROOT,
    configure_hf_cache,
    ensure_dir,
    evaluate_dataset,
    load_condq_dataset,
    markdown_table,
    scenario_file,
    set_seed,
    tokenize_dataset,
    torch_environment,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C-member environment/data smoke test.")
    parser.add_argument("--scenario", default="causal_reasoning")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-bert")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "e0_smoke"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cache_dir = configure_hf_cache(args.cache_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    import torch
    from oppositescore.model.dichotomye import DichotomyE
    from oppositescore.trainer.dichotomy_dataset import DichotomyEDataCollator, DichotomyEDataTokenizer

    data_path = scenario_file(args.scenario, args.split)
    dataset = load_condq_dataset(data_path, sample_size=args.sample_size, seed=args.seed)
    model = DichotomyE.from_pretrained(
        args.model,
        cached_hf_dir=str(cache_dir),
        max_length=args.max_length,
        pooling_strategy="cls",
        train_mode=False,
    )
    if torch.cuda.is_available():
        model.cuda()

    tokenizer_fn = DichotomyEDataTokenizer(model.tokenizer, model.max_length)
    tokenized = tokenize_dataset(dataset, tokenizer_fn, "Smoke tokenization")
    collator = DichotomyEDataCollator(model.tokenizer, return_tensors="pt", max_length=model.max_length)
    batch = collator([tokenized[i] for i in range(min(2, len(tokenized)))])
    metrics = evaluate_dataset(model, dataset, output_dir=output_dir, batch_size=args.batch_size)

    env = torch_environment()
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cache_dir": str(cache_dir),
        "data_path": str(data_path),
        "sample_size": len(dataset),
        "batch_input_shape": list(batch["input_ids"].shape),
        "metrics": metrics,
        "environment": env,
    }
    write_json(output_dir / "smoke_result.json", payload)

    rows = [
        {"item": "Platform", "value": payload["platform"]},
        {"item": "Python", "value": payload["python"]},
        {"item": "PyTorch", "value": env["torch"]},
        {"item": "CUDA runtime", "value": env["cuda"]},
        {"item": "CUDA available", "value": env["cuda_available"]},
        {"item": "GPU", "value": env["gpu"]},
        {"item": "GPU memory GB", "value": env["gpu_memory_gb"]},
        {"item": "HF cache", "value": payload["cache_dir"]},
        {"item": "Data file", "value": payload["data_path"]},
        {"item": "Smoke samples", "value": payload["sample_size"]},
        {"item": "DCF", "value": f"{metrics['DCF']:.4f}"},
    ]
    report = "# C Environment Smoke Test\n\n" + markdown_table(rows, ["item", "value"]) + "\n"
    (output_dir / "environment_check.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
