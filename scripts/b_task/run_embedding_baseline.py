import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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
    torch_environment,
    write_json,
    write_metrics_csv,
)


class HFEncoderBaseline:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        pooling_strategy: str = "cls",
        max_length: int = 128,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = outputs.last_hidden_state
        if self.pooling_strategy == "cls":
            return last_hidden[:, 0]
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts
        raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

    def encode(self, inputs: List[str], **kwargs) -> np.ndarray:
        if isinstance(inputs, str):
            inputs = [inputs]
        tokens = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
            pooled = self._pool(outputs, tokens["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3 embedding baselines on ConDQ datasets.")
    parser.add_argument("--scenario", default="all", help="debate / defeasible_nli / causal_reasoning / all")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "e3_embedding_baselines"))
    parser.add_argument("--registry", default=str(DEFAULT_OUTPUT_ROOT / "experiment_registry.csv"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pooling-strategy", choices=["cls", "mean"], default="cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--owner", default="B")
    parser.add_argument("--experiment-prefix", default="E3_baseline")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def scenario_list(name: str) -> List[str]:
    if name.lower() == "all":
        return ["debate", "defeasible_nli", "causal_reasoning"]
    return [resolve_scenario(name)]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_dir = configure_hf_cache(args.cache_dir)
    env = torch_environment()

    scenarios = scenario_list(args.scenario)
    baseline = HFEncoderBaseline(
        model_name=args.model,
        cache_dir=str(cache_dir),
        pooling_strategy=args.pooling_strategy,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )

    for scenario in scenarios:
        data_path = scenario_file(scenario, args.split)
        dataset = load_condq_dataset(data_path, sample_size=args.sample_size, seed=args.seed)
        run_name = (
            f"{args.experiment_prefix}_{scenario}_{args.split}_"
            f"{args.model.split('/')[-1]}_{args.pooling_strategy}_seed{args.seed}"
        )
        output_dir = ensure_dir(Path(args.output_root) / run_name)
        metrics = evaluate_dataset(baseline, dataset, output_dir=output_dir, batch_size=args.batch_size)

        run_info = {
            "experiment_name": run_name,
            "scenario": scenario,
            "split": args.split,
            "model": args.model,
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
                "experiment_name": run_name,
                "owner": args.owner,
                "dataset": scenario,
                "split": args.split,
                "model": args.model,
                "checkpoint": args.model,
                "seed": args.seed,
                "eval_size": len(dataset),
                "batch_size": args.batch_size,
                "pooling_strategy": args.pooling_strategy,
                "loss_profile": "baseline_embedding",
                "gpu": env["gpu"],
                "cuda": env["cuda"],
                "torch": env["torch"],
                "output_dir": str(output_dir),
                "notes": "Raw pretrained encoder baseline for E3/E9.",
            },
            Path(args.registry),
        )
        print({"scenario": scenario, **metrics})


if __name__ == "__main__":
    main()
