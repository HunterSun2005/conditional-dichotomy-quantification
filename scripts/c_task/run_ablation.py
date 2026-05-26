import argparse
import subprocess
import sys
from pathlib import Path

from condq_common import DEFAULT_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E6 ablations by invoking run_small_train.py.")
    parser.add_argument("--scenario", default="causal_reasoning")
    parser.add_argument("--profiles", nargs="+", default=["original_doge", "no_dichotomy_contrastive", "with_dichotomy_loss"])
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "e6_ablation"))
    parser.add_argument("--registry", default=str(DEFAULT_OUTPUT_ROOT / "experiment_registry.csv"))
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    args = parser.parse_args()

    script = Path(__file__).with_name("run_small_train.py")
    for profile in args.profiles:
        cmd = [
            sys.executable,
            str(script),
            "--scenario",
            args.scenario,
            "--train-size",
            str(args.train_size),
            "--eval-size",
            str(args.eval_size),
            "--batch-size",
            str(args.batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--output-root",
            args.output_root,
            "--registry",
            args.registry,
            "--model",
            args.model,
            "--loss-profile",
            profile,
            "--experiment-name",
            "E6_ablation",
        ]
        if args.cache_dir:
            cmd.extend(["--cache-dir", args.cache_dir])
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
