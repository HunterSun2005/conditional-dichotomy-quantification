import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
from datasets import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "c_task"
DEFAULT_HF_CACHE = PROJECT_ROOT / "hf_cache"

SCENARIOS: Dict[str, Dict[str, object]] = {
    "debate": {
        "aliases": ["A", "perspectrum"],
        "official_model": "shaobocui/opposite-score-debate-bert",
        "train": "data/debate/perspectrum_train_selected_from_B_C.jsonl",
        "dev": "data/debate/perspectrum_dev_processed_gpt-4o.jsonl",
        "test": "data/debate/perspectrum_test_processed_gpt-4o.jsonl",
    },
    "defeasible_nli": {
        "aliases": ["B", "defeasible", "nli"],
        "official_model": "shaobocui/opposite-score-defeasibleNLI-bert",
        "train": "data/defeasible_nli/train.jsonl",
        "dev": "data/defeasible_nli/dev_processed_gpt-4o.jsonl",
        "test": None,
    },
    "causal_reasoning": {
        "aliases": ["C", "causal", "delta_causal"],
        "official_model": "shaobocui/opposite-score-causal-reasoning-bert",
        "train": "data/causal_reasoning/deltaCausal_train_selected_from_A_B.jsonl",
        "dev": "data/causal_reasoning/deltaCausal_dev_processed_gpt-4o.jsonl",
        "test": "data/causal_reasoning/deltaCausal_test_processed_gpt-4o.jsonl",
    },
}

REQUIRED_COLUMNS = ["context_text", "supporter_text", "defeater_text", "neutral_text"]
CANONICAL_COLUMNS = {
    "context_text": "context",
    "supporter_text": "positive",
    "defeater_text": "negative",
    "neutral_text": "neutral",
}

REGISTRY_COLUMNS = [
    "timestamp",
    "experiment_name",
    "owner",
    "dataset",
    "split",
    "model",
    "checkpoint",
    "seed",
    "train_size",
    "eval_size",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "pooling_strategy",
    "loss_profile",
    "gpu",
    "cuda",
    "torch",
    "DCF",
    "DCF-positive",
    "DCF-negative",
    "pos_neg_degree",
    "pos_neutral_degree",
    "neg_neutral_degree",
    "output_dir",
    "notes",
]


def resolve_scenario(name: str) -> str:
    normalized = name.lower()
    for key, cfg in SCENARIOS.items():
        aliases = [alias.lower() for alias in cfg["aliases"]]
        if normalized == key.lower() or normalized in aliases:
            return key
    raise ValueError(f"Unknown scenario '{name}'. Choose one of: {', '.join(SCENARIOS)}")


def scenario_file(scenario: str, split: str) -> Path:
    scenario = resolve_scenario(scenario)
    split = split.lower()
    if split == "test" and SCENARIOS[scenario].get("test") is None:
        split = "dev"
    rel = SCENARIOS[scenario].get(split)
    if rel is None:
        raise ValueError(f"Scenario '{scenario}' has no split '{split}'")
    return PROJECT_ROOT / str(rel)


def official_model_for(scenario: str) -> str:
    return str(SCENARIOS[resolve_scenario(scenario)]["official_model"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_hf_cache(cache_dir: Optional[str] = None) -> Path:
    path = Path(cache_dir).expanduser().resolve() if cache_dir else DEFAULT_HF_CACHE.resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    os.environ["TRANSFORMERS_CACHE"] = str(path)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_condq_dataset(path: Path, sample_size: Optional[int] = None, seed: int = 42) -> Dataset:
    df = pd.read_json(path, lines=True)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    df = df.rename(columns=CANONICAL_COLUMNS)
    return Dataset.from_pandas(df)


def tokenize_dataset(dataset: Dataset, tokenizer_fn, desc: str) -> Dataset:
    rows = []
    from tqdm import tqdm

    for example in tqdm(dataset, desc=desc):
        rows.append(tokenizer_fn(example))
    return Dataset.from_list(rows)


def evaluate_dataset(model, dataset: Dataset, output_dir: Path, batch_size: int) -> Dict[str, float]:
    from oppositescore.evaluation.evaluation import DichotomyEvaluator

    angles_path = output_dir / "angles_data.csv"
    evaluator = DichotomyEvaluator(
        context=dataset["context"],
        positive=dataset["positive"],
        negative=dataset["negative"],
        neutral=dataset["neutral"],
        batch_size=batch_size,
        angles_output_path=str(angles_path),
    )
    metrics = evaluator(model)
    return {key: float(value) for key, value in metrics.items()}


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_metrics_csv(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def append_registry(row: Dict, registry_path: Path) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.exists()
    clean_row = {col: row.get(col, "") for col in REGISTRY_COLUMNS}
    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(clean_row)


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def torch_environment() -> Dict[str, str]:
    import torch

    gpu = "cpu"
    total_memory_gb = ""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = props.name
        total_memory_gb = f"{props.total_memory / (1024 ** 3):.1f}"
    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "gpu": gpu,
        "gpu_memory_gb": total_memory_gb,
    }


def markdown_table(rows: Iterable[Dict[str, object]], columns: List[str]) -> str:
    rows = list(rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, separator] + body)
