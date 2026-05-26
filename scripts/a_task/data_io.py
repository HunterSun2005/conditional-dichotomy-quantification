"""Data loading and writing helpers for member A artifacts."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from config import DATASET_FILES, FIELD_MAP


@dataclass
class Sample:
    dataset: str
    split: str
    row_index: int
    source_file: str
    raw: Dict
    context: str
    positive: str
    negative: str
    neutral: str

    @property
    def sample_id(self) -> str:
        context_id = self.raw.get("context_id", f"row{self.row_index}")
        positive_id = self.raw.get("positive_id", self.raw.get("supporter_id", "pos"))
        negative_id = self.raw.get("negative_id", self.raw.get("defeater_id", "neg"))
        neutral_id = self.raw.get("neutral_id", "neu")
        return f"{self.dataset}:{self.split}:{context_id}:{positive_id}:{negative_id}:{neutral_id}"

    def normalized_record(self) -> Dict:
        record = {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "source_file": self.source_file,
            "row_index": self.row_index,
            "context": self.context,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
        }
        for key in ("context_id", "positive_id", "negative_id", "neutral_id", "neutral_prob", "neutral_to"):
            if key in self.raw:
                record[key] = self.raw[key]
        return record


def relpath(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def read_jsonl(path: Path, dataset: str, split: str, source_file: str) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                values = {target: str(raw[source]).strip() for source, target in FIELD_MAP.items()}
            except KeyError as exc:
                raise KeyError(f"{source_file} missing required field {exc}") from exc
            samples.append(
                Sample(
                    dataset=dataset,
                    split=split,
                    row_index=row_index,
                    source_file=source_file,
                    raw=raw,
                    context=values["context"],
                    positive=values["positive"],
                    negative=values["negative"],
                    neutral=values["neutral"],
                )
            )
    return samples


def load_all(root: Path) -> Dict[Tuple[str, str], List[Sample]]:
    loaded: Dict[Tuple[str, str], List[Sample]] = {}
    for dataset, splits in DATASET_FILES.items():
        for split, rel_path in splits.items():
            path = root / rel_path
            if path.exists():
                loaded[(dataset, split)] = read_jsonl(path, dataset, split, rel_path)
    return loaded


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
