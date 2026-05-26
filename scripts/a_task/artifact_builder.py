"""Build member A data subsets, motivation cases, and E8 proxy outputs."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from data_io import Sample, relpath, write_csv, write_jsonl
from lexical_metrics import evaluate_bow, jaccard, tokens


def quantile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def generate_subsets(loaded: Dict[Tuple[str, str], List[Sample]], out_dir: Path, root: Path, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    report_rows = []
    for (dataset, split), samples in loaded.items():
        normalized = [sample.normalized_record() for sample in samples]
        shuffled = list(normalized)
        rng.shuffle(shuffled)
        for size in (500, 1000, 2000):
            selected = shuffled[: min(size, len(shuffled))]
            path = out_dir / dataset / split / f"small_{size}.jsonl"
            write_jsonl(path, selected)
            report_rows.append({"dataset": dataset, "split": split, "subset": f"small_{size}", "count": len(selected), "path": relpath(path, root)})

        lengths = [
            sum(len(tokens(getattr(sample, field))) for field in ("context", "positive", "negative", "neutral"))
            for sample in samples
        ]
        max_len = quantile(lengths, 0.85)
        clean = []
        for sample, length in zip(samples, lengths):
            texts = [sample.context, sample.positive, sample.negative, sample.neutral]
            if all(text.strip() for text in texts) and len(set(texts)) == 4 and length <= max_len:
                clean.append(sample.normalized_record())
        clean_path = out_dir / dataset / split / "clean.jsonl"
        write_jsonl(clean_path, clean)
        report_rows.append({"dataset": dataset, "split": split, "subset": "clean", "count": len(clean), "path": relpath(clean_path, root)})

        _, eval_rows = evaluate_bow(samples)
        by_id = {sample.sample_id: sample for sample in samples}
        hard_ids = [
            row["sample_id"]
            for row in sorted(eval_rows, key=lambda item: (item["dcf_margin"], item["pos_neg_distance"]))
            if not row["dcf_pass"] or row["dcf_margin"] < 0.05
        ]
        hard_path = out_dir / dataset / split / "hard.jsonl"
        hard = [by_id[sample_id].normalized_record() for sample_id in hard_ids[: min(500, len(hard_ids))]]
        write_jsonl(hard_path, hard)
        report_rows.append({"dataset": dataset, "split": split, "subset": "hard", "count": len(hard), "path": relpath(hard_path, root)})
    return report_rows


def motivation_cases(loaded: Dict[Tuple[str, str], List[Sample]], limit: int) -> List[Dict]:
    candidates = []
    by_context: Dict[str, List[Sample]] = defaultdict(list)
    for samples in loaded.values():
        for sample in samples:
            by_context[sample.context].append(sample)

    for context, samples in by_context.items():
        supporters = sorted({sample.positive for sample in samples})
        defeaters = sorted({sample.negative for sample in samples})
        if len(supporters) < 2 or not defeaters:
            continue
        for i, supporter_a in enumerate(supporters):
            for supporter_b in supporters[i + 1 :]:
                support_support = jaccard(supporter_a, supporter_b)
                for defeater in defeaters:
                    support_defeater = max(jaccard(supporter_a, defeater), jaccard(supporter_b, defeater))
                    gap = support_defeater - support_support
                    if gap > 0:
                        candidates.append(
                            {
                                "context": context,
                                "supporter_a": supporter_a,
                                "supporter_b": supporter_b,
                                "defeater": defeater,
                                "support_support_jaccard": round(support_support, 4),
                                "support_defeater_jaccard": round(support_defeater, 4),
                                "why_it_motivates_condq": "A lexical similarity proxy rates a supporter-defeater pair as more similar than two same-side supporters, so similarity alone is not conditional opposition.",
                            }
                        )

    candidates.sort(key=lambda row: row["support_defeater_jaccard"] - row["support_support_jaccard"], reverse=True)
    if len(candidates) < limit:
        supporter_a = "The product's unique features attract a quite large customer base."
        supporter_b = "People frequently share their happy usage experience on social media."
        defeater = "Competitors quickly release similar products, reducing the company's advantage."
        candidates.append(
            {
                "context": "A company launches a revolutionary product. The company gains a significant market share.",
                "supporter_a": supporter_a,
                "supporter_b": supporter_b,
                "defeater": defeater,
                "support_support_jaccard": round(jaccard(supporter_a, supporter_b), 4),
                "support_defeater_jaccard": round(jaccard(supporter_a, defeater), 4),
                "why_it_motivates_condq": "This is the paper-style motivation case: same-side support can be lexically distant while the defeater shares topical words.",
            }
        )
    return candidates[:limit]


def context_variants(samples: Sequence[Sample], seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    original = [sample.context for sample in samples]
    shuffled = list(original)
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == original:
        shuffled = shuffled[1:] + shuffled[:1]
    return {
        "normal_context": original,
        "random_context": shuffled,
        "empty_context": ["" for _ in samples],
        "weak_context": [" ".join(text.split()[: max(1, len(text.split()) // 2)]) for text in original],
    }


def classify_error(sample: Sample, row: Dict) -> str:
    neutral_pos_overlap = jaccard(sample.neutral, sample.positive)
    neutral_neg_overlap = jaccard(sample.neutral, sample.negative)
    if row["pos_neg_distance"] <= row["pos_neutral_distance"] and row["pos_neg_distance"] <= row["neg_neutral_distance"]:
        return "positive_negative_too_close"
    if neutral_pos_overlap > neutral_neg_overlap + 0.1:
        return "neutral_lexically_close_to_positive"
    if neutral_neg_overlap > neutral_pos_overlap + 0.1:
        return "neutral_lexically_close_to_negative"
    if len(tokens(sample.context)) > 40:
        return "long_context_conditioning_difficult"
    return "ambiguous_or_low_lexical_signal"


def e8_results(loaded: Dict[Tuple[str, str], List[Sample]], out_dir: Path, seed: int, max_eval: int) -> Tuple[List[Dict], List[Dict]]:
    result_rows = []
    error_rows = []
    for (dataset, split), all_samples in sorted(loaded.items()):
        if split != "dev":
            continue
        samples = all_samples[: min(max_eval, len(all_samples))]
        variants = context_variants(samples, seed)
        normal_eval_rows = []
        for variant_name, contexts in variants.items():
            metrics, eval_rows = evaluate_bow(samples, contexts)
            if variant_name == "normal_context":
                normal_eval_rows = eval_rows
            row = {
                "experiment": "E8_context_sensitivity_bow_proxy",
                "owner": "A",
                "dataset": dataset,
                "split": split,
                "model": "standard-library-bag-of-words-proxy",
                "checkpoint": "none",
                "seed": seed,
                "max_eval_samples": len(samples),
                "context_variant": variant_name,
            }
            row.update({key: round(value, 6) for key, value in metrics.items()})
            result_rows.append(row)

        by_id = {sample.sample_id: sample for sample in samples}
        failing = [row for row in normal_eval_rows if not row["dcf_pass"]]
        failing.sort(key=lambda row: row["dcf_margin"])
        for row in failing[:8]:
            sample = by_id[row["sample_id"]]
            error_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset": dataset,
                    "split": split,
                    "context": sample.context,
                    "positive": sample.positive,
                    "negative": sample.negative,
                    "neutral": sample.neutral,
                    "pos_neutral_distance": row["pos_neutral_distance"],
                    "pos_neg_distance": row["pos_neg_distance"],
                    "neg_neutral_distance": row["neg_neutral_distance"],
                    "error_type": classify_error(sample, row),
                    "possible_innovation_help": "Axis-aware loss may help if it explicitly separates positive-negative while keeping neutral near the midpoint under the same context.",
                }
            )
    write_csv(out_dir / "context_sensitivity_results.csv", result_rows)
    return result_rows, error_rows[:24]


def write_context_variant_files(loaded: Dict[Tuple[str, str], List[Sample]], out_dir: Path, seed: int, max_eval: int) -> None:
    for (dataset, split), all_samples in loaded.items():
        if split != "dev":
            continue
        samples = all_samples[: min(max_eval, len(all_samples))]
        variants = context_variants(samples, seed)
        for variant_name, contexts in variants.items():
            records = []
            for sample, context in zip(samples, contexts):
                record = sample.normalized_record()
                record["context_variant"] = variant_name
                record["context"] = context
                records.append(record)
            write_jsonl(out_dir / "context_variants" / dataset / f"{split}_{variant_name}.jsonl", records)
