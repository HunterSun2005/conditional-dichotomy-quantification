"""Lightweight lexical proxy metrics used before model checkpoints are available."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Dict, List, Sequence, Tuple

from config import TOKEN_RE
from data_io import Sample


def tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def jaccard(a: str, b: str) -> float:
    left = set(tokens(a))
    right = set(tokens(b))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cosine_distance(a: str, b: str) -> float:
    ca = Counter(tokens(a))
    cb = Counter(tokens(b))
    if not ca or not cb:
        return 1.0
    dot = sum(value * cb.get(key, 0) for key, value in ca.items())
    norm_a = math.sqrt(sum(value * value for value in ca.values()))
    norm_b = math.sqrt(sum(value * value for value in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def evaluate_bow(samples: Sequence[Sample], contexts: Sequence[str] | None = None) -> Tuple[Dict[str, float], List[Dict]]:
    rows = []
    dcf = dcf_p = dcf_n = 0
    pos_neg_values: List[float] = []
    pos_neu_values: List[float] = []
    neg_neu_values: List[float] = []
    use_contexts = contexts if contexts is not None else [sample.context for sample in samples]

    for sample, context in zip(samples, use_contexts):
        cp = f"{context} {sample.positive}".strip()
        cn = f"{context} {sample.negative}".strip()
        cu = f"{context} {sample.neutral}".strip()
        pos_neutral = cosine_distance(cp, cu)
        pos_neg = cosine_distance(cp, cn)
        neg_neutral = cosine_distance(cn, cu)
        ok_p = pos_neutral < pos_neg
        ok_n = neg_neutral < pos_neg
        ok = ok_p and ok_n
        dcf += int(ok)
        dcf_p += int(ok_p)
        dcf_n += int(ok_n)
        pos_neg_values.append(pos_neg)
        pos_neu_values.append(pos_neutral)
        neg_neu_values.append(neg_neutral)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "split": sample.split,
                "dcf_pass": ok,
                "dcf_positive_pass": ok_p,
                "dcf_negative_pass": ok_n,
                "pos_neutral_distance": round(pos_neutral, 6),
                "pos_neg_distance": round(pos_neg, 6),
                "neg_neutral_distance": round(neg_neutral, 6),
                "dcf_margin": round(pos_neg - max(pos_neutral, neg_neutral), 6),
            }
        )

    total = len(samples) or 1
    return {
        "DCF": dcf / total,
        "DCF-positive": dcf_p / total,
        "DCF-negative": dcf_n / total,
        "pos_neg_degree": mean(pos_neg_values) if pos_neg_values else 0.0,
        "pos_neutral_degree": mean(pos_neu_values) if pos_neu_values else 0.0,
        "neg_neutral_degree": mean(neg_neu_values) if neg_neu_values else 0.0,
    }, rows
