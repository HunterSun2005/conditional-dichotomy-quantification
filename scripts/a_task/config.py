"""Member A artifact generation configuration."""

from __future__ import annotations

import re


DATASET_FILES = {
    "debate": {
        "train": "data/debate/perspectrum_train_selected_from_B_C.jsonl",
        "dev": "data/debate/perspectrum_dev_processed_gpt-4o.jsonl",
        "test": "data/debate/perspectrum_test_processed_gpt-4o.jsonl",
    },
    "defeasible_nli": {
        "train": "data/defeasible_nli/train.jsonl",
        "dev": "data/defeasible_nli/dev_processed_gpt-4o.jsonl",
    },
    "causal_reasoning": {
        "train": "data/causal_reasoning/deltaCausal_train_selected_from_A_B.jsonl",
        "dev": "data/causal_reasoning/deltaCausal_dev_processed_gpt-4o.jsonl",
        "test": "data/causal_reasoning/deltaCausal_test_processed_gpt-4o.jsonl",
    },
}

FIELD_MAP = {
    "context_text": "context",
    "supporter_text": "positive",
    "defeater_text": "negative",
    "neutral_text": "neutral",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
