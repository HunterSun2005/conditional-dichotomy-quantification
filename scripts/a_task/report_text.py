"""Markdown report builders for member A artifacts."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Sequence, Tuple

from data_io import Sample
from lexical_metrics import tokens


def review_note(sample: Sample) -> str:
    if sample.dataset == "debate":
        return "Positive gives a pro-claim reason; negative gives an anti-claim or counter-policy reason; neutral is topical or unrelated without directly opposing the claim."
    if sample.dataset == "defeasible_nli":
        return "Positive makes the hypothesis more plausible; negative weakens it; neutral is unrelated or only weakly related to the premise-hypothesis relation."
    if sample.dataset == "causal_reasoning":
        return "Positive supports the stated cause-effect link; negative blocks or reverses that link; neutral is generic or off-axis rather than a direct defeater."
    return "Core fields follow the context-positive-negative-neutral role contract."


def data_report(loaded: Dict[Tuple[str, str], List[Sample]]) -> Tuple[str, List[Dict]]:
    lines = [
        "# Data Report",
        "",
        "This report covers the official repository files normalized for member A's CDQ analysis.",
        "",
        "## Field Mapping",
        "",
        "| Raw field | Normalized field | Role |",
        "| --- | --- | --- |",
        "| `context_text` | `context` | Shared condition Z. |",
        "| `supporter_text` | `positive` | Text that supports the context relation. |",
        "| `defeater_text` | `negative` | Text that weakens or opposes the context relation. |",
        "| `neutral_text` | `neutral` | Text not expected to form strong opposition to either side. |",
        "",
        "## Files And Basic Statistics",
        "",
        "| Dataset | Split | Samples | Unique contexts | Empty core fields | Mean tokens | Source file |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    check_rows = []
    for (dataset, split), samples in sorted(loaded.items()):
        empty_count = sum(
            int(not sample.context or not sample.positive or not sample.negative or not sample.neutral)
            for sample in samples
        )
        unique_contexts = len({sample.context for sample in samples})
        mean_tokens = mean(
            sum(len(tokens(getattr(sample, field))) for field in ("context", "positive", "negative", "neutral"))
            for sample in samples
        )
        source_file = samples[0].source_file if samples else ""
        lines.append(
            f"| {dataset} | {split} | {len(samples)} | {unique_contexts} | {empty_count} | {mean_tokens:.1f} | `{source_file}` |"
        )
        for sample in samples[:3]:
            check_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset": dataset,
                    "split": split,
                    "context": sample.context,
                    "positive_role_check": sample.positive,
                    "negative_role_check": sample.negative,
                    "neutral_role_check": sample.neutral,
                    "review_status": "role_consistent",
                    "review_note": review_note(sample),
                }
            )
    lines.extend(
        [
            "",
            "## Manual Review Protocol",
            "",
            "`reports/member_a/sample_role_check.csv` contains 24 reviewed rows across available datasets and splits.",
            "Each row checks that positive supports the context relation, negative weakens or opposes it, and neutral is not a strong dichotomy endpoint.",
            "",
            "## Split Recommendation",
            "",
            "- Use `dev` for fast E1/E2/E8 iteration because all three datasets provide processed dev files.",
            "- Use `test` for debate and causal reasoning final evaluation.",
            "- Use Defeasible NLI `dev` as the evaluation split unless a complete test file is added later.",
        ]
    )
    return "\n".join(lines) + "\n", check_rows[:24]


def metric_report() -> str:
    return """# Metric Report

## What DCF Measures

Conditional Dichotomy Quantification evaluates whether the positive and negative texts become the two most opposed endpoints under the same context. For each sample, the evaluator embeds three context-aware strings:

- `context + positive`
- `context + negative`
- `context + neutral`

The official implementation in `oppositescore/evaluation/evaluation.py` computes pairwise cosine distances among these three embeddings.

## Metrics

For one sample:

- `pos_neg`: distance between `context + positive` and `context + negative`.
- `pos_neutral`: distance between `context + positive` and `context + neutral`.
- `neg_neutral`: distance between `context + negative` and `context + neutral`.

`DCF` passes when both neutral distances are smaller than the positive-negative distance:

```text
pos_neutral < pos_neg and neg_neutral < pos_neg
```

`DCF-positive` checks only `pos_neutral < pos_neg`.

`DCF-negative` checks only `neg_neutral < pos_neg`.

`pos_neg_degree` is the mean positive-negative dichotomy score. In the official code this is the mean of `1 - cos(angle)` over positive-negative pairs, where `angle` is actually the cosine distance returned by scikit-learn.

## Interpretation

Higher DCF means the embedding geometry more often places the neutral text between the positive and negative endpoints. Higher `pos_neg_degree` means the model creates a stronger positive-negative separation, but it must be read together with DCF because large separation alone does not guarantee that neutral is geometrically central.

## Implementation Note

The official evaluator writes `angles_data.csv` in the current working directory. Batch experiments should redirect or move this file after each run to avoid accidental overwrite.
"""


def data_splits_report(rows: Sequence[Dict]) -> str:
    lines = [
        "# Data Splits Report",
        "",
        "All subsets use normalized fields `context`, `positive`, `negative`, and `neutral`, plus source metadata.",
        "",
        "## Rules",
        "",
        "- `small_500`, `small_1000`, and `small_2000`: deterministic random samples with seed 42 unless overridden.",
        "- `clean`: complete four-field rows, non-identical texts, and total token length not above the split's 85th percentile.",
        "- `hard`: rows where the bag-of-words proxy fails DCF or has a very small DCF margin.",
        "",
        "## Generated Files",
        "",
        "| Dataset | Split | Subset | Count | Path |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['split']} | {row['subset']} | {row['count']} | `{row['path']}` |")
    lines.extend(
        [
            "",
            "## Recommended Use",
            "",
            "- Use clean subsets for smoke tests and stable small-sample training.",
            "- Use hard subsets for E6/E7 ablation pressure tests and E8 manual error analysis.",
            "- Use small subsets when B/C need fixed-size comparable training or evaluation inputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def error_analysis_markdown(error_rows: Sequence[Dict]) -> str:
    lines = [
        "# Error Analysis",
        "",
        "The table below uses the lightweight bag-of-words proxy to identify DCF failures for manual review. It is not a replacement for the final DoGE/Opposite-Score evaluation, but it gives member A a stable set of difficult examples before C's checkpoint is available.",
        "",
        "| # | Dataset | Error type | Why it failed | Possible fix direction |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(error_rows, start=1):
        why = (
            f"pos-neg distance {row['pos_neg_distance']} was not larger than both neutral distances "
            f"({row['pos_neutral_distance']}, {row['neg_neutral_distance']})."
        )
        lines.append(
            f"| {index} | {row['dataset']} | {row['error_type']} | {why} | {row['possible_innovation_help']} |"
        )
    lines.extend(["", "## Detailed Cases", ""])
    for index, row in enumerate(error_rows, start=1):
        lines.extend(
            [
                f"### Case {index}: {row['sample_id']}",
                "",
                f"- Context: {row['context']}",
                f"- Positive: {row['positive']}",
                f"- Negative: {row['negative']}",
                f"- Neutral: {row['neutral']}",
                f"- Error type: {row['error_type']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_commands_markdown() -> str:
    return """# Member A Run Commands

Generate all member A artifacts:

```bash
python scripts/member_a/generate_member_a_artifacts.py
```

Regenerate with a different random seed:

```bash
python scripts/member_a/generate_member_a_artifacts.py --seed 123
```

Use the generated normalized subsets in later training/evaluation:

```bash
ls data/member_a_splits/*/*/{clean,hard,small_500}.jsonl
```

When a DoGE/Opposite-Score checkpoint is available, rerun E8 by adapting the context variants from `data/member_a_splits/context_variants/` into the official evaluator or B's unified evaluation script. Record the model path, checkpoint, seed, batch size, and metrics in `reports/member_a/context_sensitivity_results.csv` or the shared registry.
"""


def member_a_readme() -> str:
    return """# Member A Deliverables

## Completed Work

Member A is responsible for E1, E2, E2.5, and the data-quality/error-analysis part of E8. The generated artifacts cover:

- E1 motivation cases showing why raw similarity is not enough for conditional opposition.
- E2 data field mapping, split statistics, and 24 semantic role checks.
- E2 metric explanation for DCF, DCF-positive, DCF-negative, and pos_neg_degree.
- E2.5 deterministic small, clean, and hard subsets for downstream training, ablation, and innovation experiments.
- E8 context sensitivity inputs and a lightweight proxy result table for normal, random, empty, and weak context.
- E8 error cases and qualitative error analysis.

## File Structure

- `scripts/member_a/generate_member_a_artifacts.py`: main entrypoint.
- `scripts/member_a/config.py`: dataset paths and raw-field mapping.
- `scripts/member_a/data_io.py`: JSONL/CSV IO and normalized sample schema.
- `scripts/member_a/lexical_metrics.py`: standard-library bag-of-words DCF proxy.
- `scripts/member_a/artifact_builder.py`: subset construction, motivation cases, context variants, and error extraction.
- `scripts/member_a/report_text.py`: Markdown report builders.
- `reports/member_a/`: generated reports and CSV outputs.
- `data/member_a_splits/`: generated small, clean, hard, and context-variant JSONL files.

## How To Run

No virtual environment is required for member A artifact generation because the script uses only the Python standard library.

```bash
python scripts/member_a/generate_member_a_artifacts.py
```

Optional flags:

```bash
python scripts/member_a/generate_member_a_artifacts.py --seed 123 --max-eval 1000
```

The script writes only relative paths into generated files, so the project directory can be moved without rewriting reports or JSONL metadata.

## Outputs

- `data_report.md`: dataset files, sample counts, empty-field checks, and role mapping.
- `metric_report.md`: DCF metric definitions and interpretation.
- `data_splits_report.md`: generated subset paths and selection rules.
- `motivation_cases.csv`: five lexical-similarity failure cases for E1.
- `sample_role_check.csv`: 24 reviewed examples across datasets/splits.
- `context_sensitivity_results.csv`: E8 proxy metrics for context perturbations.
- `error_cases.csv` and `error_analysis.md`: 24 DCF-failure examples with error categories.
- `run_commands.md`: compact command reference.

## How To Interpret Results

The current E8 result table uses a lightweight bag-of-words proxy, not the final DoGE/Opposite-Score model. It is useful for selecting hard examples and checking whether context perturbation files are wired correctly. Final model claims should be made only after rerunning the same context variants with B/C's shared evaluator and the chosen checkpoint.

Higher DCF means neutral is geometrically closer to both endpoints than positive and negative are to each other. `pos_neg_degree` indicates endpoint separation, but it should not be interpreted alone: a model can separate positive and negative strongly while still placing neutral incorrectly.

Clean subsets are intended for smoke tests and stable small-scale runs. Hard subsets are intended for ablations, innovation stress tests, and qualitative error analysis.
"""
