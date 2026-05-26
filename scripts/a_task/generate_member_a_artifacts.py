#!/usr/bin/env python3
"""Generate all member A deliverables for the CDQ reproduction project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact_builder import e8_results, generate_subsets, motivation_cases, write_context_variant_files
from data_io import load_all, relpath, write_csv
from report_text import (
    data_report,
    data_splits_report,
    error_analysis_markdown,
    member_a_readme,
    metric_report,
    run_commands_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root containing data/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval", type=int, default=500, help="Maximum dev rows per dataset for proxy E8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    reports_dir = root / "reports" / "member_a"
    splits_dir = root / "data" / "member_a_splits"
    reports_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_all(root)

    report_md, review_rows = data_report(loaded)
    (reports_dir / "data_report.md").write_text(report_md, encoding="utf-8")
    write_csv(reports_dir / "sample_role_check.csv", review_rows)

    (reports_dir / "metric_report.md").write_text(metric_report(), encoding="utf-8")
    split_rows = generate_subsets(loaded, splits_dir, root, args.seed)
    (reports_dir / "data_splits_report.md").write_text(data_splits_report(split_rows), encoding="utf-8")

    write_csv(reports_dir / "motivation_cases.csv", motivation_cases(loaded, limit=5))

    e8_rows, errors = e8_results(loaded, reports_dir, args.seed, args.max_eval)
    write_csv(reports_dir / "error_cases.csv", errors)
    (reports_dir / "error_analysis.md").write_text(error_analysis_markdown(errors), encoding="utf-8")
    write_context_variant_files(loaded, splits_dir, args.seed, args.max_eval)

    (reports_dir / "run_commands.md").write_text(run_commands_markdown(), encoding="utf-8")
    (reports_dir / "README.md").write_text(member_a_readme(), encoding="utf-8")

    summary = {
        "loaded_files": len(loaded),
        "generated_split_rows": len(split_rows),
        "motivation_cases": 5,
        "context_sensitivity_rows": len(e8_rows),
        "error_cases": len(errors),
        "reports_dir": relpath(reports_dir, root),
        "splits_dir": relpath(splits_dir, root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
