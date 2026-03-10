#!/usr/bin/env python
"""Utility to (re)generate difficulty labels for benchmark tasks.

This script reads a tasks dataset JSON file and writes out a new JSON file
with a "difficulty" field added to each row. 

Usage (from repo root):

    python dataset/add_difficulty.py

You can also override input/output paths:

    python dataset/add_difficulty.py \
        --input dataset/tasks.json \
        --output dataset/tasks_with_difficulty.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


ERROR_KEYWORDS = ("invalid", "error", "not found", "unavailable")

COMPLEX_APP_TYPES = {
    "Analytics Platforms/Dashboards",
    "E-commerce Platforms",
    "Social Media Platforms",
    "Project Management Tools",
    "Real-time Collaboration Tools",
    "Marketplaces",
}

HARD_PRIMARY_CATEGORIES = {
    "Data Management",
    "Transaction Processing",
    "Complex Workflows",
}

HARDNESS_KEYWORDS = [
    "real-time",
    "dashboard",
    "analytics",
    "multi-company",
    "multi tenant",
    "multi-tenant",
    "role-based",
    "authentication",
    "authorization",
    "payment",
    "e-commerce",
    "collaboration",
    "workflow",
    "comparison",
    "analytics platform",
]


def _count_hardness_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in HARDNESS_KEYWORDS if kw in text_lower)


def difficulty_for_row(row: Dict[str, Any]) -> str:
    """Compute a difficulty label (easy/medium/hard) for a single task row."""

    score = 0

    ui = row.get("ui_instruct") or []
    backend = row.get("backend_test_cases") or []
    data_structures = row.get("data_structures") or []

    ui_count = len(ui)
    backend_count = len(backend)
    ds_count = len(data_structures)

    # UI tasks
    if ui_count >= 5:
        score += 1
    if ui_count >= 8:
        score += 1

    # Backend test cases
    if backend_count >= 3:
        score += 1
    if backend_count >= 5:
        score += 1

    # Data structures
    if ds_count >= 3:
        score += 1
    if ds_count >= 5:
        score += 1

    # Presence of explicit negative/error cases in backend expectations
    for case in backend:
        expected = str(case.get("expected_result", "")).lower()
        if any(kw in expected for kw in ERROR_KEYWORDS):
            score += 1
            break

    app_type = (row.get("application_type") or "").strip()
    if app_type in COMPLEX_APP_TYPES:
        score += 1

    category = row.get("Category") or {}
    primary_cat = (category.get("primary_category") or "").strip()
    if primary_cat in HARD_PRIMARY_CATEGORIES:
        score += 1

    instruction = row.get("instruction") or ""
    score += _count_hardness_keywords(instruction)

    # Map numeric score to buckets; tweak thresholds as needed.
    if score <= 3:
        return "easy"
    if score <= 6:
        return "medium"
    return "hard"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add difficulty labels to tasks dataset")
    default_input = Path(__file__).with_name("tasks.json")
    default_output = Path(__file__).with_name("tasks_with_difficulty.json")

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help=f"Path to input tasks.json (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Path to output JSON with difficulty labels (default: {default_output})",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("Expected top-level object with a 'rows' list")

    counts: Counter[str] = Counter()

    for entry in rows:
        if not isinstance(entry, dict):
            continue
        row = entry.get("row")
        if not isinstance(row, dict):
            continue
        label = difficulty_for_row(row)
        row["difficulty"] = label
        counts[label] += 1

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Difficulty distribution:", dict(counts))
    print("Wrote", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
