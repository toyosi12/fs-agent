#!/usr/bin/env python3
"""Visualize agent execution order from benchmark results.

Reads artifacts/benchmark/results/benchmark_results.json and prints run order
as plain text or Mermaid flowcharts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize agent execution order from benchmark_results.json"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/benchmark/results/benchmark_results.json"),
        help="Path to benchmark_results.json",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Filter by task id (e.g. 000001)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Filter by orchestration pattern (e.g. sequential)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "mermaid", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file; prints to stdout when omitted",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("benchmark_results.json must contain a JSON array")
    return data


def filter_rows(
    rows: list[dict[str, Any]], task_id: str | None, pattern: str | None
) -> list[dict[str, Any]]:
    filtered = rows
    if task_id:
        filtered = [r for r in filtered if str(r.get("task_id")) == task_id]
    if pattern:
        filtered = [r for r in filtered if str(r.get("pattern")) == pattern]
    return filtered


def get_roles(row: dict[str, Any]) -> list[str]:
    agents = row.get("agents", [])
    if not isinstance(agents, list):
        return []
    roles: list[str] = []
    for agent in agents:
        if isinstance(agent, dict) and isinstance(agent.get("role"), str):
            roles.append(agent["role"])
    return roles


def render_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id", "?"))
        pattern = str(row.get("pattern", "?"))
        roles = get_roles(row)
        if roles:
            order = " -> ".join(roles)
        else:
            error = row.get("error")
            order = f"(no agents recorded; error={error})" if error else "(no agents recorded)"
        lines.append(f"{task_id} | {pattern}: {order}")
    return "\n".join(lines)


def _node_id(task_id: str, pattern: str, index: int) -> str:
    safe_pattern = "".join(ch if ch.isalnum() else "_" for ch in pattern)
    return f"t{task_id}_{safe_pattern}_{index}"


def render_mermaid(rows: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id", "?"))
        pattern = str(row.get("pattern", "?"))
        roles = get_roles(row)

        blocks.append(f"### Task {task_id} - {pattern}")
        blocks.append("```mermaid")
        blocks.append("flowchart LR")

        if not roles:
            blocks.append(
                f"  { _node_id(task_id, pattern, 0) }[\"No agents recorded\"]"
            )
        else:
            for i, role in enumerate(roles):
                node = _node_id(task_id, pattern, i)
                blocks.append(f"  {node}[\"{i + 1}. {role}\"]")
                if i > 0:
                    prev = _node_id(task_id, pattern, i - 1)
                    blocks.append(f"  {prev} --> {node}")

        blocks.append("```")
        blocks.append("")

    return "\n".join(blocks).strip()


def build_output(rows: list[dict[str, Any]], fmt: str) -> str:
    if not rows:
        return "No matching runs found."

    parts: list[str] = []
    if fmt in ("text", "both"):
        parts.append("# Run Order (Text)")
        parts.append("")
        parts.append(render_text(rows))
    if fmt in ("mermaid", "both"):
        if parts:
            parts.append("")
        parts.append("# Run Order (Mermaid)")
        parts.append("")
        parts.append(render_mermaid(rows))
    return "\n".join(parts)


def main() -> int:
    args = parse_args()
    rows = load_results(args.results)
    rows = filter_rows(rows, args.task_id, args.pattern)
    output = build_output(rows, args.format)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
