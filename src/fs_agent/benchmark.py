"""Benchmark runner — executes every orchestration pattern against a task dataset.

Reads tasks from ``dataset/tasks.json``, runs each task through every
orchestration pattern, and records rich metrics (token usage, wall-clock
time, communication overhead, agent-level timings, etc.).

Results are written as both JSON and CSV to the ``results/`` directory
inside the configured artifact root.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import Settings, get_settings
from .context import AgentReport, RunContext
from .llm import BaseLLMClient, build_llm_client, build_llm_clients_from_env
from .logger import get_logger
from .orchestration import (
    AgentRegistry,
    CentralizedOrchestrator,
    DecentralizedOrchestrator,
    HierarchicalOrchestrator,
    IterativeRefinementOrchestrator,
    ParallelOrchestrator,
    SequentialOrchestrator,
    register_default_agents,
)

logger = get_logger(__name__)

ALL_PATTERNS: list[str] = [
    "sequential",
    "centralized",
    "decentralized",
    "hierarchical",
    "parallel",
    "iterative",
]


# ---------------------------------------------------------------------------
# Metrics data-classes
# ---------------------------------------------------------------------------

@dataclass
class AgentMetrics:
    """Per-agent timing and status captured during a single run."""

    role: str
    status: str
    duration_seconds: float
    artifact_count: int
    attachment_count: int


@dataclass
class RunMetrics:
    """Aggregated metrics for a single (task × pattern) run."""

    task_id: str
    task_instruction: str
    pattern: str
    success: bool
    error: str | None = None

    # Timing
    wall_clock_seconds: float = 0.0
    agent_total_seconds: float = 0.0
    orchestration_overhead_seconds: float = 0.0

    # Token usage (from LLM client)
    llm_call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Agent-level breakdown
    agent_count: int = 0
    agents: list[AgentMetrics] = field(default_factory=list)

    # Communication overhead (inter-agent messages / coordinator calls)
    coordinator_calls: int = 0

    # Output metadata
    artifact_dir: str = ""
    started_at: str = ""
    finished_at: str = ""


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def _build_pattern(
    name: str, registry: AgentRegistry, llm: BaseLLMClient
) -> Any:
    """Instantiate the appropriate OrchestrationPattern."""
    if name == "sequential":
        return SequentialOrchestrator(registry=registry)
    if name == "centralized":
        return CentralizedOrchestrator(registry=registry, llm=llm)
    if name == "decentralized":
        return DecentralizedOrchestrator(registry=registry, llm=llm)
    if name == "hierarchical":
        return HierarchicalOrchestrator(registry=registry, llm=llm)
    if name == "parallel":
        return ParallelOrchestrator(registry=registry)
    if name == "iterative":
        return IterativeRefinementOrchestrator(registry=registry, llm=llm)
    raise ValueError(f"Unknown pattern: {name}")


def _count_coordinator_calls(reports: Sequence[AgentReport]) -> int:
    """Heuristic: count metadata entries that hint at coordinator / handoff calls."""
    count = 0
    for r in reports:
        count += r.metadata.get("coordinator_calls", 0)
        count += r.metadata.get("handoff_calls", 0)
    return count


def run_single(
    task_id: str,
    instruction: str,
    pattern: str,
    artifact_dir: Path,
    settings: Settings,
    llm: BaseLLMClient,
    llm_per_role: dict[str, BaseLLMClient] | None = None,
) -> RunMetrics:
    """Execute one (task × pattern) combination and return metrics."""

    run_artifact_dir = artifact_dir / task_id / pattern
    run_artifact_dir.mkdir(parents=True, exist_ok=True)

    # Set up a per-run file logger
    run_log_path = run_artifact_dir / "run.log"
    file_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)-30s %(levelname)-8s %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Reset token counters for this run
    llm.reset_usage()

    started_at = datetime.now(timezone.utc)
    start_wall = time.perf_counter()

    logger.info(
        "▶ BENCHMARK  task=%s  pattern=%-14s  dir=%s",
        task_id, pattern, run_artifact_dir,
    )

    metrics = RunMetrics(
        task_id=task_id,
        task_instruction=instruction[:200],
        pattern=pattern,
        success=False,
        artifact_dir=str(run_artifact_dir),
        started_at=started_at.isoformat(),
    )

    try:
        run_settings = settings.model_copy(
            update={
                "artifact_dir": run_artifact_dir,
                "orchestration_pattern": pattern,
            }
        )

        # Build per-role LLM overrides so agents pick up their
        # individually-configured providers from the environment.

        context = RunContext(
            spec=None,
            user_request=instruction,
            settings=run_settings,
            workspace_dir=Path.cwd(),
            artifact_dir=run_artifact_dir,
            llm=llm,
            llm_per_role=llm_per_role or {},
        )

        registry = AgentRegistry()
        register_default_agents(registry)
        orchestrator = _build_pattern(pattern, registry, llm)

        reports: list[AgentReport] = list(orchestrator.run(context))

        # --- Collect agent-level metrics ---
        agent_metrics: list[AgentMetrics] = []
        agent_total = 0.0
        for r in reports:
            dur = (r.finished_at - r.started_at).total_seconds()
            agent_total += dur
            agent_metrics.append(
                AgentMetrics(
                    role=r.role,
                    status=r.status,
                    duration_seconds=round(dur, 3),
                    artifact_count=len(r.artifacts),
                    attachment_count=len(r.metadata.get("attachments", [])),
                )
            )

        wall_clock = time.perf_counter() - start_wall
        usage = llm.usage_stats

        metrics.success = True
        metrics.wall_clock_seconds = round(wall_clock, 3)
        metrics.agent_total_seconds = round(agent_total, 3)
        metrics.orchestration_overhead_seconds = round(
            max(wall_clock - agent_total, 0), 3
        )
        metrics.llm_call_count = usage["call_count"]
        metrics.prompt_tokens = usage["prompt_tokens"]
        metrics.completion_tokens = usage["completion_tokens"]
        metrics.total_tokens = usage["total_tokens"]
        metrics.agent_count = len(reports)
        metrics.agents = agent_metrics
        metrics.coordinator_calls = _count_coordinator_calls(reports)

    except Exception as exc:
        metrics.wall_clock_seconds = round(time.perf_counter() - start_wall, 3)
        metrics.error = f"{type(exc).__name__}: {exc}"
        logger.exception("✗ BENCHMARK FAILED  task=%s  pattern=%s", task_id, pattern)

    finished_at = datetime.now(timezone.utc)
    metrics.finished_at = finished_at.isoformat()

    logger.info(
        "■ BENCHMARK  task=%s  pattern=%-14s  %.2fs  tokens=%d  agents=%d  %s",
        task_id, pattern, metrics.wall_clock_seconds,
        metrics.total_tokens, metrics.agent_count,
        "OK" if metrics.success else "FAIL",
    )

    # Remove the per-run file handler
    root_logger.removeHandler(file_handler)
    file_handler.close()

    return metrics


# ---------------------------------------------------------------------------
# Full benchmark loop
# ---------------------------------------------------------------------------

def load_tasks(dataset_path: Path) -> list[dict[str, Any]]:
    """Load tasks from the dataset JSON file."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = raw.get("rows", raw if isinstance(raw, list) else [])
    tasks: list[dict[str, Any]] = []
    for entry in rows:
        row = entry.get("row", entry)
        tasks.append(
            {
                "id": str(row.get("id", row.get("row_idx", len(tasks)))),
                "instruction": row["instruction"],
            }
        )
    return tasks


def run_benchmark(
    dataset_path: Path,
    *,
    patterns: list[str] | None = None,
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
    artifact_root: Path | None = None,
) -> list[RunMetrics]:
    """Run the full benchmark and write results to disk.

    Parameters
    ----------
    dataset_path:
        Path to ``dataset/tasks.json``.
    patterns:
        Subset of patterns to benchmark (default: all six).
    task_ids:
        If given, only run these task IDs.
    max_tasks:
        Cap on the number of tasks to process (useful for quick tests).
    artifact_root:
        Root directory for benchmark artefacts.  Defaults to ``artifacts/benchmark/``.
    """

    from dotenv import load_dotenv

    load_dotenv()

    get_settings.cache_clear()
    base_settings = get_settings()

    patterns = patterns or list(ALL_PATTERNS)
    artifact_root = artifact_root or Path("artifacts") / "benchmark"
    artifact_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(dataset_path)
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    logger.info(
        "Benchmark starting: %d tasks × %d patterns = %d runs",
        len(tasks), len(patterns), len(tasks) * len(patterns),
    )

    # Build shared + per-role LLM clients from env vars
    llm, llm_per_role = build_llm_clients_from_env(
        default_provider=base_settings.llm_provider,
        default_model=base_settings.llm_model,
        default_api_key=base_settings.openai_api_key,
        default_base_url=base_settings.llm_base_url,
    )

    all_metrics: list[RunMetrics] = []

    for task_idx, task in enumerate(tasks, 1):
        for pat_idx, pattern in enumerate(patterns, 1):
            logger.info(
                "━━━ Task %d/%d  Pattern %d/%d (%s)  id=%s ━━━",
                task_idx, len(tasks), pat_idx, len(patterns), pattern, task["id"],
            )
            metrics = run_single(
                task_id=task["id"],
                instruction=task["instruction"],
                pattern=pattern,
                artifact_dir=artifact_root,
                settings=base_settings,
                llm=llm,
                llm_per_role=llm_per_role,
            )
            all_metrics.append(metrics)

    # Write results
    results_dir = artifact_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json_report(all_metrics, results_dir / "benchmark_results.json")
    _write_csv_report(all_metrics, results_dir / "benchmark_results.csv")
    _write_summary(all_metrics, results_dir / "benchmark_summary.json")

    logger.info(
        "Benchmark complete. Results written to %s", results_dir,
    )
    return all_metrics


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_json_report(metrics: list[RunMetrics], path: Path) -> None:
    """Write full metrics to a JSON file."""
    data = [asdict(m) for m in metrics]
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote JSON report → %s", path)


def _write_csv_report(metrics: list[RunMetrics], path: Path) -> None:
    """Write a flat CSV of top-level metrics (no nested agent breakdown)."""
    fieldnames = [
        "task_id",
        "pattern",
        "success",
        "error",
        "wall_clock_seconds",
        "agent_total_seconds",
        "orchestration_overhead_seconds",
        "llm_call_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "agent_count",
        "coordinator_calls",
        "started_at",
        "finished_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in metrics:
            row = asdict(m)
            row.pop("agents", None)
            row.pop("task_instruction", None)
            row.pop("artifact_dir", None)
            writer.writerow(row)
    logger.info("Wrote CSV report → %s", path)


def _write_summary(metrics: list[RunMetrics], path: Path) -> None:
    """Write a per-pattern aggregate summary."""
    from collections import defaultdict

    buckets: dict[str, list[RunMetrics]] = defaultdict(list)
    for m in metrics:
        buckets[m.pattern].append(m)

    summary: dict[str, Any] = {}
    for pattern, runs in buckets.items():
        successful = [r for r in runs if r.success]
        failed = [r for r in runs if not r.success]
        wall_times = [r.wall_clock_seconds for r in successful]
        token_totals = [r.total_tokens for r in successful]
        overhead = [r.orchestration_overhead_seconds for r in successful]
        llm_calls = [r.llm_call_count for r in successful]

        summary[pattern] = {
            "total_runs": len(runs),
            "successful": len(successful),
            "failed": len(failed),
            "avg_wall_clock_seconds": round(_safe_avg(wall_times), 3),
            "min_wall_clock_seconds": round(min(wall_times), 3) if wall_times else None,
            "max_wall_clock_seconds": round(max(wall_times), 3) if wall_times else None,
            "avg_total_tokens": round(_safe_avg(token_totals), 1),
            "avg_prompt_tokens": round(
                _safe_avg([r.prompt_tokens for r in successful]), 1
            ),
            "avg_completion_tokens": round(
                _safe_avg([r.completion_tokens for r in successful]), 1
            ),
            "avg_llm_calls": round(_safe_avg(llm_calls), 1),
            "avg_orchestration_overhead_seconds": round(_safe_avg(overhead), 3),
            "avg_agent_count": round(
                _safe_avg([r.agent_count for r in successful]), 1
            ),
            "avg_coordinator_calls": round(
                _safe_avg([r.coordinator_calls for r in successful]), 1
            ),
        }

    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote summary → %s", path)


def _safe_avg(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0
