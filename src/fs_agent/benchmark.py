"""Benchmark runner — executes every orchestration pattern against a task dataset.

Reads tasks from ``dataset/tasks_with_difficulty.json``, runs each task through every
orchestration pattern, and records rich metrics (token usage, wall-clock
time, coordination overhead, agent-level timings, etc.).

Results are written as both JSON and CSV to the ``results/`` directory
inside the configured artifact root.  Output is designed to answer:

* **RQ1** — Task completion & quality (success rate, wall-clock time)
* **RQ2** — Performance vs resource trade-off (total tokens, cost estimate)
* **RQ3** — Coordination overhead (coordination vs functional tokens, ratio)
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    OrchestrationError,
    OrchestrationMetrics,
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempt: int = 1


@dataclass
class RunMetrics:
    """Aggregated metrics for a single (task × pattern) run.

    Fields are laid out to answer the three research questions:

    * **RQ1** — ``success``, ``wall_clock_seconds``, ``agent_count``
    * **RQ2** — ``total_tokens``, ``cost_estimate``
    * **RQ3** — ``functional_*_tokens``, ``coordination_*_tokens``,
      ``coordination_to_functional_ratio``, ``coordination_call_count``
    """

    task_id: str
    task_instruction: str
    pattern: str
    success: bool
    error: str | None = None

    # RQ1 — Timing
    wall_clock_seconds: float = 0.0
    agent_total_seconds: float = 0.0
    orchestration_overhead_seconds: float = 0.0

    # RQ2 — Token usage & cost
    llm_call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0

    # RQ3 — Coordination vs Functional breakdown
    functional_prompt_tokens: int = 0
    functional_completion_tokens: int = 0
    functional_total_tokens: int = 0
    coordination_prompt_tokens: int = 0
    coordination_completion_tokens: int = 0
    coordination_total_tokens: int = 0
    coordination_call_count: int = 0
    coordination_to_functional_ratio: float = 0.0
    coordination_overhead_seconds: float = 0.0

    # Agent-level breakdown
    agent_count: int = 0
    agents: list[AgentMetrics] = field(default_factory=list)

    # Output metadata
    artifact_dir: str = ""
    started_at: str = ""
    finished_at: str = ""

    # Full orchestration metrics dict (for JSON report)
    orchestration_metrics: dict[str, Any] = field(default_factory=dict)


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
        return ParallelOrchestrator(registry=registry, llm=llm)
    raise ValueError(f"Unknown pattern: {name}")


def run_single(
    task_id: str,
    instruction: str,
    pattern: str,
    artifact_dir: Path,
    settings: Settings,
    llm: BaseLLMClient | None = None,
    llm_per_role: dict[str, BaseLLMClient] | None = None,
) -> RunMetrics:
    """Execute one (task × pattern) combination and return metrics.

    When *llm* is ``None`` (the default in parallel mode), fresh LLM clients
    are built from environment variables so each worker has isolated token
    counters.  When provided, the caller-supplied client is used (sequential
    mode backward-compat).

    Orchestration metrics are captured automatically via ``context.metrics``
    (an :class:`OrchestrationMetrics` instance that each pattern populates).
    If the pattern raises :class:`OrchestrationError` the run is marked as
    failed and the error is recorded — no silent fallback.
    """

    # Build per-run LLM clients when none are supplied (parallel mode).
    if llm is None:
        llm, llm_per_role = build_llm_clients_from_env(
            default_provider=settings.llm_provider,
            default_model=settings.llm_model,
            default_api_key=settings.openai_api_key,
            default_base_url=settings.llm_base_url,
        )

    run_artifact_dir = artifact_dir / task_id / pattern
    run_artifact_dir.mkdir(parents=True, exist_ok=True)

    # Set up a per-run file logger (use a dedicated logger, not root,
    # so parallel workers don't interfere with each other).
    run_logger_name = f"fs_agent.benchmark.run.{task_id}.{pattern}"
    run_logger = logging.getLogger(run_logger_name)
    run_log_path = run_artifact_dir / "run.log"
    file_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)-30s %(levelname)-8s %(message)s")
    )
    run_logger.addHandler(file_handler)
    run_logger.setLevel(logging.DEBUG)

    # Reset token counters for this run
    llm.reset_usage()
    if llm_per_role:
        for role_llm in llm_per_role.values():
            role_llm.reset_usage()

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

        # --- Populate from OrchestrationMetrics ---
        om = context.metrics
        wall_clock = time.perf_counter() - start_wall

        metrics.success = om.success
        metrics.wall_clock_seconds = round(wall_clock, 3)
        metrics.agent_total_seconds = round(
            sum(e.duration_seconds for e in om.agent_executions), 3
        )
        metrics.orchestration_overhead_seconds = round(
            max(wall_clock - metrics.agent_total_seconds, 0), 3
        )

        # RQ2 — Total token usage
        usage = llm.usage_stats
        metrics.llm_call_count = usage["call_count"]
        metrics.prompt_tokens = om.functional_prompt_tokens + om.coordination_prompt_tokens
        metrics.completion_tokens = om.functional_completion_tokens + om.coordination_completion_tokens
        metrics.total_tokens = om.total_tokens
        metrics.cost_estimate = round(om.cost_estimate(), 6)

        # RQ3 — Coordination vs Functional breakdown
        metrics.functional_prompt_tokens = om.functional_prompt_tokens
        metrics.functional_completion_tokens = om.functional_completion_tokens
        metrics.functional_total_tokens = om.functional_total_tokens
        metrics.coordination_prompt_tokens = om.coordination_prompt_tokens
        metrics.coordination_completion_tokens = om.coordination_completion_tokens
        metrics.coordination_total_tokens = om.coordination_total_tokens
        metrics.coordination_call_count = om.coordination_call_count
        metrics.coordination_to_functional_ratio = round(
            om.coordination_to_functional_ratio, 6
        )
        metrics.coordination_overhead_seconds = round(
            om.coordination_overhead_seconds, 3
        )

        # Agent-level breakdown
        agent_metrics: list[AgentMetrics] = []
        for ex in om.agent_executions:
            agent_metrics.append(
                AgentMetrics(
                    role=ex.role,
                    status=ex.status,
                    duration_seconds=round(ex.duration_seconds, 3),
                    artifact_count=ex.artifact_count,
                    attachment_count=ex.attachment_count,
                    prompt_tokens=ex.prompt_tokens,
                    completion_tokens=ex.completion_tokens,
                    total_tokens=ex.total_tokens,
                    attempt=ex.attempt,
                )
            )
        metrics.agent_count = len(reports)
        metrics.agents = agent_metrics

        # Store full orchestration metrics dict for the JSON report
        metrics.orchestration_metrics = om.to_dict()

    except OrchestrationError as exc:
        metrics.wall_clock_seconds = round(time.perf_counter() - start_wall, 3)
        metrics.error = f"OrchestrationError[{exc.pattern}]: {exc.reason}"
        metrics.success = False
        logger.error(
            "✗ BENCHMARK ORCHESTRATION ERROR  task=%s  pattern=%s  reason=%s",
            task_id, pattern, exc.reason,
        )

    except Exception as exc:
        metrics.wall_clock_seconds = round(time.perf_counter() - start_wall, 3)
        metrics.error = f"{type(exc).__name__}: {exc}"
        metrics.success = False
        logger.exception("✗ BENCHMARK FAILED  task=%s  pattern=%s", task_id, pattern)

    finished_at = datetime.now(timezone.utc)
    metrics.finished_at = finished_at.isoformat()

    logger.info(
        "■ BENCHMARK  task=%s  pattern=%-14s  %.2fs  "
        "tokens=%d (func=%d coord=%d ratio=%.4f)  "
        "agents=%d  cost=$%.6f  %s",
        task_id, pattern, metrics.wall_clock_seconds,
        metrics.total_tokens, metrics.functional_total_tokens,
        metrics.coordination_total_tokens,
        metrics.coordination_to_functional_ratio,
        metrics.agent_count, metrics.cost_estimate,
        "OK" if metrics.success else "FAIL",
    )

    # Remove the per-run file handler
    run_logger.removeHandler(file_handler)
    file_handler.close()

    return metrics


# ---------------------------------------------------------------------------
# Full benchmark loop
# ---------------------------------------------------------------------------

def load_tasks(dataset_path: Path) -> list[dict[str, Any]]:
    """Load tasks from the dataset JSON file.

    Preserves all task fields (ui_instruct, backend_test_cases,
    data_structures, difficulty, etc.) for downstream evaluation.
    """
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = raw.get("rows", raw if isinstance(raw, list) else [])
    tasks: list[dict[str, Any]] = []
    for entry in rows:
        row = entry.get("row", entry)
        task = dict(row)
        task["id"] = str(task.get("id", task.get("row_idx", len(tasks))))
        tasks.append(task)
    return tasks


def run_benchmark(
    dataset_path: Path,
    *,
    patterns: list[str] | None = None,
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
    artifact_root: Path | None = None,
    max_validation_retries: int | None = None,
    workers: int = 1,
) -> list[RunMetrics]:
    """Run the full benchmark and write results to disk.

    When *workers* > 1, task × pattern combinations are executed in
    parallel using a thread pool.  Each worker gets its own LLM client
    instances so token counters are isolated.
    """

    from dotenv import load_dotenv

    load_dotenv()

    get_settings.cache_clear()
    base_settings = get_settings()
    if max_validation_retries is not None:
        base_settings = base_settings.model_copy(
            update={"max_validation_retries": max_validation_retries}
        )

    patterns = patterns or list(ALL_PATTERNS)
    artifact_root = artifact_root or Path("artifacts") / "benchmark"
    artifact_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(dataset_path)
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    # Build the full work queue: list of (task, pattern) tuples.
    work_items: list[tuple[dict[str, Any], str]] = [
        (task, pattern)
        for task in tasks
        for pattern in patterns
    ]

    total_runs = len(work_items)
    logger.info(
        "Benchmark starting: %d tasks × %d patterns = %d runs  (workers=%d)",
        len(tasks), len(patterns), total_runs, workers,
    )

    results_dir = artifact_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[RunMetrics] = []
    _results_lock = threading.Lock()

    def _on_result(metrics: RunMetrics) -> None:
        """Thread-safe callback to collect a result and write incrementally."""
        with _results_lock:
            all_metrics.append(metrics)
            _write_json_report(all_metrics, results_dir / "benchmark_results.json")
            _write_csv_report(all_metrics, results_dir / "benchmark_results.csv")

    if workers <= 1:
        # Sequential execution — reuse a single shared LLM client for
        # backward-compatibility and lower connection overhead.
        llm, llm_per_role = build_llm_clients_from_env(
            default_provider=base_settings.llm_provider,
            default_model=base_settings.llm_model,
            default_api_key=base_settings.openai_api_key,
            default_base_url=base_settings.llm_base_url,
        )
        for run_idx, (task, pattern) in enumerate(work_items, 1):
            logger.info(
                "━━━ Run %d/%d  task=%s  pattern=%s ━━━",
                run_idx, total_runs, task["id"], pattern,
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
            _on_result(metrics)
    else:
        # Parallel execution — each worker builds its own LLM clients.
        completed = 0

        def _worker(task: dict[str, Any], pattern: str) -> RunMetrics:
            return run_single(
                task_id=task["id"],
                instruction=task["instruction"],
                pattern=pattern,
                artifact_dir=artifact_root,
                settings=base_settings,
                # llm=None → run_single builds its own clients
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_key = {
                pool.submit(_worker, task, pattern): (task["id"], pattern)
                for task, pattern in work_items
            }
            for future in as_completed(future_to_key):
                task_id, pattern = future_to_key[future]
                completed += 1
                try:
                    metrics = future.result()
                except Exception as exc:
                    logger.error(
                        "✗ Worker crashed  task=%s  pattern=%s  %s",
                        task_id, pattern, exc,
                    )
                    metrics = RunMetrics(
                        task_id=task_id,
                        task_instruction="",
                        pattern=pattern,
                        success=False,
                        error=f"WorkerCrash: {exc}",
                    )
                logger.info(
                    "━━━ Completed %d/%d  task=%s  pattern=%s  %s ━━━",
                    completed, total_runs, task_id, pattern,
                    "OK" if metrics.success else "FAIL",
                )
                _on_result(metrics)

    # Final summary (requires all runs).
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
    """Write full metrics to a JSON file, including orchestration metrics."""
    data = []
    for m in metrics:
        d = asdict(m)
        # Include the full orchestration metrics dict at the top level
        d["orchestration_metrics"] = m.orchestration_metrics
        data.append(d)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote JSON report → %s", path)


def _write_csv_report(metrics: list[RunMetrics], path: Path) -> None:
    """Write a flat CSV with RQ-specific columns (no nested agent breakdown)."""
    fieldnames = [
        # Identification
        "task_id",
        "pattern",
        "success",
        "error",
        # RQ1 — Task completion & quality
        "wall_clock_seconds",
        "agent_total_seconds",
        "orchestration_overhead_seconds",
        "agent_count",
        # RQ2 — Performance vs resource trade-off
        "llm_call_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_estimate",
        # RQ3 — Coordination overhead
        "functional_prompt_tokens",
        "functional_completion_tokens",
        "functional_total_tokens",
        "coordination_prompt_tokens",
        "coordination_completion_tokens",
        "coordination_total_tokens",
        "coordination_call_count",
        "coordination_to_functional_ratio",
        "coordination_overhead_seconds",
        # Timestamps
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
            row.pop("orchestration_metrics", None)
            writer.writerow(row)
    logger.info("Wrote CSV report → %s", path)


def _write_summary(metrics: list[RunMetrics], path: Path) -> None:
    """Write a per-pattern aggregate summary with RQ-specific aggregates."""
    from collections import defaultdict

    buckets: dict[str, list[RunMetrics]] = defaultdict(list)
    for m in metrics:
        buckets[m.pattern].append(m)

    summary: dict[str, Any] = {}
    for pattern, runs in buckets.items():
        successful = [r for r in runs if r.success]
        failed = [r for r in runs if not r.success]

        # RQ1 — Task completion
        wall_times = [r.wall_clock_seconds for r in successful]

        # RQ2 — Resource usage
        token_totals = [r.total_tokens for r in successful]
        costs = [r.cost_estimate for r in successful]

        # RQ3 — Coordination overhead
        func_tokens = [r.functional_total_tokens for r in successful]
        coord_tokens = [r.coordination_total_tokens for r in successful]
        coord_ratios = [r.coordination_to_functional_ratio for r in successful]
        coord_calls = [r.coordination_call_count for r in successful]
        coord_overhead = [r.coordination_overhead_seconds for r in successful]

        summary[pattern] = {
            # Identification
            "total_runs": len(runs),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": round(len(successful) / len(runs), 4) if runs else 0.0,

            # RQ1 — Task completion & quality
            "avg_wall_clock_seconds": round(_safe_avg(wall_times), 3),
            "min_wall_clock_seconds": round(min(wall_times), 3) if wall_times else None,
            "max_wall_clock_seconds": round(max(wall_times), 3) if wall_times else None,
            "avg_agent_count": round(
                _safe_avg([r.agent_count for r in successful]), 1
            ),

            # RQ2 — Performance vs resource trade-off
            "avg_total_tokens": round(_safe_avg(token_totals), 1),
            "avg_prompt_tokens": round(
                _safe_avg([r.prompt_tokens for r in successful]), 1
            ),
            "avg_completion_tokens": round(
                _safe_avg([r.completion_tokens for r in successful]), 1
            ),
            "avg_cost_estimate": round(_safe_avg(costs), 6),
            "total_cost_estimate": round(sum(costs), 6),

            # RQ3 — Coordination overhead
            "avg_functional_tokens": round(_safe_avg(func_tokens), 1),
            "avg_coordination_tokens": round(_safe_avg(coord_tokens), 1),
            "avg_coordination_to_functional_ratio": round(
                _safe_avg(coord_ratios), 6
            ),
            "avg_coordination_call_count": round(_safe_avg(coord_calls), 1),
            "avg_coordination_overhead_seconds": round(
                _safe_avg(coord_overhead), 3
            ),

            # Errors
            "errors": [r.error for r in failed if r.error],
        }

    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote summary → %s", path)


def _safe_avg(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0
