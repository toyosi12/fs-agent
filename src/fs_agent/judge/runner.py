"""Judge runner — loads benchmark artifacts and evaluates them.

This module is fully isolated from the worker agents.  It reads the
already-generated project files from the benchmark artifact directory
and sends them to the GPT-4o judge for scoring.

Results are written to ``<artifact_root>/results/judge_results.json``
and ``judge_summary.json``, with every score linked back to the
originating task ID and difficulty level.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..benchmark import load_tasks
from ..llm import BaseLLMClient, OpenAILLMClient, OPENAI_BASE_URL
from ..logger import get_logger
from .executor import find_project_dir, start_project, stop_project
from .models import JudgeResult, TaskJudgeResult
from .runtime import evaluate_application_runtime
from .scoring import evaluate_application

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
# Artifact loader helpers
# ---------------------------------------------------------------------------


def _read_file_safe(path: Path) -> str:
    """Read a text file, returning empty string on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _collect_files_by_ext(directory: Path, extensions: set[str]) -> str:
    """Recursively collect file contents matching the given extensions."""
    if not directory.exists():
        return ""
    parts: list[str] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix in extensions:
            content = _read_file_safe(p)
            if content.strip():
                rel = p.relative_to(directory)
                parts.append(f"// === {rel} ===\n{content}")
    return "\n\n".join(parts)


def load_project_code(
    artifact_root: Path, task_id: str, pattern: str
) -> dict[str, str]:
    """Load generated frontend code, backend code, and migration SQL.

    Looks for the standard project layout under:
        ``<artifact_root>/<task_id>/<pattern>/projects/<slug>/``

    Returns a dict with keys: frontend_code, backend_code, migration_sql
    """
    run_dir = artifact_root / task_id / pattern
    projects_dir = run_dir / "projects"

    frontend_code = ""
    backend_code = ""
    migration_sql = ""

    if not projects_dir.exists():
        return {
            "frontend_code": frontend_code,
            "backend_code": backend_code,
            "migration_sql": migration_sql,
        }

    # Find the project slug directory (first child of projects/)
    project_dirs = [d for d in projects_dir.iterdir() if d.is_dir()]
    if not project_dirs:
        return {
            "frontend_code": frontend_code,
            "backend_code": backend_code,
            "migration_sql": migration_sql,
        }

    project_dir = project_dirs[0]

    # Frontend: collect all .jsx, .js, .css files under frontend/
    fe_dir = project_dir / "frontend"
    frontend_code = _collect_files_by_ext(fe_dir, {".jsx", ".js", ".css"})

    # Backend: collect all .js files under backend/src/
    be_dir = project_dir / "backend"
    be_src = be_dir / "src"
    backend_code = _collect_files_by_ext(be_src, {".js"})
    # Also include test files
    be_tests = be_dir / "__tests__"
    if be_tests.exists():
        backend_code += "\n\n" + _collect_files_by_ext(be_tests, {".js"})

    # Migration SQL: collect all .sql files under backend/migrations/
    migration_dir = be_dir / "migrations"
    migration_sql = _collect_files_by_ext(migration_dir, {".sql"})

    return {
        "frontend_code": frontend_code,
        "backend_code": backend_code,
        "migration_sql": migration_sql,
    }


# ---------------------------------------------------------------------------
# Build judge LLM client
# ---------------------------------------------------------------------------


def build_judge_llm() -> BaseLLMClient:
    """Construct the GPT-4o judge client from environment variables.

    Uses dedicated env vars so the judge is fully isolated from
    worker agent configuration:
    - ``FS_AGENT_JUDGE_API_KEY`` (required — OpenAI API key)
    - ``FS_AGENT_JUDGE_MODEL`` (default: ``gpt-4o``)
    - ``FS_AGENT_JUDGE_BASE_URL`` (default: OpenAI)
    """
    api_key = os.getenv("FS_AGENT_JUDGE_API_KEY")
    model = os.getenv("FS_AGENT_JUDGE_MODEL", "gpt-4o")
    base_url = os.getenv("FS_AGENT_JUDGE_BASE_URL", OPENAI_BASE_URL)

    if not api_key:
        raise RuntimeError(
            "FS_AGENT_JUDGE_API_KEY is required. Set it in .env or as an "
            "environment variable with your OpenAI API key."
        )

    return OpenAILLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=120.0,
        max_retries=5,
    )


# ---------------------------------------------------------------------------
# Main judge runner
# ---------------------------------------------------------------------------


def run_judge(
    dataset_path: Path,
    artifact_root: Path,
    *,
    patterns: list[str] | None = None,
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
    mode: str = "static",
) -> list[JudgeResult]:
    """Run the LLM judge across all benchmark outputs.

    Parameters
    ----------
    dataset_path:
        Path to ``dataset/tasks_with_difficulty.json``.
    artifact_root:
        Root of benchmark artifacts (e.g. ``artifacts/benchmark``).
    patterns:
        Subset of patterns to evaluate (default: all found on disk).
    task_ids:
        If given, only evaluate these task IDs.
    max_tasks:
        Cap on number of tasks to evaluate.
    mode:
        ``"static"`` — evaluate generated code via LLM code review only.
        ``"runtime"`` — boot projects in Docker, send real HTTP requests,
        check the SQLite schema, then evaluate.

    Returns a list of JudgeResult, one per (task × pattern).
    """
    from dotenv import load_dotenv

    load_dotenv()

    judge_llm = build_judge_llm()

    # Load task dataset for test cases and difficulty
    tasks = load_tasks(dataset_path)
    task_map: dict[str, dict] = {t["id"]: t for t in tasks}

    if task_ids:
        task_map = {k: v for k, v in task_map.items() if k in task_ids}
    if max_tasks is not None:
        task_map = dict(list(task_map.items())[:max_tasks])

    patterns = patterns or ALL_PATTERNS

    logger.info(
        "Judge starting: %d tasks × %d patterns (mode=%s)",
        len(task_map), len(patterns), mode,
    )
    print(f"[judge] Starting: {len(task_map)} tasks × {len(patterns)} patterns, model={judge_llm.model}, mode={mode}")

    all_results: list[JudgeResult] = []

    for task_id, task in task_map.items():
        instruction = task.get("instruction", "")
        difficulty = task.get("difficulty", "unknown")
        ui_tests = task.get("ui_instruct", [])
        backend_tests = task.get("backend_test_cases", [])
        data_structures = task.get("data_structures", [])

        for pattern in patterns:
            run_dir = artifact_root / task_id / pattern
            if not run_dir.exists():
                logger.info(
                    "Skipping task=%s pattern=%s (no artifacts found)", task_id, pattern
                )
                continue

            logger.info(
                "▶ JUDGE  task=%s  pattern=%s  difficulty=%s",
                task_id, pattern, difficulty,
            )
            print(f"[judge] ▶ Evaluating task={task_id}  pattern={pattern}  difficulty={difficulty}  mode={mode}")

            code = load_project_code(artifact_root, task_id, pattern)

            # Skip if no code was generated at all
            if not code["frontend_code"] and not code["backend_code"]:
                result = JudgeResult(
                    task_id=task_id,
                    pattern=pattern,
                    difficulty=difficulty,
                    judge_model=judge_llm.model,
                    error="No generated code found in artifacts",
                )
                all_results.append(result)
                continue

            try:
                # Snapshot token counters before evaluation
                _before = judge_llm.usage_stats

                if mode == "runtime":
                    result = _evaluate_runtime(
                        judge_llm, artifact_root, task_id, pattern,
                        difficulty, instruction, code,
                        ui_tests, backend_tests, data_structures,
                    )
                else:
                    result = evaluate_application(
                        llm=judge_llm,
                        task_id=task_id,
                        pattern=pattern,
                        difficulty=difficulty,
                        task_instruction=instruction,
                        frontend_code=code["frontend_code"],
                        backend_code=code["backend_code"],
                        migration_sql=code["migration_sql"],
                        ui_test_cases=ui_tests,
                        backend_test_cases=backend_tests,
                        data_structures=data_structures,
                    )

                # Record delta tokens consumed by this evaluation
                _after = judge_llm.usage_stats
                result.prompt_tokens = _after["prompt_tokens"] - _before["prompt_tokens"]
                result.completion_tokens = _after["completion_tokens"] - _before["completion_tokens"]
                result.total_tokens = _after["total_tokens"] - _before["total_tokens"]
            except Exception as exc:
                logger.exception(
                    "✗ JUDGE FAILED  task=%s  pattern=%s", task_id, pattern
                )
                print(f"[judge] ✗ FAILED task={task_id}  pattern={pattern}: {exc}")
                result = JudgeResult(
                    task_id=task_id,
                    pattern=pattern,
                    difficulty=difficulty,
                    judge_model=judge_llm.model,
                    error=f"{type(exc).__name__}: {exc}",
                )

            all_results.append(result)
            logger.info(
                "■ JUDGE  task=%s  pattern=%s  fe=%.2f  be=%.2f  db=%.2f  appearance=%.1f  tokens=%d",
                task_id,
                pattern,
                result.frontend_weighted_accuracy,
                result.backend_accuracy,
                result.database_accuracy,
                result.appearance.overall if result.appearance else 0.0,
                result.total_tokens,
            )
            print(
                f"[judge] ■ Done task={task_id}  pattern={pattern}"
                f"  fe={result.frontend_weighted_accuracy:.2f}"
                f"  be={result.backend_accuracy:.2f}"
                f"  db={result.database_accuracy:.2f}"
                f"  appearance={result.appearance.overall if result.appearance else 0.0:.1f}"
                f"  tokens={result.total_tokens}"
            )

    # Write results
    results_dir = artifact_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_judge_results(all_results, results_dir / "judge_results.json")
    _write_judge_summary(all_results, results_dir / "judge_summary.json")

    logger.info("Judge complete. Results written to %s", results_dir)
    return all_results


# ---------------------------------------------------------------------------
# Runtime evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_runtime(
    judge_llm: BaseLLMClient,
    artifact_root: Path,
    task_id: str,
    pattern: str,
    difficulty: str,
    instruction: str,
    code: dict[str, str],
    ui_tests: list[dict],
    backend_tests: list[dict],
    data_structures: list[str],
) -> JudgeResult:
    """Boot the project in Docker, run runtime tests, then tear down."""
    project_dir = find_project_dir(artifact_root, task_id, pattern)

    if not project_dir:
        return JudgeResult(
            task_id=task_id,
            pattern=pattern,
            difficulty=difficulty,
            judge_model=judge_llm.model,
            error="Project directory not found for runtime evaluation",
        )

    instance = start_project(project_dir)

    if not instance:
        return JudgeResult(
            task_id=task_id,
            pattern=pattern,
            difficulty=difficulty,
            judge_model=judge_llm.model,
            error="Failed to start Docker containers",
        )

    try:
        return evaluate_application_runtime(
            llm=judge_llm,
            instance=instance,
            task_id=task_id,
            pattern=pattern,
            difficulty=difficulty,
            task_instruction=instruction,
            frontend_code=code["frontend_code"],
            backend_code=code["backend_code"],
            migration_sql=code["migration_sql"],
            ui_test_cases=ui_tests,
            backend_test_cases=backend_tests,
            data_structures=data_structures,
        )
    finally:
        stop_project(project_dir)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_judge_results(results: list[JudgeResult], path: Path) -> None:
    """Write full judge results to JSON."""
    data = [r.model_dump(mode="json") for r in results]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Wrote judge results → %s", path)


def _write_judge_summary(results: list[JudgeResult], path: Path) -> None:
    """Write aggregated judge summary grouped by pattern and difficulty."""
    # Group by pattern
    by_pattern: dict[str, list[JudgeResult]] = defaultdict(list)
    for r in results:
        by_pattern[r.pattern].append(r)

    # Group by difficulty
    by_difficulty: dict[str, list[JudgeResult]] = defaultdict(list)
    for r in results:
        by_difficulty[r.difficulty].append(r)

    # Group by (pattern, difficulty)
    by_pattern_difficulty: dict[str, dict[str, list[JudgeResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        by_pattern_difficulty[r.pattern][r.difficulty].append(r)

    summary: dict[str, Any] = {
        "total_evaluations": len(results),
        "by_pattern": {},
        "by_difficulty": {},
        "by_pattern_difficulty": {},
    }

    for pattern, runs in by_pattern.items():
        summary["by_pattern"][pattern] = _aggregate_scores(runs)

    for difficulty, runs in by_difficulty.items():
        summary["by_difficulty"][difficulty] = _aggregate_scores(runs)

    for pattern, diff_map in by_pattern_difficulty.items():
        summary["by_pattern_difficulty"][pattern] = {}
        for difficulty, runs in diff_map.items():
            summary["by_pattern_difficulty"][pattern][difficulty] = _aggregate_scores(runs)

    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote judge summary → %s", path)


def _aggregate_scores(results: list[JudgeResult]) -> dict[str, Any]:
    """Compute aggregate scores for a group of results."""
    n = len(results)
    if n == 0:
        return {"count": 0}

    valid = [r for r in results if r.error is None]
    errored = [r for r in results if r.error is not None]

    fe_accs = [r.frontend_weighted_accuracy for r in valid]
    be_accs = [r.backend_accuracy for r in valid]
    db_accs = [r.database_accuracy for r in valid]
    appearances = [r.appearance.overall for r in valid if r.appearance]

    total_toks = [r.total_tokens for r in valid if r.total_tokens > 0]

    return {
        "count": n,
        "evaluated": len(valid),
        "errors": len(errored),
        "total_tokens": sum(r.total_tokens for r in results),
        "tokens_per_eval": {
            "mean": round(_safe_avg(total_toks)),
            "min": min(total_toks) if total_toks else None,
            "max": max(total_toks) if total_toks else None,
        },
        "frontend_weighted_accuracy": {
            "mean": round(_safe_avg(fe_accs), 4),
            "min": round(min(fe_accs), 4) if fe_accs else None,
            "max": round(max(fe_accs), 4) if fe_accs else None,
        },
        "backend_accuracy": {
            "mean": round(_safe_avg(be_accs), 4),
            "min": round(min(be_accs), 4) if be_accs else None,
            "max": round(max(be_accs), 4) if be_accs else None,
        },
        "database_accuracy": {
            "mean": round(_safe_avg(db_accs), 4),
            "min": round(min(db_accs), 4) if db_accs else None,
            "max": round(max(db_accs), 4) if db_accs else None,
        },
        "appearance_score": {
            "mean": round(_safe_avg(appearances), 2),
            "min": round(min(appearances), 2) if appearances else None,
            "max": round(max(appearances), 2) if appearances else None,
        },
    }


def _safe_avg(values: list[float]) -> float:
    """Average that returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0
