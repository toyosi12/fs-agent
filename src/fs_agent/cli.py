"""Command-line entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import typer
from rich.console import Console
from rich.table import Table

from .context import AgentReport
from .orchestrator import run_orchestration

app = typer.Typer(help="Multi-agent orchestrator for full-stack scaffolding.")
console = Console()


@app.command()
def run(
    request: str = typer.Argument(..., help="Natural language description of the desired app"),
    artifact_dir: Path | None = typer.Option(None, help="Directory to store generated artifacts"),
    dry_run: bool = typer.Option(False, help="Skip shell-side effects and focus on planning"),
    pattern: str = typer.Option("sequential", help="Orchestration pattern: sequential or centralized"),
) -> None:
    """Execute the orchestrator against a project spec."""

    reports = list(
        run_orchestration(
            request,
            artifact_dir=artifact_dir,
            dry_run=dry_run,
            orchestration_pattern=pattern,
        )
    )
    _print_summary(reports)


@app.command()
def benchmark(
    dataset: Path = typer.Argument(
        ..., help="Path to the tasks JSON file (e.g. dataset/tasks_with_difficulty.json)"
    ),
    patterns: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of patterns to benchmark (default: all six)",
    ),
    task_ids: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of task IDs to run (default: all)",
    ),
    max_tasks: Optional[int] = typer.Option(
        None,
        help="Maximum number of tasks to process (useful for quick tests)",
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts") / "benchmark",
        help="Root directory for benchmark outputs",
    ),
) -> None:
    """Run the benchmark suite: every pattern against each task in the dataset."""

    from .benchmark import run_benchmark

    pat_list = [p.strip() for p in patterns.split(",")] if patterns else None
    id_list = [i.strip() for i in task_ids.split(",")] if task_ids else None

    results = run_benchmark(
        dataset_path=dataset,
        patterns=pat_list,
        task_ids=id_list,
        max_tasks=max_tasks,
        artifact_root=artifact_root,
    )

    # Print a rich summary table
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Pattern", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Wall Clock (s)", justify="right")
    table.add_column("LLM Calls", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Overhead (s)", justify="right")
    table.add_column("Agents", justify="right")

    for m in results:
        status = "[green]OK[/green]" if m.success else f"[red]FAIL[/red]"
        table.add_row(
            m.task_id,
            m.pattern,
            status,
            f"{m.wall_clock_seconds:.2f}",
            str(m.llm_call_count),
            str(m.total_tokens),
            f"{m.orchestration_overhead_seconds:.2f}",
            str(m.agent_count),
        )

    console.print(table)
    console.print(
        f"\n[bold green]Results written to:[/bold green] {artifact_root / 'results'}"
    )


@app.command()
def judge(
    dataset: Path = typer.Argument(
        ..., help="Path to the tasks JSON file (e.g. dataset/tasks_with_difficulty.json)"
    ),
    artifact_root: Path = typer.Option(
        Path("artifacts") / "benchmark",
        help="Root directory containing benchmark outputs to evaluate",
    ),
    patterns: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of patterns to evaluate (default: all found)",
    ),
    task_ids: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of task IDs to evaluate (default: all)",
    ),
    max_tasks: Optional[int] = typer.Option(
        None,
        help="Maximum number of tasks to evaluate",
    ),
    mode: str = typer.Option(
        "static",
        help="Evaluation mode: 'static' (code review only) or 'runtime' (boot in Docker, send real HTTP requests)",
    ),
) -> None:
    """Score generated applications using the LLM-as-a-judge (GPT-4o).

    Reads generated code from benchmark artifacts and evaluates:
    - Frontend functional tests (YES / PARTIAL / NO)
    - Backend API tests (YES / NO)
    - Database schema tests (YES / NO)
    - Appearance score (1-5 on four criteria)

    Two modes:
    - **static** (default): reviews generated source code only.
    - **runtime**: boots projects in Docker, sends real HTTP requests,
      checks the SQLite database, then evaluates responses.

    Requires FS_AGENT_JUDGE_API_KEY to be set (OpenAI API key).
    Runtime mode also requires Docker to be installed and running.
    Results are written to <artifact-root>/results/judge_results.json
    and judge_summary.json, tracked by task ID and difficulty level.
    """

    from .judge import run_judge

    pat_list = [p.strip() for p in patterns.split(",")] if patterns else None
    id_list = [i.strip() for i in task_ids.split(",")] if task_ids else None

    results = run_judge(
        dataset_path=dataset,
        artifact_root=artifact_root,
        patterns=pat_list,
        task_ids=id_list,
        max_tasks=max_tasks,
        mode=mode,
    )

    # Print a rich summary table
    table = Table(title="Judge Results", show_lines=True)
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Pattern", style="magenta")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Frontend", justify="right")
    table.add_column("Backend", justify="right")
    table.add_column("Database", justify="right")
    table.add_column("Appearance", justify="right")
    table.add_column("Error", style="red")

    for r in results:
        fe_str = f"{r.frontend_weighted_accuracy:.2f}" if r.frontend_total > 0 else "-"
        be_str = f"{r.backend_accuracy:.2f}" if r.backend_total > 0 else "-"
        db_str = f"{r.database_accuracy:.2f}" if r.database_total > 0 else "-"
        ap_str = f"{r.appearance.overall:.1f}" if r.appearance else "-"
        err_str = r.error[:40] if r.error else ""
        table.add_row(
            r.task_id,
            r.pattern,
            r.difficulty,
            fe_str,
            be_str,
            db_str,
            ap_str,
            err_str,
        )

    console.print(table)
    console.print(
        f"\n[bold green]Judge results written to:[/bold green] {artifact_root / 'results'}"
    )


def _print_summary(reports: Iterable[AgentReport]) -> None:
    for report in reports:
        console.rule(f"[bold cyan]{report.role.upper()} stage")
        console.print(report.summary)
        if report.artifacts:
            console.print("Artifacts:")
            for key in report.artifacts.keys():
                console.print(f"  - {key}")
    console.rule("[bold green]Pipeline complete")