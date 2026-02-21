"""Command-line entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import typer
from rich.console import Console

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


def _print_summary(reports: Iterable[AgentReport]) -> None:
    for report in reports:
        console.rule(f"[bold cyan]{report.role.upper()} stage")
        console.print(report.summary)
        if report.artifacts:
            console.print("Artifacts:")
            for key in report.artifacts.keys():
                console.print(f"  - {key}")
    console.rule("[bold green]Pipeline complete")