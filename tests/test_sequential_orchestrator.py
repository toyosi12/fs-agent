"""Smoke tests for the sequential orchestrator."""

from __future__ import annotations

from pathlib import Path

from fs_agent.orchestrator import run_orchestration


def test_sequential_orchestrator_runs_in_order(tmp_path: Path) -> None:
    spec_path = Path(__file__).parents[1] / "examples" / "specs" / "todo_app.yaml"
    reports = list(run_orchestration(spec_path, artifact_dir=tmp_path))
    roles = [report.role for report in reports]
    assert roles == ["backend", "frontend", "infra"]
    assert reports[-1].artifacts["infra_pipeline"]["artifacts"] == [
        "backend_blueprint",
        "frontend_blueprint",
    ]

    saved = sorted(p.name for p in tmp_path.iterdir())
    assert saved == [
        "backend_backend_blueprint.json",
        "backend_plan.md",
        "frontend_frontend_blueprint.json",
        "frontend_plan.md",
        "infra_infra_pipeline.json",
        "infra_plan.md",
    ]

    for report in reports:
        assert report.metadata["artifact_files"], "artifact paths should be recorded"
        assert report.metadata["attachment_files"], "attachment paths should be recorded"
