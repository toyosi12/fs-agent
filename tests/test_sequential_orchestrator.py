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
        "backend_source",
        "frontend_blueprint",
        "frontend_source",
    ]

    saved = sorted(p.name for p in tmp_path.iterdir())
    expected = {
        "backend_backend_blueprint.json",
        "backend_backend_source.json",
        "backend_plan.md",
        "backend_service.ts",
        "frontend_frontend_blueprint.json",
        "frontend_frontend_source.json",
        "frontend_app.tsx",
        "frontend_plan.md",
        "infra_infra_pipeline.json",
        "infra_infra_runbook.json",
        "infra_plan.md",
        "infra_runbook.md",
    }
    assert saved == sorted(expected)

    for report in reports:
        assert report.metadata["artifact_files"], "artifact paths should be recorded"
        assert report.metadata["attachment_files"], "attachment paths should be recorded"
