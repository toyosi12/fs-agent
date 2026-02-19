"""Smoke tests for the sequential orchestrator."""

from __future__ import annotations

from pathlib import Path

from fs_agent.orchestrator import run_orchestration


def test_sequential_orchestrator_runs_in_order(tmp_path: Path) -> None:
    request = "Build a shared task tracker for teams"
    reports = list(run_orchestration(request, artifact_dir=tmp_path))
    roles = [report.role for report in reports]
    assert roles == ["architect", "backend", "frontend", "infra"]

    # --- Architect ---
    architect_spec = reports[0].artifacts["architect_spec"]
    assert architect_spec["metadata"]["summary"] == request
    assert "endpoints" in architect_spec["backend"]
    assert "routes" in architect_spec["frontend"]

    # --- Backend ---
    backend_plan = reports[1].artifacts["backend_mcp_plan"]
    assert backend_plan["tool"] == "mcp.fs"
    assert backend_plan["project_root"].endswith("-backend")
    backend_files = reports[1].artifacts["backend_project_files"]
    assert backend_files, "backend should record created files"

    # --- Frontend ---
    frontend_plan = reports[2].artifacts["frontend_mcp_plan"]
    assert frontend_plan["tool"] == "mcp.fs"
    assert frontend_plan["project_root"].endswith("-frontend")
    frontend_files = reports[2].artifacts["frontend_project_files"]
    assert frontend_files, "frontend should record created files"

    # --- Infra should see all prior artifacts ---
    infra_artifacts = reports[-1].artifacts["infra_pipeline"]["artifacts"]
    assert "architect_spec" in infra_artifacts
    assert "backend_blueprint" in infra_artifacts
    assert "frontend_blueprint" in infra_artifacts

    # --- Saved files ---
    saved = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    # Check key files exist (the exact set depends on agent output)
    assert "architect_architect_spec.json" in saved
    assert "architect_architect_summary.json" in saved
    assert "architect_spec.json" in saved

    # --- Projects directory ---
    projects_dir = tmp_path / "projects"
    assert projects_dir.exists()
    backend_root = projects_dir / backend_plan["project_root"]
    frontend_root = projects_dir / frontend_plan["project_root"]
    assert (backend_root / "package.json").exists()
    assert (frontend_root / "package.json").exists()

    for recorded in backend_files + frontend_files:
        assert Path(recorded).exists()

    for report in reports:
        assert report.metadata["artifact_files"], "artifact paths should be recorded"
        assert report.metadata["attachment_files"], "attachment paths should be recorded"
