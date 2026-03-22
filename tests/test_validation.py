"""Tests for the post-generation validation module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fs_agent.validation import (
    ValidationIssue,
    ValidationResult,
    validate_project,
    _check_dockerfile_build,
    _check_frontend_api_wiring,
    _check_database_config,
    _check_markdown_fences,
    _check_package_json,
    _check_required_files,
    _check_docker_compose,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Scaffold a minimal valid full-stack project."""
    backend = tmp_path / "backend"
    frontend = tmp_path / "frontend"
    backend.mkdir()
    frontend.mkdir()
    (backend / "src").mkdir(parents=True)
    (frontend / "src").mkdir(parents=True)

    # backend files
    (backend / "package.json").write_text(json.dumps({
        "name": "backend",
        "scripts": {"start": "node src/server.js", "dev": "node src/server.js"},
        "dependencies": {"express": "^4.18.0", "better-sqlite3": "^9.0.0"},
    }))
    (backend / "src" / "server.js").write_text("const app = require('./app'); app.listen(4000);")
    (backend / "src" / "app.js").write_text("const express = require('express'); module.exports = express();")
    (backend / "src" / "db.js").write_text("const Database = require('better-sqlite3');")
    (backend / "Dockerfile").write_text(
        "FROM node:20-slim\nWORKDIR /app\nCOPY . .\nRUN npm ci --production\nEXPOSE 4000\nCMD [\"node\", \"src/server.js\"]"
    )

    # backend migrations
    migrations = backend / "migrations"
    migrations.mkdir()
    (migrations / "001_init.sql").write_text("CREATE TABLE tasks (id INTEGER PRIMARY KEY);")

    # backend .env
    (backend / ".env").write_text("PORT=4000\nDB_PATH=./data/test_db.db")

    # frontend files
    (frontend / "package.json").write_text(json.dumps({
        "name": "frontend",
        "scripts": {"dev": "vite", "build": "vite build"},
        "dependencies": {"react": "^18.0.0"},
    }))
    (frontend / "index.html").write_text("<!DOCTYPE html><html></html>")
    (frontend / "src" / "main.jsx").write_text("import React from 'react';")
    (frontend / "src" / "App.jsx").write_text(
        "const API = import.meta.env.VITE_API_BASE || 'http://localhost:4000';"
    )
    (frontend / "vite.config.js").write_text(
        "import { defineConfig } from 'vite'; export default defineConfig({});"
    )
    (frontend / "Dockerfile").write_text(
        "FROM node:20 AS build\nWORKDIR /app\nCOPY . .\nRUN npm ci\nRUN npm run build\n"
        "FROM nginx:alpine\nCOPY --from=build /app/dist /usr/share/nginx/html\n"
    )

    # docker-compose.yml
    (tmp_path / "docker-compose.yml").write_text(
        "version: '3.8'\nservices:\n  backend:\n    build: ./backend\n    environment:\n"
        "      - DB_PATH=/app/data/test_db.db\n  frontend:\n    build: ./frontend\n"
    )

    return tmp_path


# ---------------------------------------------------------------------------
# ValidationResult model tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_empty_result_passes(self):
        r = ValidationResult()
        assert r.passed
        assert r.error_count == 0
        assert r.warning_count == 0

    def test_warning_only_passes(self):
        r = ValidationResult(issues=[
            ValidationIssue(component="backend", severity="warning", message="minor"),
        ])
        assert r.passed
        assert r.warning_count == 1

    def test_error_fails(self):
        r = ValidationResult(issues=[
            ValidationIssue(component="backend", severity="error", message="bad"),
        ])
        assert not r.passed
        assert r.error_count == 1

    def test_feedback_prompt_empty_when_passed(self):
        r = ValidationResult()
        assert r.feedback_prompt() == ""

    def test_feedback_prompt_lists_errors(self):
        r = ValidationResult(issues=[
            ValidationIssue(component="backend", severity="error", message="missing file", file="backend/a.js"),
            ValidationIssue(component="frontend", severity="warning", message="minor"),
        ])
        prompt = r.feedback_prompt()
        assert "missing file" in prompt
        assert "[backend]" in prompt
        assert "minor" not in prompt  # warnings excluded

    def test_errors_for_component(self):
        r = ValidationResult(issues=[
            ValidationIssue(component="backend", severity="error", message="a"),
            ValidationIssue(component="frontend", severity="error", message="b"),
            ValidationIssue(component="backend", severity="warning", message="c"),
        ])
        be = r.errors_for_component("backend")
        assert len(be) == 1
        assert be[0].message == "a"


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------


class TestCheckRequiredFiles:
    def test_valid_project(self, project_dir: Path):
        issues = _check_required_files(project_dir)
        assert not issues

    def test_missing_backend_dir(self, tmp_path: Path):
        (tmp_path / "frontend").mkdir()
        (tmp_path / "frontend" / "package.json").write_text("{}")
        (tmp_path / "frontend" / "index.html").write_text("")
        (tmp_path / "frontend" / "vite.config.js").write_text("")
        (tmp_path / "frontend" / "Dockerfile").write_text("")
        (tmp_path / "frontend" / "src").mkdir()
        (tmp_path / "frontend" / "src" / "main.jsx").write_text("")
        issues = _check_required_files(tmp_path)
        assert any(i.component == "backend" and "directory" in i.message for i in issues)

    def test_missing_server_js(self, project_dir: Path):
        (project_dir / "backend" / "src" / "server.js").unlink()
        issues = _check_required_files(project_dir)
        assert any("server.js" in i.message for i in issues)


class TestCheckPackageJson:
    def test_valid(self, project_dir: Path):
        issues = _check_package_json(project_dir, "backend")
        assert not issues

    def test_missing(self, tmp_path: Path):
        (tmp_path / "backend").mkdir()
        issues = _check_package_json(tmp_path, "backend")
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_invalid_json(self, project_dir: Path):
        (project_dir / "backend" / "package.json").write_text("{invalid")
        issues = _check_package_json(project_dir, "backend")
        assert any("Invalid" in i.message for i in issues)

    def test_missing_scripts(self, project_dir: Path):
        (project_dir / "backend" / "package.json").write_text(json.dumps({"name": "x"}))
        issues = _check_package_json(project_dir, "backend")
        assert any("scripts" in i.message for i in issues)


class TestCheckMarkdownFences:
    def test_clean_files(self, project_dir: Path):
        issues = _check_markdown_fences(project_dir, "backend")
        assert not issues

    def test_detects_fence(self, project_dir: Path):
        (project_dir / "backend" / "src" / "server.js").write_text(
            "```javascript\nconst x = 1;\n```"
        )
        issues = _check_markdown_fences(project_dir, "backend")
        assert len(issues) >= 1
        assert "markdown fence" in issues[0].message.lower()


class TestCheckDockerfileBuild:
    def test_valid_dockerfile(self, project_dir: Path):
        issues = _check_dockerfile_build(project_dir, "frontend")
        assert not issues

    def test_prod_before_build(self, tmp_path: Path):
        comp = tmp_path / "bad"
        comp.mkdir()
        (comp / "Dockerfile").write_text(
            "FROM node:20\nCOPY . .\nRUN npm ci --production\nRUN npm run build\n"
        )
        issues = _check_dockerfile_build(tmp_path, "bad")
        assert len(issues) == 1
        assert "production" in issues[0].message.lower()

    def test_no_dockerfile(self, tmp_path: Path):
        (tmp_path / "comp").mkdir()
        issues = _check_dockerfile_build(tmp_path, "comp")
        assert not issues


class TestCheckFrontendApiWiring:
    def test_valid_wiring(self, project_dir: Path):
        issues = _check_frontend_api_wiring(project_dir)
        assert not issues

    def test_wrong_port(self, project_dir: Path):
        (project_dir / "frontend" / "src" / "App.jsx").write_text(
            "fetch('http://localhost:3001/api/tasks')"
        )
        issues = _check_frontend_api_wiring(project_dir)
        assert any("port" in i.message.lower() for i in issues)


class TestCheckDatabaseConfig:
    def test_valid_config(self, project_dir: Path):
        issues = _check_database_config(project_dir)
        assert not issues

    def test_missing_db_js_with_migrations(self, project_dir: Path):
        (project_dir / "backend" / "src" / "db.js").unlink()
        issues = _check_database_config(project_dir)
        assert any("db.js" in i.message for i in issues)

    def test_missing_sqlite_dep(self, project_dir: Path):
        pkg = json.loads((project_dir / "backend" / "package.json").read_text())
        del pkg["dependencies"]["better-sqlite3"]
        (project_dir / "backend" / "package.json").write_text(json.dumps(pkg))
        issues = _check_database_config(project_dir)
        assert any("better-sqlite3" in i.message for i in issues)

    def test_db_path_mismatch(self, project_dir: Path):
        (project_dir / "backend" / ".env").write_text("DB_PATH=./data/wrong.db")
        issues = _check_database_config(project_dir)
        assert any("mismatch" in i.message.lower() for i in issues)


class TestCheckDockerCompose:
    def test_valid(self, project_dir: Path):
        issues = _check_docker_compose(project_dir)
        assert not issues

    def test_missing_compose(self, tmp_path: Path):
        issues = _check_docker_compose(tmp_path)
        assert any("docker-compose" in i.message.lower() for i in issues)

    def test_missing_backend_service(self, project_dir: Path):
        (project_dir / "docker-compose.yml").write_text("services:\n  frontend:\n    build: .")
        issues = _check_docker_compose(project_dir)
        assert any("backend" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Full validate_project integration test
# ---------------------------------------------------------------------------


class TestValidateProject:
    def test_valid_project_passes(self, project_dir: Path):
        result = validate_project(project_dir)
        assert result.passed, f"Expected pass, got: {[str(i.message) for i in result.issues]}"

    def test_nonexistent_dir(self, tmp_path: Path):
        result = validate_project(tmp_path / "nope")
        assert not result.passed
        assert result.error_count >= 1

    def test_multiple_issues(self, project_dir: Path):
        # Break backend Dockerfile (prod before build)
        (project_dir / "backend" / "Dockerfile").write_text(
            "FROM node:20\nCOPY . .\nRUN npm ci --production\nRUN npm run build\n"
        )
        # Add markdown fence in frontend
        (project_dir / "frontend" / "src" / "main.jsx").write_text("```jsx\nimport React;\n```")
        result = validate_project(project_dir)
        assert not result.passed
        assert result.error_count >= 2


# ---------------------------------------------------------------------------
# Validation loop tests (using _helpers.run_validation_loop)
# ---------------------------------------------------------------------------


class TestValidationLoop:
    def test_skips_when_retries_zero(self, project_dir: Path):
        """With max_retries=0, loop returns reports unchanged."""
        from unittest.mock import MagicMock, patch

        from fs_agent.orchestration._helpers import run_validation_loop

        context = MagicMock()
        context.settings.max_validation_retries = 0
        context.projects_dir = project_dir
        registry = MagicMock()
        metrics = MagicMock()
        reports = [MagicMock()]

        result = run_validation_loop(context, registry, reports, metrics, max_retries=0)
        assert result is reports

    def test_passes_immediately(self, project_dir: Path):
        """When project is valid, loop returns without retries."""
        from unittest.mock import MagicMock

        from fs_agent.orchestration._helpers import run_validation_loop

        context = MagicMock()
        context.settings.max_validation_retries = 3
        context.projects_dir = project_dir
        registry = MagicMock()
        metrics = MagicMock()
        reports = [MagicMock()]

        result = run_validation_loop(context, registry, reports, metrics)
        assert len(result) == 1  # no retries added
