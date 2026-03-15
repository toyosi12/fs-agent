"""Tests for the runtime evaluation pipeline.

These tests use stubs/fakes to avoid needing Docker or a real LLM.
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from fs_agent.judge.executor import (
    ProjectInstance,
    ensure_dockerfiles,
    find_project_dir,
)
from fs_agent.judge.models import BinaryVerdict, FrontendVerdict
from fs_agent.judge.runtime import (
    _execute_requests,
    _find_sqlite_file,
    _get_schema,
    _get_table_names,
    _parse_json,
    evaluate_application_runtime,
    run_backend_tests,
    run_database_tests,
    run_frontend_tests,
)


# ---------------------------------------------------------------------------
# Fake LLM for deterministic testing
# ---------------------------------------------------------------------------


class FakeRuntimeLLM:
    """Returns canned JSON responses based on the system prompt content."""

    model = "fake-runtime"

    def generate(self, user: str, *, system: str = "", temperature: float = 0.7) -> str:
        # Backend request spec generation
        if "HTTP request specifications" in system:
            return json.dumps([
                {
                    "test_index": 0,
                    "method": "GET",
                    "path": "/api/stocks/search",
                    "headers": {"Content-Type": "application/json"},
                    "body": None,
                    "query_params": {"q": "AAPL"},
                },
                {
                    "test_index": 1,
                    "method": "POST",
                    "path": "/api/reports/generate",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"symbols": ["INVALID"]},
                    "query_params": None,
                },
            ])

        # Backend response evaluation
        if "evaluating API test results" in system:
            return json.dumps([
                {"test_index": 0, "verdict": "YES", "reasoning": "Returned stock data"},
                {"test_index": 1, "verdict": "NO", "reasoning": "Server error"},
            ])

        # Database evaluation
        if "SQLite database" in system:
            return json.dumps([
                {"data_structure": "stock information", "verdict": "YES", "reasoning": "stocks table exists"},
                {"data_structure": "generated reports", "verdict": "YES", "reasoning": "reports table exists"},
            ])

        # Frontend evaluation
        if "frontend QA evaluator" in system or "UI test case" in system:
            return json.dumps([
                {"test_index": 0, "verdict": "YES", "reasoning": "Search works"},
                {"test_index": 1, "verdict": "PARTIAL", "reasoning": "Partially works"},
            ])

        # Appearance evaluation
        if "UI/UX designer" in system:
            return json.dumps({
                "layout": 4,
                "color": 3,
                "typography": 4,
                "component_polish": 3,
                "reasoning": "Clean layout, decent colors",
            })

        return '{"verdict": "NO", "reasoning": "unknown prompt"}'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm():
    return FakeRuntimeLLM()


@pytest.fixture
def tmp_project(tmp_path: Path):
    """Create a minimal project directory structure."""
    project_dir = tmp_path / "projects" / "test-app"
    backend_dir = project_dir / "backend"
    frontend_dir = project_dir / "frontend"

    backend_dir.mkdir(parents=True)
    frontend_dir.mkdir(parents=True)

    # Write a package.json so Dockerfiles make sense
    (backend_dir / "package.json").write_text('{"name":"test"}')
    (frontend_dir / "package.json").write_text('{"name":"test-fe"}')

    # Write docker-compose.yml
    (project_dir / "docker-compose.yml").write_text(
        "version: '3.8'\nservices:\n  backend:\n    build: ./backend\n"
    )

    return project_dir


@pytest.fixture
def sqlite_db(tmp_path: Path):
    """Create a test SQLite database with tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE stocks (id INTEGER PRIMARY KEY, symbol TEXT, name TEXT, price REAL)"
    )
    conn.execute(
        "CREATE TABLE reports (id INTEGER PRIMARY KEY, stock_symbol TEXT, format TEXT)"
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# executor tests
# ---------------------------------------------------------------------------


class TestEnsureDockerfiles:
    def test_generates_missing_backend_dockerfile(self, tmp_project: Path):
        assert not (tmp_project / "backend" / "Dockerfile").exists()
        ensure_dockerfiles(tmp_project)
        assert (tmp_project / "backend" / "Dockerfile").exists()
        assert (tmp_project / "backend" / ".dockerignore").exists()
        content = (tmp_project / "backend" / "Dockerfile").read_text()
        assert "node" in content.lower()

    def test_generates_missing_frontend_dockerfile(self, tmp_project: Path):
        assert not (tmp_project / "frontend" / "Dockerfile").exists()
        ensure_dockerfiles(tmp_project)
        assert (tmp_project / "frontend" / "Dockerfile").exists()
        content = (tmp_project / "frontend" / "Dockerfile").read_text()
        assert "nginx" in content.lower()

    def test_does_not_overwrite_existing(self, tmp_project: Path):
        df = tmp_project / "backend" / "Dockerfile"
        df.write_text("CUSTOM")
        ensure_dockerfiles(tmp_project)
        assert df.read_text() == "CUSTOM"


class TestFindProjectDir:
    def test_finds_project(self, tmp_path: Path):
        p = tmp_path / "000001" / "sequential" / "projects" / "myapp"
        p.mkdir(parents=True)
        result = find_project_dir(tmp_path, "000001", "sequential")
        assert result == p

    def test_returns_none_when_missing(self, tmp_path: Path):
        result = find_project_dir(tmp_path, "000001", "sequential")
        assert result is None


# ---------------------------------------------------------------------------
# runtime - JSON parsing
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_array(self):
        result = _parse_json('[{"v":"YES"}]')
        assert result == [{"v": "YES"}]


# ---------------------------------------------------------------------------
# runtime - SQLite helpers
# ---------------------------------------------------------------------------


class TestSqliteHelpers:
    def test_find_sqlite_file_in_data(self, tmp_path: Path):
        backend = tmp_path / "backend" / "data"
        backend.mkdir(parents=True)
        db = backend / "test.db"
        db.write_bytes(b"")
        result = _find_sqlite_file(tmp_path)
        assert result == db

    def test_find_sqlite_file_direct(self, tmp_path: Path):
        backend = tmp_path / "backend"
        backend.mkdir(parents=True)
        db = backend / "database.sqlite"
        db.write_bytes(b"")
        result = _find_sqlite_file(tmp_path)
        assert result == db

    def test_find_sqlite_file_not_found(self, tmp_path: Path):
        (tmp_path / "backend").mkdir(parents=True)
        result = _find_sqlite_file(tmp_path)
        assert result is None

    def test_get_schema(self, sqlite_db: Path):
        schema = _get_schema(sqlite_db)
        assert "stocks" in schema
        assert "reports" in schema

    def test_get_table_names(self, sqlite_db: Path):
        names = _get_table_names(sqlite_db)
        assert set(names) == {"stocks", "reports"}


# ---------------------------------------------------------------------------
# runtime - HTTP execution (with mock server)
# ---------------------------------------------------------------------------


class TestExecuteRequests:
    def test_handles_connection_error(self):
        """Requests to a non-existent server should record errors."""
        specs = [{"test_index": 0, "method": "GET", "path": "/api/test"}]
        results = _execute_requests("http://localhost:19999", specs, timeout=1.0)
        assert len(results) == 1
        assert results[0]["error"] is not None
        assert results[0]["status_code"] is None


# ---------------------------------------------------------------------------
# runtime - backend test integration (with fake LLM)
# ---------------------------------------------------------------------------


class TestRunBackendTests:
    def test_empty_test_cases(self, fake_llm):
        scores = run_backend_tests(fake_llm, "http://localhost:4000", "", [])
        assert scores == []

    def test_spec_generation_failure(self, fake_llm):
        """If LLM fails to generate specs, all tests should score NO."""

        class FailLLM:
            model = "fail"
            def generate(self, *a, **kw):
                raise RuntimeError("LLM down")

        cases = [
            {"instruction": "Test search", "expected_result": "Returns data"},
        ]
        scores = run_backend_tests(FailLLM(), "http://localhost:4000", "code", cases)
        assert len(scores) == 1
        assert scores[0].verdict == BinaryVerdict.NO
        assert "Could not generate request spec" in scores[0].reasoning


# ---------------------------------------------------------------------------
# runtime - database tests
# ---------------------------------------------------------------------------


class TestRunDatabaseTests:
    def test_with_sqlite_file(self, fake_llm, sqlite_db: Path):
        project_dir = sqlite_db.parent
        # _find_sqlite_file expects project_dir/backend/...
        backend = project_dir / "backend"
        backend.mkdir(exist_ok=True)
        # Move the db to backend/
        import shutil
        dest = backend / "database.sqlite"
        shutil.copy2(sqlite_db, dest)

        scores = run_database_tests(
            fake_llm, project_dir, "Build a stock app",
            ["stock information", "generated reports"], "",
        )
        assert len(scores) == 2
        assert scores[0].verdict == BinaryVerdict.YES
        assert "database.sqlite" in scores[0].reasoning

    def test_no_db_no_migrations(self, fake_llm, tmp_path: Path):
        (tmp_path / "backend").mkdir()
        scores = run_database_tests(
            fake_llm, tmp_path, "task", ["data"], "",
        )
        assert len(scores) == 1
        assert scores[0].verdict == BinaryVerdict.NO

    def test_falls_back_to_migrations(self, fake_llm, tmp_path: Path):
        (tmp_path / "backend").mkdir()
        scores = run_database_tests(
            fake_llm, tmp_path, "task",
            ["stock information", "generated reports"],
            "CREATE TABLE stocks (...);",
        )
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# runtime - frontend tests
# ---------------------------------------------------------------------------


class TestRunFrontendTests:
    def test_unhealthy_frontend_returns_no(self, fake_llm):
        cases = [
            {"task": "Click search", "expected_result": "Shows results"},
        ]
        scores = run_frontend_tests(
            fake_llm, "http://localhost:3000", False, "task", "", cases,
        )
        assert len(scores) == 1
        assert scores[0].verdict == FrontendVerdict.NO
        assert "not returning http 200" in scores[0].reasoning.lower()

    def test_empty_test_cases(self, fake_llm):
        assert run_frontend_tests(fake_llm, "http://x", True, "", "", []) == []


# ---------------------------------------------------------------------------
# runtime - full evaluation
# ---------------------------------------------------------------------------


class TestEvaluateApplicationRuntime:
    def test_full_evaluation_with_no_services(self, fake_llm, tmp_path: Path):
        """When both services are unhealthy, backend/frontend should all be NO."""
        (tmp_path / "backend").mkdir()
        instance = ProjectInstance(
            project_dir=tmp_path,
            backend_healthy=False,
            frontend_healthy=False,
        )
        result = evaluate_application_runtime(
            llm=fake_llm,
            instance=instance,
            task_id="t1",
            pattern="sequential",
            difficulty="easy",
            task_instruction="Build an app",
            frontend_code="const App = () => <div />",
            backend_code="app.get('/api', ...)",
            migration_sql="CREATE TABLE t (...);",
            ui_test_cases=[{"task": "Click", "expected_result": "Works"}],
            backend_test_cases=[{"instruction": "Test API", "expected_result": "200"}],
            data_structures=["users"],
        )
        assert result.task_id == "t1"
        assert result.pattern == "sequential"
        assert len(result.backend_tests) == 1
        assert result.backend_tests[0].verdict == BinaryVerdict.NO
        assert len(result.frontend_tests) == 1
        assert result.frontend_tests[0].verdict == FrontendVerdict.NO
        # Database should still work (LLM evaluates migration SQL)
        assert len(result.database_tests) == 1
        # Appearance should still work (static code analysis)
        assert result.appearance is not None
