"""Runtime evaluation engine — tests generated projects by executing them.

Unlike the static scorer (``scoring.py``) which only reads source code,
this module boots the project via Docker, sends real HTTP requests to the
backend, checks the SQLite schema, and optionally takes frontend
screenshots for visual evaluation.

Design decisions
----------------
* **Batched LLM calls** — to stay within rate limits, backend test HTTP
  request specs are generated in one LLM call, and responses are evaluated
  in another.  This keeps the total LLM call count to ~4 per (task × pattern)
  regardless of how many individual test cases exist.
* **Programmatic database checks** — the SQLite schema is queried directly ;
  the LLM maps human-readable data-structure names to table names only once.
* **Frontend** — if the frontend container is healthy we record a pass
  for the "renders at all" check.  Deeper functional testing would require
  Playwright (not yet implemented).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from ..llm import BaseLLMClient
from ..logger import get_logger
from .executor import ProjectInstance
from .models import (
    AppearanceScore,
    BackendTestScore,
    BinaryVerdict,
    DatabaseTestScore,
    FrontendTestScore,
    FrontendVerdict,
    JudgeResult,
)

logger = get_logger(__name__)

_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_RATE_DELAY = 5  # seconds between LLM calls


# ---------------------------------------------------------------------------
# JSON parsing helper (shared with static scorer)
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> Any:
    """Extract and parse JSON from an LLM response, stripping fences."""
    text = raw.strip()
    match = _CODE_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ===================================================================
# 1.  BACKEND RUNTIME TESTS
# ===================================================================

def _generate_request_specs(
    llm: BaseLLMClient,
    backend_code: str,
    test_cases: list[dict],
) -> list[dict]:
    """Ask the LLM to produce concrete HTTP request specs for all test cases.

    Returns a list of dicts, one per test case, each containing:
    ``method``, ``path``, ``headers``, ``body``, ``query_params``.
    """
    if not test_cases:
        return []

    cases_text = "\n".join(
        f"{i+1}. Instruction: {tc['instruction']}\n"
        f"   Expected: {tc['expected_result']}"
        for i, tc in enumerate(test_cases)
    )

    system = (
        "You are a QA engineer. Given REST API source code and test-case "
        "descriptions, produce a JSON array of concrete HTTP request specifications. "
        "Output ONLY the JSON array — no prose, no fences.\n\n"
        "Each element:\n"
        '{"test_index":<0-based>,"method":"GET|POST|PUT|DELETE",'
        '"path":"/api/...","headers":{"Content-Type":"application/json"},'
        '"body":null,"query_params":null}\n\n'
        "Rules:\n"
        "- Use paths that match routes in the source code.\n"
        "- Generate realistic test data.\n"
        "- For invalid-input tests, use clearly invalid data.\n"
        "- query_params is a dict of URL query parameters (for GET).\n"
        "- body is a dict for POST/PUT JSON bodies (or null)."
    )

    user = (
        f"## API Source Code\n```\n{_truncate(backend_code, 10000)}\n```\n\n"
        f"## Test Cases\n{cases_text}\n\n"
        "Produce the JSON array."
    )

    raw = llm.generate(user, system=system, temperature=0.1)
    specs = _parse_json(raw)
    if not isinstance(specs, list):
        return []
    return specs


def _execute_requests(
    base_url: str,
    specs: list[dict],
    timeout: float = 10.0,
) -> list[dict]:
    """Execute each HTTP request spec against the running backend.

    Returns a list of result dicts: ``status_code``, ``body`` (str),
    ``error`` (str or None).
    """
    results: list[dict] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for spec in specs:
            method = spec.get("method", "GET").upper()
            path = spec.get("path", "/")
            headers = spec.get("headers") or {}
            body = spec.get("body")
            params = spec.get("query_params")

            try:
                resp = client.request(
                    method,
                    path,
                    headers=headers,
                    json=body if body is not None else None,
                    params=params,
                )
                results.append({
                    "test_index": spec.get("test_index", len(results)),
                    "status_code": resp.status_code,
                    "body": resp.text[:4000],
                    "error": None,
                })
            except httpx.HTTPError as exc:
                results.append({
                    "test_index": spec.get("test_index", len(results)),
                    "status_code": None,
                    "body": "",
                    "error": str(exc),
                })
    return results


def _evaluate_responses(
    llm: BaseLLMClient,
    test_cases: list[dict],
    specs: list[dict],
    responses: list[dict],
) -> list[dict]:
    """Ask the LLM to judge each HTTP response against the expected result.

    Returns a list of dicts with ``test_index``, ``verdict``, ``reasoning``.
    """
    entries: list[str] = []
    for i, tc in enumerate(test_cases):
        spec = specs[i] if i < len(specs) else {}
        resp = responses[i] if i < len(responses) else {}
        request_line = f"{spec.get('method','?')} {spec.get('path','?')}"
        if spec.get("body"):
            request_line += f"\nBody: {json.dumps(spec['body'])[:500]}"
        if spec.get("query_params"):
            request_line += f"\nParams: {json.dumps(spec['query_params'])}"

        status = resp.get("status_code", "N/A")
        body = resp.get("body", "")
        error = resp.get("error", "")
        resp_text = f"Status: {status}\n"
        if error:
            resp_text += f"Error: {error}\n"
        else:
            resp_text += f"Body: {_truncate(body, 1500)}\n"

        entries.append(
            f"### Test {i+1}\n"
            f"**Instruction:** {tc['instruction']}\n"
            f"**Expected:** {tc['expected_result']}\n"
            f"**Request:** {request_line}\n"
            f"**Response:**\n{resp_text}"
        )

    system = (
        "You are a QA engineer evaluating API test results. For each test, "
        "determine if the actual HTTP response satisfies the expected result. "
        'Output ONLY a JSON array: [{"test_index":0,"verdict":"YES"|"NO","reasoning":"..."},...]'
    )
    user = "\n".join(entries) + "\n\nEvaluate each test."

    raw = llm.generate(user, system=system, temperature=0.1)
    verdicts = _parse_json(raw)
    if not isinstance(verdicts, list):
        return []
    return verdicts


def run_backend_tests(
    llm: BaseLLMClient,
    backend_url: str,
    backend_code: str,
    test_cases: list[dict],
) -> list[BackendTestScore]:
    """Full runtime backend test flow: generate specs → execute → evaluate."""
    if not test_cases:
        return []

    # Step 1: LLM generates request specs
    print(f"[runtime]   Generating {len(test_cases)} backend HTTP request specs ...")
    try:
        specs = _generate_request_specs(llm, backend_code, test_cases)
    except Exception as exc:
        logger.warning("Failed to generate request specs: %s", exc)
        return [
            BackendTestScore(
                test_case=tc["instruction"],
                expected_result=tc["expected_result"],
                verdict=BinaryVerdict.NO,
                reasoning=f"Could not generate request spec: {exc}",
            )
            for tc in test_cases
        ]

    # Pad specs to match test_cases length
    while len(specs) < len(test_cases):
        specs.append({"test_index": len(specs), "method": "GET", "path": "/"})

    # Step 2: Execute HTTP requests
    print(f"[runtime]   Executing {len(specs)} HTTP requests against {backend_url} ...")
    responses = _execute_requests(backend_url, specs)

    # Step 3: LLM evaluates responses
    time.sleep(_RATE_DELAY)
    print("[runtime]   Evaluating responses ...")
    try:
        verdicts = _evaluate_responses(llm, test_cases, specs, responses)
    except Exception as exc:
        logger.warning("Failed to evaluate responses: %s", exc)
        verdicts = []

    # Build BackendTestScore list
    scores: list[BackendTestScore] = []
    for i, tc in enumerate(test_cases):
        v = verdicts[i] if i < len(verdicts) else {}
        verdict_str = v.get("verdict", "NO").upper() if isinstance(v, dict) else "NO"
        verdict = BinaryVerdict(verdict_str) if verdict_str in ("YES", "NO") else BinaryVerdict.NO
        reasoning = v.get("reasoning", "") if isinstance(v, dict) else ""

        # Enrich reasoning with actual HTTP details
        resp = responses[i] if i < len(responses) else {}
        if resp.get("status_code") is not None:
            reasoning = f"[HTTP {resp['status_code']}] {reasoning}"
        elif resp.get("error"):
            reasoning = f"[Request failed: {resp['error']}] {reasoning}"
            verdict = BinaryVerdict.NO

        scores.append(BackendTestScore(
            test_case=tc["instruction"],
            expected_result=tc["expected_result"],
            verdict=verdict,
            reasoning=reasoning,
        ))
    return scores


# ===================================================================
# 2.  DATABASE RUNTIME TESTS
# ===================================================================

def _find_sqlite_file(project_dir: Path) -> Path | None:
    """Locate the SQLite database file in the project."""
    backend_dir = project_dir / "backend"
    # Common locations
    candidates = [
        backend_dir / "data" / "*.db",
        backend_dir / "data" / "*.sqlite",
        backend_dir / "*.db",
        backend_dir / "*.sqlite",
        backend_dir / "database.sqlite",
    ]
    for pattern in candidates:
        if "*" in str(pattern):
            matches = list(pattern.parent.glob(pattern.name))
            if matches:
                return matches[0]
        elif pattern.exists():
            return pattern
    return None


def _get_schema(db_path: Path) -> str:
    """Read the full schema from a SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        )
        rows = cursor.fetchall()
        return "\n\n".join(f"-- Table: {name}\n{sql}" for name, sql in rows)
    finally:
        conn.close()


def _get_table_names(db_path: Path) -> list[str]:
    """Return list of user table names from a SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def run_database_tests(
    llm: BaseLLMClient,
    project_dir: Path,
    task_instruction: str,
    data_structures: list[str],
    migration_sql: str,
) -> list[DatabaseTestScore]:
    """Check database schema by opening the actual SQLite file.

    Falls back to migration SQL analysis if no database file is found.
    """
    if not data_structures:
        return []

    db_path = _find_sqlite_file(project_dir)

    if db_path and db_path.exists():
        schema = _get_schema(db_path)
        table_names = _get_table_names(db_path)
        print(f"[runtime]   Found SQLite database: {db_path.name} ({len(table_names)} tables)")
        source_label = "Actual Database Schema"
    elif migration_sql.strip():
        schema = migration_sql
        table_names = []
        source_label = "Migration SQL (database file not found)"
        print("[runtime]   No SQLite file found; falling back to migration SQL")
    else:
        print("[runtime]   No database or migrations found")
        return [
            DatabaseTestScore(
                data_structure=ds,
                verdict=BinaryVerdict.NO,
                reasoning="No database file or migration SQL found.",
            )
            for ds in data_structures
        ]

    # Single LLM call to evaluate all data structures
    ds_list = "\n".join(f"- {ds}" for ds in data_structures)

    system = (
        "You are a database engineer evaluating whether a SQLite database "
        "correctly models required data structures. "
        'Output ONLY a JSON array: [{"data_structure":"...","verdict":"YES"|"NO","reasoning":"..."},...]'
    )
    user = (
        f"## Required Data Structures\n{ds_list}\n\n"
        f"## {source_label}\n```sql\n{_truncate(schema, 6000)}\n```\n\n"
        f"For each data structure, determine if a corresponding table with "
        f"appropriate columns exists."
    )

    print(f"[runtime]   Evaluating {len(data_structures)} data structures ...")
    try:
        raw = llm.generate(user, system=system, temperature=0.1)
        verdicts = _parse_json(raw)
        if not isinstance(verdicts, list):
            verdicts = []
    except Exception as exc:
        logger.warning("Database evaluation failed: %s", exc)
        verdicts = []

    scores: list[DatabaseTestScore] = []
    for i, ds in enumerate(data_structures):
        v = verdicts[i] if i < len(verdicts) else {}
        verdict_str = v.get("verdict", "NO").upper() if isinstance(v, dict) else "NO"
        verdict = BinaryVerdict(verdict_str) if verdict_str in ("YES", "NO") else BinaryVerdict.NO
        reasoning = v.get("reasoning", "") if isinstance(v, dict) else ""
        if db_path:
            reasoning = f"[Schema from {db_path.name}] {reasoning}"
        scores.append(DatabaseTestScore(
            data_structure=ds,
            verdict=verdict,
            reasoning=reasoning,
        ))
    return scores


# ===================================================================
# 3.  FRONTEND RUNTIME TESTS
# ===================================================================

def run_frontend_tests(
    llm: BaseLLMClient,
    frontend_url: str,
    frontend_healthy: bool,
    task_instruction: str,
    frontend_code: str,
    ui_test_cases: list[dict],
) -> list[FrontendTestScore]:
    """Evaluate frontend test cases.

    Currently a hybrid approach:
    - If the frontend is healthy (returns HTTP 200), it gets credit for
      rendering. Individual test cases are still evaluated via static
      code analysis (until Playwright support is added).
    - If the frontend is NOT healthy, all tests score NO.
    """
    if not ui_test_cases:
        return []

    if not frontend_healthy:
        print("[runtime]   Frontend is not healthy — all UI tests score NO")
        return [
            FrontendTestScore(
                test_case=tc["task"],
                expected_result=tc["expected_result"],
                verdict=FrontendVerdict.NO,
                reasoning="Frontend container failed to start or is not returning HTTP 200.",
            )
            for tc in ui_test_cases
        ]

    # Fetch the actual HTML to confirm rendering
    html_snippet = ""
    try:
        resp = httpx.get(frontend_url, timeout=10)
        html_snippet = resp.text[:3000]
    except httpx.HTTPError:
        pass

    # Batched LLM evaluation — single call for all test cases
    cases_text = "\n".join(
        f"{i+1}. **Task:** {tc['task']}\n   **Expected:** {tc['expected_result']}"
        for i, tc in enumerate(ui_test_cases)
    )

    system = (
        "You are a frontend QA evaluator. The frontend application is running "
        "and serving HTTP 200. You are given the task description, the frontend "
        "source code, and the served HTML page. Evaluate each UI test case.\n\n"
        'Output ONLY a JSON array: [{"test_index":0,"verdict":"YES"|"PARTIAL"|"NO","reasoning":"..."},...]'
    )
    user = (
        f"## Task Description\n{task_instruction}\n\n"
        f"## Frontend Source Code\n```\n{_truncate(frontend_code, 10000)}\n```\n\n"
    )
    if html_snippet:
        user += f"## Served HTML (index)\n```html\n{_truncate(html_snippet, 2000)}\n```\n\n"
    user += f"## UI Test Cases\n{cases_text}\n\nEvaluate each test case."

    print(f"[runtime]   Evaluating {len(ui_test_cases)} frontend test cases ...")
    try:
        raw = llm.generate(user, system=system, temperature=0.1)
        verdicts = _parse_json(raw)
        if not isinstance(verdicts, list):
            verdicts = []
    except Exception as exc:
        logger.warning("Frontend evaluation failed: %s", exc)
        verdicts = []

    scores: list[FrontendTestScore] = []
    for i, tc in enumerate(ui_test_cases):
        v = verdicts[i] if i < len(verdicts) else {}
        verdict_str = v.get("verdict", "NO").upper() if isinstance(v, dict) else "NO"
        if verdict_str in ("YES", "PARTIAL", "NO"):
            verdict = FrontendVerdict(verdict_str)
        else:
            verdict = FrontendVerdict.NO
        reasoning = v.get("reasoning", "") if isinstance(v, dict) else ""
        reasoning = f"[Frontend running ✓] {reasoning}"
        scores.append(FrontendTestScore(
            test_case=tc["task"],
            expected_result=tc["expected_result"],
            verdict=verdict,
            reasoning=reasoning,
        ))
    return scores


# ===================================================================
# 4.  APPEARANCE (static code analysis — no Playwright yet)
# ===================================================================

def run_appearance_test(
    llm: BaseLLMClient,
    task_instruction: str,
    frontend_code: str,
    frontend_healthy: bool,
) -> AppearanceScore:
    """Evaluate visual appearance via code analysis.

    When Playwright support is added, this will use a real screenshot
    sent to GPT-4o vision instead.
    """
    system = (
        "You are an expert UI/UX designer evaluator. Evaluate the visual "
        "quality of the generated frontend code on four criteria (1-5).\n\n"
        'Output ONLY JSON: {"layout":<1-5>,"color":<1-5>,"typography":<1-5>,'
        '"component_polish":<1-5>,"reasoning":"..."}'
    )
    context_note = ""
    if frontend_healthy:
        context_note = (
            "NOTE: The frontend application successfully builds, starts, and "
            "serves HTTP 200 in Docker, which confirms the code is functional.\n\n"
        )
    user = (
        f"{context_note}"
        f"## Task Description\n{task_instruction}\n\n"
        f"## Generated Frontend Code\n```\n{_truncate(frontend_code, 12000)}\n```\n\n"
        "Evaluate the visual quality. Respond with JSON only."
    )

    print("[runtime]   Evaluating appearance ...")
    try:
        raw = llm.generate(user, system=system, temperature=0.1)
        data = _parse_json(raw)
        layout = _clamp(data.get("layout", 1))
        color = _clamp(data.get("color", 1))
        typography = _clamp(data.get("typography", 1))
        polish = _clamp(data.get("component_polish", 1))
        overall = round((layout + color + typography + polish) / 4, 2)
        return AppearanceScore(
            layout=layout,
            color=color,
            typography=typography,
            component_polish=polish,
            overall=overall,
            reasoning=data.get("reasoning", ""),
        )
    except Exception as exc:
        logger.warning("Appearance scoring failed: %s", exc)
        return AppearanceScore(
            layout=1, color=1, typography=1, component_polish=1,
            overall=1.0, reasoning=f"Judge error: {exc}",
        )


def _clamp(value: Any) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return 1


# ===================================================================
# 5.  FULL RUNTIME EVALUATION ORCHESTRATOR
# ===================================================================

def evaluate_application_runtime(
    llm: BaseLLMClient,
    instance: ProjectInstance,
    task_id: str,
    pattern: str,
    difficulty: str,
    task_instruction: str,
    frontend_code: str,
    backend_code: str,
    migration_sql: str,
    ui_test_cases: list[dict],
    backend_test_cases: list[dict],
    data_structures: list[str],
) -> JudgeResult:
    """Run the full runtime evaluation for one generated application.

    This is the runtime equivalent of ``scoring.evaluate_application()``.
    It uses live HTTP requests and real database checks instead of pure
    static code analysis.
    """
    result = JudgeResult(
        task_id=task_id,
        pattern=pattern,
        difficulty=difficulty,
        judge_model=llm.model,
    )

    # --- Backend tests (real HTTP) ---
    if instance.backend_healthy and backend_test_cases:
        result.backend_tests = run_backend_tests(
            llm, instance.backend_url, backend_code, backend_test_cases,
        )
    elif backend_test_cases:
        result.backend_tests = [
            BackendTestScore(
                test_case=tc["instruction"],
                expected_result=tc["expected_result"],
                verdict=BinaryVerdict.NO,
                reasoning="Backend container is not healthy — cannot test.",
            )
            for tc in backend_test_cases
        ]

    time.sleep(_RATE_DELAY)

    # --- Database tests (real SQLite) ---
    if data_structures:
        result.database_tests = run_database_tests(
            llm, instance.project_dir, task_instruction,
            data_structures, migration_sql,
        )

    time.sleep(_RATE_DELAY)

    # --- Frontend tests ---
    if ui_test_cases:
        result.frontend_tests = run_frontend_tests(
            llm, instance.frontend_url, instance.frontend_healthy,
            task_instruction, frontend_code, ui_test_cases,
        )

    time.sleep(_RATE_DELAY)

    # --- Appearance ---
    if frontend_code.strip():
        result.appearance = run_appearance_test(
            llm, task_instruction, frontend_code, instance.frontend_healthy,
        )

    result.compute_aggregates()
    return result
