"""Tests for judge tracing and fix-attempt generation."""

from __future__ import annotations

import json

from fs_agent.judge.models import (
    BackendTestScore,
    BinaryVerdict,
    DatabaseTestScore,
    FixAttempt,
    FrontendTestScore,
    FrontendVerdict,
    JudgeResult,
    JudgeTrace,
    LLMCallTrace,
    HTTPRequestTrace,
    TraceEntry,
)
from fs_agent.judge.scoring import (
    evaluate_application,
    score_frontend_test,
    score_backend_test,
    score_database_test,
    score_appearance,
)
from fs_agent.judge.fixes import generate_fix_attempts
from fs_agent.llm import DummyLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeJudgeLLM(DummyLLMClient):
    """LLM stub that returns pre-configured JSON responses."""

    def __init__(self, response) -> None:
        super().__init__("fake-judge")
        self._response = response
        self._call_count = 0
        self._calls: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        self._call_count += 1
        self._calls.append({"prompt": prompt, "system": system})
        est_prompt = len(prompt) // 4
        est_system = len(system) // 4 if system else 0
        if isinstance(self._response, list):
            resp = self._response[min(self._call_count - 1, len(self._response) - 1)]
        else:
            resp = self._response
        result = json.dumps(resp)
        self._record_usage(
            prompt_tokens=est_prompt + est_system,
            completion_tokens=len(result) // 4,
        )
        return result


class SequenceLLM(DummyLLMClient):
    """LLM stub that returns different responses for sequential calls."""

    def __init__(self, responses: list) -> None:
        super().__init__("fake-judge")
        self._responses = responses
        self._idx = 0

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return json.dumps(resp)


# ---------------------------------------------------------------------------
# Trace model tests
# ---------------------------------------------------------------------------


class TestJudgeTrace:
    def test_add_llm_call(self):
        trace = JudgeTrace(task_id="001", pattern="sequential", mode="static")
        trace.add_llm_call(
            "frontend_test_0",
            system_prompt="sys",
            user_prompt="user",
            raw_response='{"verdict":"YES"}',
            parsed_result={"verdict": "YES"},
            latency_seconds=1.5,
            model="gpt-4o",
        )
        assert len(trace.entries) == 1
        entry = trace.entries[0]
        assert entry.step == "frontend_test_0"
        assert entry.llm_call is not None
        assert entry.llm_call.system_prompt == "sys"
        assert entry.llm_call.latency_seconds == 1.5
        assert entry.http_request is None

    def test_add_http_request(self):
        trace = JudgeTrace(task_id="001", pattern="parallel", mode="runtime")
        trace.add_http_request(
            "backend_http_0",
            method="POST",
            url="http://localhost:4000/api/todos",
            request_body={"title": "test"},
            status_code=201,
            response_body='{"id":1}',
            latency_seconds=0.05,
        )
        assert len(trace.entries) == 1
        entry = trace.entries[0]
        assert entry.http_request is not None
        assert entry.http_request.method == "POST"
        assert entry.http_request.status_code == 201
        assert entry.llm_call is None

    def test_multiple_entries(self):
        trace = JudgeTrace(task_id="001", pattern="sequential", mode="static")
        trace.add_llm_call("step_1", raw_response="r1")
        trace.add_llm_call("step_2", raw_response="r2")
        trace.add_http_request("step_3", method="GET", url="/api")
        assert len(trace.entries) == 3

    def test_serialization(self):
        trace = JudgeTrace(task_id="001", pattern="sequential", mode="static")
        trace.add_llm_call("test_step", raw_response='{"ok": true}', model="gpt-4o")
        data = trace.model_dump(mode="json")
        assert data["task_id"] == "001"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["llm_call"]["model"] == "gpt-4o"
        # Round-trip
        trace2 = JudgeTrace.model_validate(data)
        assert len(trace2.entries) == 1

    def test_error_trace(self):
        trace = JudgeTrace(task_id="001", pattern="sequential", mode="static")
        trace.add_llm_call(
            "failed_step",
            system_prompt="sys",
            user_prompt="user",
            error="Connection timeout",
            model="gpt-4o",
        )
        entry = trace.entries[0]
        assert entry.llm_call.error == "Connection timeout"
        assert entry.llm_call.raw_response == ""


# ---------------------------------------------------------------------------
# Scoring with trace tests
# ---------------------------------------------------------------------------


class TestScoringWithTrace:
    def test_frontend_test_records_trace(self):
        llm = FakeJudgeLLM({"verdict": "YES", "reasoning": "ok"})
        trace = JudgeTrace(task_id="001", pattern="seq", mode="static")
        score = score_frontend_test(
            llm,
            task_instruction="Build app",
            test_case={"task": "Add item", "expected_result": "Item shows"},
            frontend_code="function App() {}",
            trace=trace,
            test_index=0,
        )
        assert score.verdict == FrontendVerdict.YES
        assert len(trace.entries) == 1
        assert trace.entries[0].step == "frontend_test_0"
        assert trace.entries[0].llm_call.latency_seconds > 0

    def test_backend_test_records_trace(self):
        llm = FakeJudgeLLM({"verdict": "NO", "reasoning": "missing"})
        trace = JudgeTrace(task_id="001", pattern="seq", mode="static")
        score = score_backend_test(
            llm,
            task_instruction="Build app",
            test_case={"instruction": "POST /api/items", "expected_result": "201"},
            backend_code="app.get('/health');",
            trace=trace,
            test_index=2,
        )
        assert score.verdict == BinaryVerdict.NO
        assert trace.entries[0].step == "backend_test_2"

    def test_database_test_records_trace(self):
        llm = FakeJudgeLLM({"verdict": "YES", "reasoning": "table exists"})
        trace = JudgeTrace(task_id="001", pattern="seq", mode="static")
        score = score_database_test(
            llm,
            task_instruction="Build app",
            data_structure="users",
            migration_sql="CREATE TABLE users (id INTEGER);",
            trace=trace,
            test_index=0,
        )
        assert score.verdict == BinaryVerdict.YES
        assert trace.entries[0].step == "database_test_0"

    def test_appearance_records_trace(self):
        llm = FakeJudgeLLM({
            "layout": 4, "color": 3, "typography": 4,
            "component_polish": 3, "reasoning": "decent design",
        })
        trace = JudgeTrace(task_id="001", pattern="seq", mode="static")
        score = score_appearance(
            llm,
            task_instruction="Build app",
            frontend_code="function App() {}",
            trace=trace,
        )
        assert score.overall > 1.0
        assert trace.entries[0].step == "appearance"

    def test_evaluate_application_creates_full_trace(self):
        llm = SequenceLLM([
            {"verdict": "YES", "reasoning": "ok"},          # frontend_test_0
            {"verdict": "NO", "reasoning": "missing"},       # backend_test_0
            {"verdict": "YES", "reasoning": "table ok"},     # database_test_0
            {"layout": 3, "color": 3, "typography": 3,       # appearance
             "component_polish": 3, "reasoning": "ok"},
        ])
        trace = JudgeTrace(task_id="001", pattern="seq", mode="static")
        result = evaluate_application(
            llm=llm,
            task_id="001",
            pattern="seq",
            difficulty="easy",
            task_instruction="Build a todo app",
            frontend_code="function App() {}",
            backend_code="app.get('/');",
            migration_sql="CREATE TABLE todos (id INTEGER);",
            ui_test_cases=[{"task": "Add item", "expected_result": "Shows item"}],
            backend_test_cases=[{"instruction": "POST /api/todos", "expected_result": "201"}],
            data_structures=["todos"],
            trace=trace,
        )
        # Should have traces for: frontend_test_0, backend_test_0, database_test_0, appearance
        assert len(trace.entries) == 4
        steps = [e.step for e in trace.entries]
        assert "frontend_test_0" in steps
        assert "backend_test_0" in steps
        assert "database_test_0" in steps
        assert "appearance" in steps

    def test_scoring_without_trace_works(self):
        """Passing trace=None should still work normally."""
        llm = FakeJudgeLLM({"verdict": "YES", "reasoning": "ok"})
        score = score_frontend_test(
            llm,
            task_instruction="Build app",
            test_case={"task": "Add item", "expected_result": "Shows"},
            frontend_code="function App() {}",
            trace=None,
        )
        assert score.verdict == FrontendVerdict.YES

    def test_failed_scoring_records_error_trace(self):
        """When LLM returns invalid JSON, the trace records the error."""

        class BadLLM(DummyLLMClient):
            def generate(self, prompt, *, system=None, temperature=0.2):
                return "not json at all"

        llm = BadLLM("fake")
        trace = JudgeTrace(task_id="001", pattern="seq", mode="static")
        score = score_frontend_test(
            llm,
            task_instruction="Build app",
            test_case={"task": "Add item", "expected_result": "Shows"},
            frontend_code="function App() {}",
            trace=trace,
            test_index=0,
        )
        assert score.verdict == FrontendVerdict.NO
        assert len(trace.entries) == 1
        assert trace.entries[0].llm_call.error is not None


# ---------------------------------------------------------------------------
# Fix attempt model tests
# ---------------------------------------------------------------------------


class TestFixAttemptModel:
    def test_fix_attempt_fields(self):
        fix = FixAttempt(
            dimension="frontend",
            test_case="Add item",
            issue_summary="Missing event handler",
            suggested_fix="Add onClick handler",
            code_patch="+ onClick={() => addItem()}",
        )
        assert fix.dimension == "frontend"
        assert fix.code_patch.startswith("+")

    def test_judge_result_with_fixes(self):
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.fix_attempts = [
            FixAttempt(
                dimension="backend",
                test_case="POST /api/todos",
                issue_summary="Route missing",
                suggested_fix="Add POST route",
            ),
        ]
        data = result.model_dump(mode="json")
        assert len(data["fix_attempts"]) == 1
        assert data["fix_attempts"][0]["dimension"] == "backend"

    def test_judge_result_default_empty_fixes(self):
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        assert result.fix_attempts == []


# ---------------------------------------------------------------------------
# Fix generation tests
# ---------------------------------------------------------------------------


class TestGenerateFixAttempts:
    def test_generates_fixes_for_failures(self):
        llm = FakeJudgeLLM([[
            {
                "dimension": "backend",
                "test_case": "POST /api/todos",
                "issue_summary": "Missing route",
                "suggested_fix": "Add router.post('/api/todos')",
                "code_patch": "+ router.post('/api/todos', ...)",
            },
        ]])
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.backend_tests = [
            BackendTestScore(
                test_case="POST /api/todos",
                expected_result="201",
                verdict=BinaryVerdict.NO,
                reasoning="endpoint missing",
            ),
        ]
        fixes = generate_fix_attempts(
            llm=llm,
            result=result,
            frontend_code="",
            backend_code="app.get('/');",
            migration_sql="",
            task_instruction="Build a todo app",
        )
        assert len(fixes) == 1
        assert fixes[0].dimension == "backend"

    def test_no_fixes_when_all_pass(self):
        llm = FakeJudgeLLM({})
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.frontend_tests = [
            FrontendTestScore(
                test_case="Add item",
                expected_result="Shows",
                verdict=FrontendVerdict.YES,
                reasoning="ok",
            ),
        ]
        result.backend_tests = [
            BackendTestScore(
                test_case="POST /api/todos",
                expected_result="201",
                verdict=BinaryVerdict.YES,
                reasoning="ok",
            ),
        ]
        fixes = generate_fix_attempts(
            llm=llm,
            result=result,
            frontend_code="",
            backend_code="",
            migration_sql="",
            task_instruction="Build a todo app",
        )
        assert fixes == []

    def test_fix_generation_with_trace(self):
        llm = FakeJudgeLLM([[
            {
                "dimension": "database",
                "test_case": "users",
                "issue_summary": "No table",
                "suggested_fix": "Add migration",
                "code_patch": "CREATE TABLE users (...)",
            },
        ]])
        trace = JudgeTrace(task_id="001", pattern="seq", mode="static")
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.database_tests = [
            DatabaseTestScore(
                data_structure="users",
                verdict=BinaryVerdict.NO,
                reasoning="missing table",
            ),
        ]
        fixes = generate_fix_attempts(
            llm=llm,
            result=result,
            frontend_code="",
            backend_code="",
            migration_sql="",
            task_instruction="Build app",
            trace=trace,
        )
        assert len(fixes) == 1
        # Trace should have the fix generation call
        assert any(e.step == "generate_fixes" for e in trace.entries)

    def test_fix_generation_handles_llm_error(self):
        class ErrorLLM(DummyLLMClient):
            def generate(self, prompt, *, system=None, temperature=0.2):
                raise ConnectionError("LLM unavailable")

        llm = ErrorLLM("fake")
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.backend_tests = [
            BackendTestScore(
                test_case="POST /api/todos",
                expected_result="201",
                verdict=BinaryVerdict.NO,
                reasoning="missing",
            ),
        ]
        fixes = generate_fix_attempts(
            llm=llm,
            result=result,
            frontend_code="",
            backend_code="",
            migration_sql="",
            task_instruction="Build app",
        )
        assert fixes == []

    def test_partial_frontend_gets_fix(self):
        llm = FakeJudgeLLM([[
            {
                "dimension": "frontend",
                "test_case": "Add item",
                "issue_summary": "Partial implementation",
                "suggested_fix": "Complete the handler",
                "code_patch": "",
            },
        ]])
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.frontend_tests = [
            FrontendTestScore(
                test_case="Add item",
                expected_result="Shows",
                verdict=FrontendVerdict.PARTIAL,
                reasoning="half done",
            ),
        ]
        fixes = generate_fix_attempts(
            llm=llm,
            result=result,
            frontend_code="function App() {}",
            backend_code="",
            migration_sql="",
            task_instruction="Build app",
        )
        assert len(fixes) == 1
        assert fixes[0].dimension == "frontend"

    def test_low_appearance_gets_fix(self):
        from fs_agent.judge.models import AppearanceScore
        llm = FakeJudgeLLM([[
            {
                "dimension": "appearance",
                "test_case": "visual_quality",
                "issue_summary": "Poor layout",
                "suggested_fix": "Add CSS grid",
                "code_patch": ".container { display: grid; }",
            },
        ]])
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.appearance = AppearanceScore(
            layout=1, color=2, typography=2, component_polish=1,
            overall=1.5, reasoning="very basic",
        )
        fixes = generate_fix_attempts(
            llm=llm,
            result=result,
            frontend_code="<div>App</div>",
            backend_code="",
            migration_sql="",
            task_instruction="Build app",
        )
        assert len(fixes) == 1
        assert fixes[0].dimension == "appearance"

    def test_good_appearance_no_fix(self):
        from fs_agent.judge.models import AppearanceScore
        llm = FakeJudgeLLM({})
        result = JudgeResult(task_id="001", pattern="seq", difficulty="easy")
        result.appearance = AppearanceScore(
            layout=4, color=4, typography=4, component_polish=4,
            overall=4.0, reasoning="good",
        )
        fixes = generate_fix_attempts(
            llm=llm,
            result=result,
            frontend_code="<div>App</div>",
            backend_code="",
            migration_sql="",
            task_instruction="Build app",
        )
        assert fixes == []
