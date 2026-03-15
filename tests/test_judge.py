"""Tests for the LLM-as-a-judge scoring module."""

from __future__ import annotations

import json

from fs_agent.judge.models import (
    AppearanceScore,
    BackendTestScore,
    BinaryVerdict,
    DatabaseTestScore,
    FrontendTestScore,
    FrontendVerdict,
    JudgeResult,
)
from fs_agent.judge.scoring import (
    _parse_json_response,
    evaluate_application,
    score_appearance,
    score_backend_test,
    score_database_test,
    score_frontend_test,
)
from fs_agent.llm import DummyLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeJudgeLLM(DummyLLMClient):
    """LLM stub that returns pre-configured JSON responses."""

    def __init__(self, response: dict) -> None:
        super().__init__("fake-judge")
        self._response = response

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        est_prompt = len(prompt) // 4
        est_system = len(system) // 4 if system else 0
        result = json.dumps(self._response)
        self._record_usage(
            prompt_tokens=est_prompt + est_system,
            completion_tokens=len(result) // 4,
        )
        return result


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_judge_result_compute_aggregates():
    result = JudgeResult(task_id="001", pattern="sequential", difficulty="easy")
    result.frontend_tests = [
        FrontendTestScore(test_case="t1", expected_result="r1", verdict=FrontendVerdict.YES, reasoning="ok"),
        FrontendTestScore(test_case="t2", expected_result="r2", verdict=FrontendVerdict.PARTIAL, reasoning="half"),
        FrontendTestScore(test_case="t3", expected_result="r3", verdict=FrontendVerdict.NO, reasoning="miss"),
    ]
    result.backend_tests = [
        BackendTestScore(test_case="b1", expected_result="r1", verdict=BinaryVerdict.YES, reasoning="ok"),
        BackendTestScore(test_case="b2", expected_result="r2", verdict=BinaryVerdict.NO, reasoning="miss"),
    ]
    result.database_tests = [
        DatabaseTestScore(data_structure="users", verdict=BinaryVerdict.YES, reasoning="ok"),
    ]
    result.compute_aggregates()

    assert result.frontend_yes == 1
    assert result.frontend_partial == 1
    assert result.frontend_no == 1
    assert result.frontend_total == 3
    assert result.frontend_weighted_accuracy == round((1 + 0.5 * 1) / 3, 4)

    assert result.backend_yes == 1
    assert result.backend_no == 1
    assert result.backend_accuracy == 0.5

    assert result.database_yes == 1
    assert result.database_accuracy == 1.0


def test_parse_json_response_plain():
    raw = '{"verdict": "YES", "reasoning": "looks good"}'
    data = _parse_json_response(raw)
    assert data["verdict"] == "YES"


def test_parse_json_response_fenced():
    raw = '```json\n{"verdict": "NO", "reasoning": "missing"}\n```'
    data = _parse_json_response(raw)
    assert data["verdict"] == "NO"


# ---------------------------------------------------------------------------
# Scoring function tests (with stub LLM)
# ---------------------------------------------------------------------------


def test_score_frontend_test_yes():
    llm = FakeJudgeLLM({"verdict": "YES", "reasoning": "fully implemented"})
    score = score_frontend_test(
        llm,
        task_instruction="Build a todo app",
        test_case={"task": "Add a todo item", "expected_result": "Item appears in list"},
        frontend_code="function App() { return <div>Todo</div>; }",
    )
    assert score.verdict == FrontendVerdict.YES
    assert score.reasoning == "fully implemented"


def test_score_backend_test_no():
    llm = FakeJudgeLLM({"verdict": "NO", "reasoning": "endpoint not found"})
    score = score_backend_test(
        llm,
        task_instruction="Build a todo app",
        test_case={"instruction": "POST /api/todos", "expected_result": "201 created"},
        backend_code="router.get('/healthz', ...);",
    )
    assert score.verdict == BinaryVerdict.NO


def test_score_database_test_yes():
    llm = FakeJudgeLLM({"verdict": "YES", "reasoning": "table exists"})
    score = score_database_test(
        llm,
        task_instruction="Build a todo app",
        data_structure="todos",
        migration_sql="CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY);",
    )
    assert score.verdict == BinaryVerdict.YES


def test_score_appearance():
    llm = FakeJudgeLLM({
        "layout": 4,
        "color": 3,
        "typography": 4,
        "component_polish": 3,
        "reasoning": "decent design",
    })
    score = score_appearance(
        llm,
        task_instruction="Build a todo app",
        frontend_code="function App() { return <div className='p-4'>Todo</div>; }",
    )
    assert score.layout == 4
    assert score.color == 3
    assert score.overall == 3.5


def test_evaluate_application_full():
    """End-to-end test with stub LLM returning YES for everything."""
    llm = FakeJudgeLLM({"verdict": "YES", "reasoning": "good",
                         "layout": 5, "color": 5, "typography": 5,
                         "component_polish": 5})
    result = evaluate_application(
        llm=llm,
        task_id="000001",
        pattern="sequential",
        difficulty="medium",
        task_instruction="Build a stock report app",
        frontend_code="function App() { return <div>Stocks</div>; }",
        backend_code="router.get('/api/stocks', ...);",
        migration_sql="CREATE TABLE stocks (id INTEGER PRIMARY KEY);",
        ui_test_cases=[
            {"task": "Search stocks", "expected_result": "Results shown"},
        ],
        backend_test_cases=[
            {"instruction": "GET /api/stocks", "expected_result": "200 with data"},
        ],
        data_structures=["stocks"],
    )

    assert result.task_id == "000001"
    assert result.difficulty == "medium"
    assert result.frontend_weighted_accuracy == 1.0
    assert result.backend_accuracy == 1.0
    assert result.database_accuracy == 1.0
    assert result.appearance is not None
    assert result.appearance.overall == 5.0


def test_scoring_handles_llm_error():
    """Scoring should gracefully handle LLM failures."""

    class BrokenLLM(DummyLLMClient):
        def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
            raise RuntimeError("API down")

    llm = BrokenLLM("broken")
    score = score_frontend_test(
        llm,
        task_instruction="Build a todo app",
        test_case={"task": "Add todo", "expected_result": "Item appears"},
        frontend_code="code",
    )
    assert score.verdict == FrontendVerdict.NO
    assert "Judge error" in score.reasoning
