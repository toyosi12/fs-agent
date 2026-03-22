"""Scoring engine — calls the judge LLM and parses structured responses."""

from __future__ import annotations

import json
import re
import time

from ..llm import BaseLLMClient
from ..logger import get_logger
from .models import (
    AppearanceScore,
    BackendTestScore,
    BinaryVerdict,
    DatabaseTestScore,
    FrontendTestScore,
    FrontendVerdict,
    JudgeResult,
    JudgeTrace,
)
from .prompts import (
    appearance_prompt,
    backend_test_prompt,
    database_test_prompt,
    frontend_test_prompt,
)

logger = get_logger(__name__)

_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from an LLM response, stripping fences."""
    text = raw.strip()
    # Try extracting from code fences first
    match = _CODE_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def score_frontend_test(
    llm: BaseLLMClient,
    task_instruction: str,
    test_case: dict,
    frontend_code: str,
    trace: JudgeTrace | None = None,
    test_index: int = 0,
) -> FrontendTestScore:
    """Evaluate a single frontend test case."""
    logger.info("  scoring frontend test: %s", test_case.get('task', '')[:80])
    system, user = frontend_test_prompt(task_instruction, test_case, frontend_code)
    try:
        t0 = time.monotonic()
        raw = llm.generate(user, system=system, temperature=0.1)
        latency = time.monotonic() - t0
        data = _parse_json_response(raw)
        verdict_str = data.get("verdict", "NO").upper()
        verdict = FrontendVerdict(verdict_str) if verdict_str in ("YES", "PARTIAL", "NO") else FrontendVerdict.NO
        if trace:
            trace.add_llm_call(
                f"frontend_test_{test_index}",
                system_prompt=system, user_prompt=user,
                raw_response=raw, parsed_result=data,
                latency_seconds=latency, model=llm.model,
            )
        return FrontendTestScore(
            test_case=test_case.get("task", ""),
            expected_result=test_case.get("expected_result", ""),
            verdict=verdict,
            reasoning=data.get("reasoning", ""),
        )
    except Exception as exc:
        logger.warning("Frontend test scoring failed: %s", exc)
        if trace:
            trace.add_llm_call(
                f"frontend_test_{test_index}",
                system_prompt=system, user_prompt=user,
                error=str(exc), model=llm.model,
            )
        return FrontendTestScore(
            test_case=test_case.get("task", ""),
            expected_result=test_case.get("expected_result", ""),
            verdict=FrontendVerdict.NO,
            reasoning=f"Judge error: {exc}",
        )


def score_backend_test(
    llm: BaseLLMClient,
    task_instruction: str,
    test_case: dict,
    backend_code: str,
    trace: JudgeTrace | None = None,
    test_index: int = 0,
) -> BackendTestScore:
    """Evaluate a single backend test case."""
    logger.info("  scoring backend test: %s", test_case.get('instruction', '')[:80])
    system, user = backend_test_prompt(task_instruction, test_case, backend_code)
    try:
        t0 = time.monotonic()
        raw = llm.generate(user, system=system, temperature=0.1)
        latency = time.monotonic() - t0
        data = _parse_json_response(raw)
        verdict_str = data.get("verdict", "NO").upper()
        verdict = BinaryVerdict(verdict_str) if verdict_str in ("YES", "NO") else BinaryVerdict.NO
        if trace:
            trace.add_llm_call(
                f"backend_test_{test_index}",
                system_prompt=system, user_prompt=user,
                raw_response=raw, parsed_result=data,
                latency_seconds=latency, model=llm.model,
            )
        return BackendTestScore(
            test_case=test_case.get("instruction", ""),
            expected_result=test_case.get("expected_result", ""),
            verdict=verdict,
            reasoning=data.get("reasoning", ""),
        )
    except Exception as exc:
        logger.warning("Backend test scoring failed: %s", exc)
        if trace:
            trace.add_llm_call(
                f"backend_test_{test_index}",
                system_prompt=system, user_prompt=user,
                error=str(exc), model=llm.model,
            )
        return BackendTestScore(
            test_case=test_case.get("instruction", ""),
            expected_result=test_case.get("expected_result", ""),
            verdict=BinaryVerdict.NO,
            reasoning=f"Judge error: {exc}",
        )


def score_database_test(
    llm: BaseLLMClient,
    task_instruction: str,
    data_structure: str,
    migration_sql: str,
    trace: JudgeTrace | None = None,
    test_index: int = 0,
) -> DatabaseTestScore:
    """Evaluate a single data structure requirement."""
    logger.info("  scoring database test: %s", data_structure)
    system, user = database_test_prompt(task_instruction, data_structure, migration_sql)
    try:
        t0 = time.monotonic()
        raw = llm.generate(user, system=system, temperature=0.1)
        latency = time.monotonic() - t0
        data = _parse_json_response(raw)
        verdict_str = data.get("verdict", "NO").upper()
        verdict = BinaryVerdict(verdict_str) if verdict_str in ("YES", "NO") else BinaryVerdict.NO
        if trace:
            trace.add_llm_call(
                f"database_test_{test_index}",
                system_prompt=system, user_prompt=user,
                raw_response=raw, parsed_result=data,
                latency_seconds=latency, model=llm.model,
            )
        return DatabaseTestScore(
            data_structure=data_structure,
            verdict=verdict,
            reasoning=data.get("reasoning", ""),
        )
    except Exception as exc:
        logger.warning("Database test scoring failed: %s", exc)
        if trace:
            trace.add_llm_call(
                f"database_test_{test_index}",
                system_prompt=system, user_prompt=user,
                error=str(exc), model=llm.model,
            )
        return DatabaseTestScore(
            data_structure=data_structure,
            verdict=BinaryVerdict.NO,
            reasoning=f"Judge error: {exc}",
        )


def score_appearance(
    llm: BaseLLMClient,
    task_instruction: str,
    frontend_code: str,
    trace: JudgeTrace | None = None,
) -> AppearanceScore:
    """Evaluate the visual appearance of the frontend."""
    logger.info("  scoring appearance")
    system, user = appearance_prompt(task_instruction, frontend_code)
    try:
        t0 = time.monotonic()
        raw = llm.generate(user, system=system, temperature=0.1)
        latency = time.monotonic() - t0
        data = _parse_json_response(raw)
        layout = _clamp(data.get("layout", 1), 1, 5)
        color = _clamp(data.get("color", 1), 1, 5)
        typography = _clamp(data.get("typography", 1), 1, 5)
        polish = _clamp(data.get("component_polish", 1), 1, 5)
        overall = round((layout + color + typography + polish) / 4, 2)
        if trace:
            trace.add_llm_call(
                "appearance",
                system_prompt=system, user_prompt=user,
                raw_response=raw, parsed_result=data,
                latency_seconds=latency, model=llm.model,
            )
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
        if trace:
            trace.add_llm_call(
                "appearance",
                system_prompt=system, user_prompt=user,
                error=str(exc), model=llm.model,
            )
        return AppearanceScore(
            layout=1,
            color=1,
            typography=1,
            component_polish=1,
            overall=1.0,
            reasoning=f"Judge error: {exc}",
        )


def evaluate_application(
    llm: BaseLLMClient,
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
    trace: JudgeTrace | None = None,
) -> JudgeResult:
    """Run the full judge evaluation for a single generated application."""

    result = JudgeResult(
        task_id=task_id,
        pattern=pattern,
        difficulty=difficulty,
        judge_model=llm.model,
    )

    # --- Frontend tests ---
    for i, tc in enumerate(ui_test_cases):
        if i > 0:
            time.sleep(5)
        score = score_frontend_test(llm, task_instruction, tc, frontend_code, trace=trace, test_index=i)
        result.frontend_tests.append(score)

    # --- Backend tests ---
    for i, tc in enumerate(backend_test_cases):
        if i > 0 or ui_test_cases:
            time.sleep(5)
        score = score_backend_test(llm, task_instruction, tc, backend_code, trace=trace, test_index=i)
        result.backend_tests.append(score)

    # --- Database tests ---
    for i, ds in enumerate(data_structures):
        if i > 0 or backend_test_cases or ui_test_cases:
            time.sleep(5)
        score = score_database_test(llm, task_instruction, ds, migration_sql, trace=trace, test_index=i)
        result.database_tests.append(score)

    # --- Appearance ---
    if frontend_code.strip():
        if data_structures or backend_test_cases or ui_test_cases:
            time.sleep(5)
        result.appearance = score_appearance(llm, task_instruction, frontend_code, trace=trace)

    result.compute_aggregates()
    return result


def _clamp(value: int | float, lo: int, hi: int) -> int:
    """Clamp a numeric value to [lo, hi] and cast to int."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo
