"""Post-scoring fix-attempt generator.

After the judge has scored an application (scores frozen on original work),
this module asks the LLM to suggest concrete fixes for every failed test
case.  The results are attached to the JudgeResult but never modify scores.
"""

from __future__ import annotations

import json
import re
import time

from ..llm import BaseLLMClient
from ..logger import get_logger
from .models import (
    BinaryVerdict,
    FixAttempt,
    FrontendVerdict,
    JudgeResult,
    JudgeTrace,
)

logger = get_logger(__name__)

_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_RATE_DELAY = 5


def _parse_json(raw: str):
    text = raw.strip()
    match = _CODE_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def generate_fix_attempts(
    llm: BaseLLMClient,
    result: JudgeResult,
    frontend_code: str,
    backend_code: str,
    migration_sql: str,
    task_instruction: str,
    trace: JudgeTrace | None = None,
) -> list[FixAttempt]:
    """Generate fix suggestions for all failed test cases in *result*.

    This runs AFTER scoring is complete.  It collects all failures,
    batches them into a single LLM call, and returns FixAttempt objects.
    """
    failures: list[dict] = []

    for t in result.frontend_tests:
        if t.verdict != FrontendVerdict.YES:
            failures.append({
                "dimension": "frontend",
                "test_case": t.test_case,
                "verdict": t.verdict.value,
                "reasoning": t.reasoning,
            })

    for t in result.backend_tests:
        if t.verdict != BinaryVerdict.YES:
            failures.append({
                "dimension": "backend",
                "test_case": t.test_case,
                "verdict": t.verdict.value,
                "reasoning": t.reasoning,
            })

    for t in result.database_tests:
        if t.verdict != BinaryVerdict.YES:
            failures.append({
                "dimension": "database",
                "test_case": t.data_structure,
                "verdict": t.verdict.value,
                "reasoning": t.reasoning,
            })

    if result.appearance and result.appearance.overall < 3.0:
        failures.append({
            "dimension": "appearance",
            "test_case": "visual_quality",
            "verdict": f"overall={result.appearance.overall}",
            "reasoning": result.appearance.reasoning,
        })

    if not failures:
        logger.info("No failures to fix for task=%s pattern=%s", result.task_id, result.pattern)
        return []

    logger.info(
        "Generating fix attempts for %d failures (task=%s pattern=%s)",
        len(failures), result.task_id, result.pattern,
    )
    print(f"[judge-fix] Generating fixes for {len(failures)} failures ...")

    failures_text = "\n".join(
        f"{i+1}. [{f['dimension']}] {f['test_case']}\n"
        f"   Verdict: {f['verdict']}\n"
        f"   Reasoning: {f['reasoning']}"
        for i, f in enumerate(failures)
    )

    code_context = ""
    if frontend_code.strip():
        code_context += f"## Frontend Code\n```\n{frontend_code[:8000]}\n```\n\n"
    if backend_code.strip():
        code_context += f"## Backend Code\n```\n{backend_code[:8000]}\n```\n\n"
    if migration_sql.strip():
        code_context += f"## Migration SQL\n```sql\n{migration_sql[:4000]}\n```\n\n"

    system = (
        "You are an expert full-stack developer. Given a list of failed test "
        "cases from a code evaluation, suggest concrete fixes for each failure. "
        "For each fix, provide a brief issue summary, what should change, and "
        "a code snippet or diff showing the fix.\n\n"
        "Output ONLY a JSON array:\n"
        '[{"dimension":"frontend|backend|database|appearance",'
        '"test_case":"...",'
        '"issue_summary":"...",'
        '"suggested_fix":"...",'
        '"code_patch":"..."},...]\n\n'
        "Keep patches concise but specific enough to be actionable."
    )

    user = (
        f"## Task Description\n{task_instruction}\n\n"
        f"{code_context}"
        f"## Failed Test Cases\n{failures_text}\n\n"
        "Suggest a fix for each failure."
    )

    try:
        t0 = time.monotonic()
        raw = llm.generate(user, system=system, temperature=0.2)
        latency = time.monotonic() - t0
        data = _parse_json(raw)

        if trace:
            trace.add_llm_call(
                "generate_fixes",
                system_prompt=system, user_prompt=user,
                raw_response=raw, parsed_result=data,
                latency_seconds=latency, model=llm.model,
            )

        if not isinstance(data, list):
            logger.warning("Fix generation returned non-list")
            return []

        attempts: list[FixAttempt] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            attempts.append(FixAttempt(
                dimension=item.get("dimension", "unknown"),
                test_case=item.get("test_case", ""),
                issue_summary=item.get("issue_summary", ""),
                suggested_fix=item.get("suggested_fix", ""),
                code_patch=item.get("code_patch", ""),
            ))

        print(f"[judge-fix] Generated {len(attempts)} fix suggestions")
        return attempts

    except Exception as exc:
        logger.warning("Fix generation failed: %s", exc)
        if trace:
            trace.add_llm_call(
                "generate_fixes",
                system_prompt=system, user_prompt=user,
                error=str(exc), model=llm.model,
            )
        return []
