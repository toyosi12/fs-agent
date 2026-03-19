"""Pydantic models for judge scoring results.

These models capture the FullStack-Bench rubric scores and link each
evaluation back to the originating task and its difficulty level.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Score enums
# ---------------------------------------------------------------------------


class FrontendVerdict(str, Enum):
    YES = "YES"
    PARTIAL = "PARTIAL"
    NO = "NO"


class BinaryVerdict(str, Enum):
    YES = "YES"
    NO = "NO"


# ---------------------------------------------------------------------------
# Per-dimension scores
# ---------------------------------------------------------------------------


class FrontendTestScore(BaseModel):
    """Frontend functional test evaluation.

    Each ui_instruct test case is judged as YES / PARTIAL / NO.
    - YES: the task/expected_result is clearly achievable from the generated code.
    - PARTIAL: some elements work but not everything.
    - NO: the feature is missing or non-functional.
    """

    test_case: str = Field(description="The UI test instruction evaluated")
    expected_result: str = Field(description="Expected outcome from the dataset")
    verdict: FrontendVerdict = Field(description="YES | PARTIAL | NO")
    reasoning: str = Field(description="Judge's explanation for the verdict")


class BackendTestScore(BaseModel):
    """Backend API test evaluation.

    Each backend_test_case is judged as YES / NO based on whether the
    generated backend code correctly implements the expected behavior.
    """

    test_case: str = Field(description="The backend test instruction evaluated")
    expected_result: str = Field(description="Expected outcome from the dataset")
    verdict: BinaryVerdict = Field(description="YES | NO")
    reasoning: str = Field(description="Judge's explanation for the verdict")


class DatabaseTestScore(BaseModel):
    """Database schema evaluation.

    Checks whether the generated migrations / schema correctly define the
    data structures required by the task.
    """

    data_structure: str = Field(description="The required data structure name")
    verdict: BinaryVerdict = Field(description="YES | NO")
    reasoning: str = Field(description="Judge's explanation for the verdict")


class AppearanceScore(BaseModel):
    """Visual appearance evaluation on a 1-5 scale.

    Criteria evaluated:
    1. Layout & structure (proper spacing, alignment, responsive design)
    2. Color & theming (matches spec, consistent palette)
    3. Typography & readability (font sizes, hierarchy, contrast)
    4. Component polish (buttons, forms, cards look professional)
    """

    layout: int = Field(ge=1, le=5, description="Layout & structure score 1-5")
    color: int = Field(ge=1, le=5, description="Color & theming score 1-5")
    typography: int = Field(ge=1, le=5, description="Typography & readability score 1-5")
    component_polish: int = Field(ge=1, le=5, description="Component polish score 1-5")
    overall: float = Field(ge=1.0, le=5.0, description="Average of the four criteria")
    reasoning: str = Field(description="Judge's explanation for the scores")


# ---------------------------------------------------------------------------
# Trace models — capture every LLM / HTTP interaction for investigation
# ---------------------------------------------------------------------------


class LLMCallTrace(BaseModel):
    """Record of a single LLM interaction during judge evaluation."""

    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    parsed_result: Any = None
    error: str | None = None
    latency_seconds: float = 0.0
    model: str = ""


class HTTPRequestTrace(BaseModel):
    """Record of a single HTTP request made during runtime evaluation."""

    method: str = ""
    url: str = ""
    request_body: Any = None
    status_code: int | None = None
    response_body: str = ""
    error: str | None = None
    latency_seconds: float = 0.0


class TraceEntry(BaseModel):
    """One step in the judge evaluation trace."""

    step: str = Field(description="Identifies the scoring step, e.g. 'frontend_test_0'")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    llm_call: LLMCallTrace | None = None
    http_request: HTTPRequestTrace | None = None


class JudgeTrace(BaseModel):
    """Complete trace of all interactions for one (task × pattern) evaluation."""

    task_id: str = ""
    pattern: str = ""
    mode: str = ""
    started_at: str = ""
    finished_at: str = ""
    entries: list[TraceEntry] = Field(default_factory=list)

    def add_llm_call(
        self,
        step: str,
        *,
        system_prompt: str = "",
        user_prompt: str = "",
        raw_response: str = "",
        parsed_result: Any = None,
        error: str | None = None,
        latency_seconds: float = 0.0,
        model: str = "",
    ) -> None:
        self.entries.append(TraceEntry(
            step=step,
            llm_call=LLMCallTrace(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw_response,
                parsed_result=parsed_result,
                error=error,
                latency_seconds=latency_seconds,
                model=model,
            ),
        ))

    def add_http_request(
        self,
        step: str,
        *,
        method: str = "",
        url: str = "",
        request_body: Any = None,
        status_code: int | None = None,
        response_body: str = "",
        error: str | None = None,
        latency_seconds: float = 0.0,
    ) -> None:
        self.entries.append(TraceEntry(
            step=step,
            http_request=HTTPRequestTrace(
                method=method,
                url=url,
                request_body=request_body,
                status_code=status_code,
                response_body=response_body,
                error=error,
                latency_seconds=latency_seconds,
            ),
        ))


# ---------------------------------------------------------------------------
# Fix-attempt models — post-scoring remediation suggestions
# ---------------------------------------------------------------------------


class FixAttempt(BaseModel):
    """A suggested fix for a failed test case, generated after scoring."""

    dimension: str = Field(description="frontend | backend | database | appearance")
    test_case: str = Field(description="The test case or data structure that failed")
    issue_summary: str = Field(description="What went wrong")
    suggested_fix: str = Field(description="What should be changed")
    code_patch: str = Field(default="", description="Suggested code diff or snippet")


# ---------------------------------------------------------------------------
# Aggregate result for a single (task × pattern) evaluation
# ---------------------------------------------------------------------------


class JudgeResult(BaseModel):
    """Complete judge evaluation for one generated application."""

    # --- Traceability ---
    task_id: str
    pattern: str
    difficulty: str = Field(default="unknown")

    # --- Frontend test ---
    frontend_tests: list[FrontendTestScore] = Field(default_factory=list)
    frontend_yes: int = 0
    frontend_partial: int = 0
    frontend_no: int = 0
    frontend_total: int = 0
    frontend_weighted_accuracy: float = Field(
        default=0.0,
        description=(
            "Weighted accuracy: (YES_count + 0.5 * PARTIAL_count) / total"
        ),
    )

    # --- Backend test ---
    backend_tests: list[BackendTestScore] = Field(default_factory=list)
    backend_yes: int = 0
    backend_no: int = 0
    backend_total: int = 0
    backend_accuracy: float = 0.0

    # --- Database test ---
    database_tests: list[DatabaseTestScore] = Field(default_factory=list)
    database_yes: int = 0
    database_no: int = 0
    database_total: int = 0
    database_accuracy: float = 0.0

    # --- Appearance ---
    appearance: AppearanceScore | None = None

    # --- Metadata ---
    judge_model: str = "gpt-4o"
    error: str | None = None

    # --- Token usage ---
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # --- Fix attempts (post-scoring, do not affect scores) ---
    fix_attempts: list[FixAttempt] = Field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Recompute aggregate counts and accuracy from individual scores."""
        # Frontend
        self.frontend_yes = sum(
            1 for t in self.frontend_tests if t.verdict == FrontendVerdict.YES
        )
        self.frontend_partial = sum(
            1 for t in self.frontend_tests if t.verdict == FrontendVerdict.PARTIAL
        )
        self.frontend_no = sum(
            1 for t in self.frontend_tests if t.verdict == FrontendVerdict.NO
        )
        self.frontend_total = len(self.frontend_tests)
        if self.frontend_total > 0:
            self.frontend_weighted_accuracy = round(
                (self.frontend_yes + 0.5 * self.frontend_partial) / self.frontend_total,
                4,
            )

        # Backend
        self.backend_yes = sum(
            1 for t in self.backend_tests if t.verdict == BinaryVerdict.YES
        )
        self.backend_no = sum(
            1 for t in self.backend_tests if t.verdict == BinaryVerdict.NO
        )
        self.backend_total = len(self.backend_tests)
        if self.backend_total > 0:
            self.backend_accuracy = round(
                self.backend_yes / self.backend_total, 4
            )

        # Database
        self.database_yes = sum(
            1 for t in self.database_tests if t.verdict == BinaryVerdict.YES
        )
        self.database_no = sum(
            1 for t in self.database_tests if t.verdict == BinaryVerdict.NO
        )
        self.database_total = len(self.database_tests)
        if self.database_total > 0:
            self.database_accuracy = round(
                self.database_yes / self.database_total, 4
            )


# ---------------------------------------------------------------------------
# Top-level output: per-task across all patterns
# ---------------------------------------------------------------------------


class TaskJudgeResult(BaseModel):
    """All judge results for a single task across patterns."""

    task_id: str
    difficulty: str
    instruction: str = ""
    results: list[JudgeResult] = Field(default_factory=list)
