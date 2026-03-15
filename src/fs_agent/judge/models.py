"""Pydantic models for judge scoring results.

These models capture the FullStack-Bench rubric scores and link each
evaluation back to the originating task and its difficulty level.
"""

from __future__ import annotations

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
