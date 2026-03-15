"""Prompt templates for the LLM judge.

Each function builds a (system, user) prompt pair for one scoring
dimension.  The judge LLM is expected to return structured JSON.
"""

from __future__ import annotations

import json


def frontend_test_prompt(
    task_instruction: str,
    test_case: dict,
    frontend_code: str,
) -> tuple[str, str]:
    """Build prompts for evaluating a single frontend test case."""
    system = (
        "You are an expert frontend QA evaluator. You are given the original "
        "task description, a specific UI test case with its expected result, "
        "and the generated frontend source code. You must determine whether "
        "the generated code correctly implements the functionality described "
        "in the test case.\n\n"
        "Respond with JSON only (no markdown fences):\n"
        '{"verdict": "YES" | "PARTIAL" | "NO", "reasoning": "..."}\n\n'
        "Verdict meanings:\n"
        "- YES: The functionality is fully implemented and the expected result "
        "would be achievable.\n"
        "- PARTIAL: Some elements of the feature work but not the full expected result.\n"
        "- NO: The feature is missing, broken, or would not produce the expected result."
    )
    user = (
        f"## Task Description\n{task_instruction}\n\n"
        f"## UI Test Case\n"
        f"**Task:** {test_case['task']}\n"
        f"**Expected Result:** {test_case['expected_result']}\n\n"
        f"## Generated Frontend Code\n```\n{_truncate(frontend_code, 12000)}\n```\n\n"
        "Evaluate whether this code implements the test case. "
        "Respond with JSON only."
    )
    return system, user


def backend_test_prompt(
    task_instruction: str,
    test_case: dict,
    backend_code: str,
) -> tuple[str, str]:
    """Build prompts for evaluating a single backend test case."""
    system = (
        "You are an expert backend QA evaluator. You are given the original "
        "task description, a specific backend test case with its expected result, "
        "and the generated backend source code. You must determine whether "
        "the generated code correctly implements the API behavior described "
        "in the test case.\n\n"
        "Respond with JSON only (no markdown fences):\n"
        '{"verdict": "YES" | "NO", "reasoning": "..."}\n\n'
        "Verdict meanings:\n"
        "- YES: The API endpoint is implemented and would return the expected response.\n"
        "- NO: The endpoint is missing, incorrectly implemented, or would not "
        "produce the expected result."
    )
    user = (
        f"## Task Description\n{task_instruction}\n\n"
        f"## Backend Test Case\n"
        f"**Instruction:** {test_case['instruction']}\n"
        f"**Expected Result:** {test_case['expected_result']}\n\n"
        f"## Generated Backend Code\n```\n{_truncate(backend_code, 12000)}\n```\n\n"
        "Evaluate whether this code implements the test case. "
        "Respond with JSON only."
    )
    return system, user


def database_test_prompt(
    task_instruction: str,
    data_structure: str,
    migration_sql: str,
) -> tuple[str, str]:
    """Build prompts for evaluating a single data structure requirement."""
    system = (
        "You are an expert database engineer evaluator. You are given the "
        "original task description, a required data structure name, and the "
        "generated database migration SQL. You must determine whether the "
        "migrations correctly define a table/schema for the required data "
        "structure.\n\n"
        "Respond with JSON only (no markdown fences):\n"
        '{"verdict": "YES" | "NO", "reasoning": "..."}\n\n'
        "Verdict meanings:\n"
        "- YES: The migration SQL creates an appropriate table/schema for this "
        "data structure with reasonable columns and types.\n"
        "- NO: The data structure is not represented in the migrations, or the "
        "schema is fundamentally wrong."
    )
    user = (
        f"## Task Description\n{task_instruction}\n\n"
        f"## Required Data Structure\n\"{data_structure}\"\n\n"
        f"## Generated Migration SQL\n```sql\n{_truncate(migration_sql, 8000)}\n```\n\n"
        "Evaluate whether these migrations correctly define this data structure. "
        "Respond with JSON only."
    )
    return system, user


def appearance_prompt(
    task_instruction: str,
    frontend_code: str,
) -> tuple[str, str]:
    """Build prompts for evaluating the visual appearance of the frontend."""
    system = (
        "You are an expert UI/UX designer evaluator. You are given the "
        "original task description and the generated frontend source code "
        "(React/JSX with Tailwind CSS classes). Evaluate the visual quality "
        "of the generated UI code on four criteria, each scored 1-5.\n\n"
        "Respond with JSON only (no markdown fences):\n"
        "{\n"
        '  "layout": <1-5>,\n'
        '  "color": <1-5>,\n'
        '  "typography": <1-5>,\n'
        '  "component_polish": <1-5>,\n'
        '  "reasoning": "..."\n'
        "}\n\n"
        "Criteria:\n"
        "1. **Layout & structure** (1-5): Proper spacing, alignment, responsive "
        "design patterns, logical component arrangement.\n"
        "2. **Color & theming** (1-5): Matches specification colors, consistent "
        "palette, good contrast, professional look.\n"
        "3. **Typography & readability** (1-5): Proper font sizing, heading "
        "hierarchy, adequate contrast, readable text.\n"
        "4. **Component polish** (1-5): Buttons, forms, cards, lists look "
        "professional; hover states, transitions, proper styling.\n\n"
        "A score of 1 means unusable/missing, 3 means acceptable, 5 means "
        "excellent/production-ready."
    )
    user = (
        f"## Task Description\n{task_instruction}\n\n"
        f"## Generated Frontend Code\n```\n{_truncate(frontend_code, 15000)}\n```\n\n"
        "Evaluate the visual quality of this frontend code. "
        "Respond with JSON only."
    )
    return system, user


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to stay within token budget."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"
