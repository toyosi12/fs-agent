"""Runtime configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Top-level runtime configuration."""

    orchestration_pattern: Literal["sequential", "centralized", "decentralized", "hierarchical", "parallel"] = "sequential"
    artifact_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    dry_run: bool = False
    max_validation_retries: int = Field(
        default_factory=lambda: int(os.getenv("FS_AGENT_MAX_VALIDATION_RETRIES", "3"))
    )
    llm_provider: Literal["dummy", "openai", "qwen", "ollama", "auto"] = Field(
        default_factory=lambda: os.getenv("FS_AGENT_LLM_PROVIDER", "dummy")
    )
    llm_model: str = Field(default_factory=lambda: os.getenv("FS_AGENT_LLM_MODEL", "gpt-4o-mini"))
    llm_base_url: str | None = Field(default_factory=lambda: os.getenv("FS_AGENT_LLM_BASE_URL"))
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("FS_AGENT_OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
