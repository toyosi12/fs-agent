"""Runtime configuration helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Top-level runtime configuration."""

    orchestration_pattern: Literal["sequential"] = "sequential"
    artifact_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    dry_run: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
