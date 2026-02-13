"""Load project specifications from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models.spec import ProjectSpec


def load_spec(path: Path) -> ProjectSpec:
    """Parse a YAML specification file into a ProjectSpec."""

    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")

    try:
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - pass details upstream
        raise ValueError(f"Failed to parse spec file: {exc}") from exc

    try:
        return ProjectSpec.model_validate(data)
    except Exception as exc:  # pragma: no cover - validation details vary
        raise ValueError(f"Invalid spec content: {exc}") from exc
