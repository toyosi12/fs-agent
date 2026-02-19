"""Lightweight wrapper around python-dotenv for testability."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_env_file(path: Path | None = None, *, override: bool = False) -> Path | None:
    """Load variables from the provided .env path (defaults to workspace root)."""

    loaded = load_dotenv(dotenv_path=path, override=override)
    if not loaded:
        return None
    if path is not None:
        return path
    default_path = Path.cwd() / ".env"
    return default_path if default_path.exists() else None
