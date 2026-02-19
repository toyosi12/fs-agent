"""Helpers for interacting with the file-system reference MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class MCPApplicationResult:
    """Result of applying an MCP filesystem plan."""

    project_path: Path
    created_files: list[Path]


def apply_filesystem_plan(
    plan: dict[str, Any],
    base_dir: Path,
    *,
    dry_run: bool = False,
) -> MCPApplicationResult:
    """Materialize a plan produced for the file-system reference MCP server."""

    tool = (plan or {}).get("tool")
    if tool != "mcp.fs":
        raise ValueError(f"Unsupported MCP tool '{tool}'. Expected 'mcp.fs'.")

    project_root = plan.get("project_root")
    if not project_root:
        raise ValueError("MCP plan missing 'project_root'.")

    files = plan.get("files") or []
    if not isinstance(files, list) or not files:
        raise ValueError("MCP plan must include a non-empty 'files' array.")

    directories = plan.get("directories") or []
    if directories and not isinstance(directories, list):
        raise ValueError("'directories' must be a list if provided.")

    base_dir.mkdir(parents=True, exist_ok=True)
    project_path = base_dir / project_root
    created_files: list[Path] = []

    if not dry_run:
        project_path.mkdir(parents=True, exist_ok=True)

    for directory in directories:
        directory_path = project_path / Path(directory)
        if not dry_run:
            directory_path.mkdir(parents=True, exist_ok=True)

    for file_entry in files:
        relative = file_entry.get("path")
        contents = file_entry.get("contents", "")
        if not relative:
            raise ValueError("Each MCP file entry must include a 'path'.")
        file_path = project_path / Path(relative)
        created_files.append(file_path)
        if dry_run:
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(contents)
        logger.debug("Wrote MCP file %s", file_path)

    return MCPApplicationResult(project_path=project_path, created_files=created_files)
