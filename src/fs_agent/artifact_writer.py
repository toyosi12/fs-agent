"""Utilities for persisting agent outputs to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents.base import AgentArtifact, AgentResult
from .logger import get_logger

logger = get_logger(__name__)


def persist_agent_output(result: AgentResult, artifact_dir: Path) -> dict[str, list[str]]:
    """Persist artifacts and attachments emitted by an agent.

    Returns a mapping of saved artifact and attachment file paths for later reporting.
    """

    artifact_dir.mkdir(parents=True, exist_ok=True)
    saved_artifacts: list[str] = []
    saved_attachments: list[str] = []

    role_slug = _role_slug(result.role)

    for key, payload in result.artifacts.items():
        path = artifact_dir / f"{role_slug}_{key}.json"
        _write_json(path, payload)
        saved_artifacts.append(str(path))
        logger.debug("Saved artifact %s", path)

    for attachment in result.attachments:
        path = artifact_dir / attachment.name
        _write_attachment(path, attachment)
        saved_attachments.append(str(path))
        logger.debug("Saved attachment %s", path)

    logger.info(
        "Persisted %d artifacts and %d attachments for %s",
        len(saved_artifacts),
        len(saved_attachments),
        role_slug,
    )
    return {"artifacts": saved_artifacts, "attachments": saved_attachments}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _write_attachment(path: Path, attachment: AgentArtifact) -> None:
    body = attachment.body
    if isinstance(body, bytes):
        path.write_bytes(body)
        return
    if isinstance(body, (dict, list)):
        _write_json(path, body)
        return
    path.write_text(str(body))


def _role_slug(role: Any) -> str:
    if hasattr(role, "value"):
        return str(role.value)
    return str(role)
