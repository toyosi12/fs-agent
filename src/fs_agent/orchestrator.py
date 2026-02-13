"""Public orchestration entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import Settings, get_settings
from .context import AgentReport, RunContext
from .logger import get_logger
from .orchestration import AgentRegistry, SequentialOrchestrator, register_default_agents
from .spec_loader import load_spec


logger = get_logger(__name__)


def _resolve_settings(artifact_dir: Path | None, dry_run: bool | None) -> Settings:
    base = get_settings()
    update: dict[str, object] = {}
    if artifact_dir is not None:
        update["artifact_dir"] = artifact_dir
    if dry_run is not None:
        update["dry_run"] = dry_run
    return base.model_copy(update=update)


def _build_pattern(settings: Settings, registry: AgentRegistry) -> SequentialOrchestrator:
    if settings.orchestration_pattern == "sequential":
        return SequentialOrchestrator(registry=registry)
    raise ValueError(f"Unsupported orchestration pattern: {settings.orchestration_pattern}")


def run_orchestration(
    spec_path: Path,
    *,
    artifact_dir: Path | None = None,
    dry_run: bool | None = None,
) -> Iterable[AgentReport]:
    """Load the spec and execute the configured orchestration pattern."""

    settings = _resolve_settings(artifact_dir, dry_run)
    logger.info(
        "Starting orchestration pattern=%s spec=%s artifact_dir=%s dry_run=%s",
        settings.orchestration_pattern,
        spec_path,
        artifact_dir or settings.artifact_dir,
        settings.dry_run,
    )
    spec = load_spec(spec_path)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    context = RunContext(
        spec=spec,
        settings=settings,
        workspace_dir=Path.cwd(),
        artifact_dir=settings.artifact_dir,
    )

    registry = AgentRegistry()
    register_default_agents(registry)
    pattern = _build_pattern(settings, registry)
    logger.debug("Instantiated pattern %s", pattern.__class__.__name__)
    reports = pattern.run(context)
    logger.info(
        "Completed orchestration (%d stages, %d artifacts)",
        len(context.transcripts),
        len(context.artifacts),
    )
    return reports
