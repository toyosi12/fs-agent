"""Public orchestration entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import Settings, get_settings
from .context import AgentReport, RunContext
from dotenv import load_dotenv
from .logger import get_logger
from .orchestration import AgentRegistry, CentralizedOrchestrator, DecentralizedOrchestrator, HierarchicalOrchestrator, IterativeRefinementOrchestrator, ParallelOrchestrator, SequentialOrchestrator, register_default_agents
from .llm import BaseLLMClient, build_llm_client
import os

load_dotenv()

logger = get_logger(__name__)

openai_api_key = os.getenv("FS_AGENT_OPENAI_API_KEY")
llm_provider = os.getenv("LLM_PROVIDER")
llm_model = os.getenv("LLM_MODEL")

def _resolve_settings(
    artifact_dir: Path | None,
    dry_run: bool | None,
    orchestration_pattern: str | None = None,
) -> Settings:
    base = get_settings()
    update: dict[str, object] = {}
    if artifact_dir is not None:
        update["artifact_dir"] = artifact_dir
    if dry_run is not None:
        update["dry_run"] = dry_run
    if orchestration_pattern is not None:
        update["orchestration_pattern"] = orchestration_pattern
    if llm_provider is not None:
        update["llm_provider"] = llm_provider
    if llm_model is not None:
        update["llm_model"] = llm_model
    if openai_api_key is not None:
        update["openai_api_key"] = openai_api_key
    return base.model_copy(update=update)


def _build_pattern(
    settings: Settings, registry: AgentRegistry, llm: BaseLLMClient
) -> SequentialOrchestrator | CentralizedOrchestrator | DecentralizedOrchestrator | HierarchicalOrchestrator | ParallelOrchestrator | IterativeRefinementOrchestrator:
    if settings.orchestration_pattern == "sequential":
        return SequentialOrchestrator(registry=registry)
    if settings.orchestration_pattern == "centralized":
        return CentralizedOrchestrator(registry=registry, llm=llm)
    if settings.orchestration_pattern == "decentralized":
        return DecentralizedOrchestrator(registry=registry, llm=llm)
    if settings.orchestration_pattern == "hierarchical":
        return HierarchicalOrchestrator(registry=registry, llm=llm)
    if settings.orchestration_pattern == "parallel":
        return ParallelOrchestrator(registry=registry)
    if settings.orchestration_pattern == "iterative":
        return IterativeRefinementOrchestrator(registry=registry, llm=llm)
    raise ValueError(f"Unsupported orchestration pattern: {settings.orchestration_pattern}")


def run_orchestration(
    user_request: str,
    *,
    artifact_dir: Path | None = None,
    dry_run: bool | None = None,
    orchestration_pattern: str | None = None,
) -> Iterable[AgentReport]:
    """Execute the configured orchestration pattern given a natural language brief."""

    get_settings.cache_clear()

    settings = _resolve_settings(artifact_dir, dry_run=dry_run, orchestration_pattern=orchestration_pattern)
    logger.info(
        "Starting orchestration pattern=%s request=%s artifact_dir=%s dry_run=%s",
        settings.orchestration_pattern,
        user_request[:80].replace("\n", " "),
        artifact_dir or settings.artifact_dir,
        settings.dry_run,
    )
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    llm_client = _build_llm(settings)

    context = RunContext(
        spec=None,
        user_request=user_request,
        settings=settings,
        workspace_dir=Path.cwd(),
        artifact_dir=settings.artifact_dir,
        llm=llm_client,
    )

    registry = AgentRegistry()
    register_default_agents(registry)
    pattern = _build_pattern(settings, registry, llm_client)
    logger.debug("Instantiated pattern %s", pattern.__class__.__name__)
    reports = pattern.run(context)
    logger.info(
        "Completed orchestration (%d stages, %d artifacts)",
        len(context.transcripts),
        len(context.artifacts),
    )
    return reports


def _build_llm(settings: Settings) -> BaseLLMClient:
    try:
        return build_llm_client(
            settings.llm_provider,
            model=settings.llm_model,
            api_key=settings.openai_api_key,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("LLM client creation failed: %s; defaulting to dummy", exc)
        return build_llm_client("dummy", model=settings.llm_model, api_key=None)
