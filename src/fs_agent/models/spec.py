"""pydantic models describing the project specification."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


# ---------------------------------------------------------------------------
# Backend models
# ---------------------------------------------------------------------------


class ApiEndpoint(BaseModel):
    name: str
    method: HttpMethod
    path: str
    description: str
    auth_required: bool = False
    websocket: bool = False
    request_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class DataModel(BaseModel):
    """Application-level data model (ORM entity / table)."""

    name: str
    description: str | None = None
    table_name: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    relationships: list[str] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=list)


class Migration(BaseModel):
    """Database migration step."""

    name: str
    up: str = ""
    down: str = ""
    depends_on: list[str] = Field(default_factory=list)


class DatabaseSpec(BaseModel):
    provider: Literal["postgres", "mysql", "sqlite", "mongodb"] = "sqlite"
    models: list[DataModel] = Field(default_factory=list)
    migrations: list[Migration] = Field(default_factory=list)


class BackendSpec(BaseModel):
    language: Literal["javascript", "typescript"] = "javascript"
    framework: str = "express"
    style: Literal["rest", "graphql"] = "rest"
    endpoints: list[ApiEndpoint] = Field(default_factory=list)
    data_models: list[DataModel] = Field(default_factory=list)
    database: DatabaseSpec | None = None


# ---------------------------------------------------------------------------
# Frontend models
# ---------------------------------------------------------------------------


class FrontendComponent(BaseModel):
    """Discrete UI component."""

    name: str
    description: str = ""
    props: dict[str, str] = Field(default_factory=dict)
    consumes: list[str] = Field(default_factory=list)


class FrontendRoute(BaseModel):
    path: str
    description: str
    auth_required: bool = False
    layout: str | None = None
    consumes: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)


class FrontendSpec(BaseModel):
    language: Literal["javascript", "typescript"] = "javascript"
    framework: str = "react"
    styling: Literal["tailwind", "css-modules", "chakra"] = "tailwind"
    routes: list[FrontendRoute] = Field(default_factory=list)
    components: list[FrontendComponent] = Field(default_factory=list)
    theme: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Infra models
# ---------------------------------------------------------------------------


class InfraTarget(BaseModel):
    name: str
    environment: Literal["dev", "staging", "test", "prod"]
    description: str
    runtime: Literal["docker", "serverless", "kubernetes"] = "docker"


class InfraSpec(BaseModel):
    ci: str = "github-actions"
    cd: str = "fly-io"
    targets: list[InfraTarget] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class ProjectMetadata(BaseModel):
    name: str
    summary: str
    owner: str
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------


class ProjectSpec(BaseModel):
    metadata: ProjectMetadata
    backend: BackendSpec
    frontend: FrontendSpec
    infra: InfraSpec

    @model_validator(mode="after")
    def _cross_reference_check(self) -> "ProjectSpec":
        """Warn (don't reject) when frontend consumes references
        don't match any declared backend endpoint."""

        declared: set[str] = set()
        for ep in self.backend.endpoints:
            declared.add(f"{ep.method.value} {ep.path}")

        consumed: set[str] = set()
        for route in self.frontend.routes:
            consumed.update(route.consumes)
        for comp in self.frontend.components:
            consumed.update(comp.consumes)

        missing = consumed - declared
        if missing:
            logger.warning(
                "Frontend references endpoints not declared in backend: %s",
                ", ".join(sorted(missing)),
            )

        orphaned = declared - consumed
        if orphaned:
            logger.warning(
                "Backend endpoints not consumed by any frontend route/component: %s",
                ", ".join(sorted(orphaned)),
            )

        return self

    @classmethod
    def prompt_schema(cls) -> str:
        """Return the JSON Schema string suitable for LLM prompts."""
        return json.dumps(cls.model_json_schema(), indent=2)
