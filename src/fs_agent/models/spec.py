"""pydantic models describing the project specification."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ApiEndpoint(BaseModel):
    name: str
    method: HttpMethod
    path: str
    description: str
    request_schema: dict[str, str] | None = None
    response_schema: dict[str, str] | None = None
    errors: list[str] = Field(default_factory=list)


class DataModel(BaseModel):
    name: str
    description: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)


class FrontendRoute(BaseModel):
    path: str
    description: str
    consumes: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)


class BackendSpec(BaseModel):
    language: Literal["javascript", "typescript"] = "typescript"
    framework: str = "express"
    style: Literal["rest", "graphql"] = "rest"
    endpoints: list[ApiEndpoint] = Field(default_factory=list)
    data_models: list[DataModel] = Field(default_factory=list)


class FrontendSpec(BaseModel):
    language: Literal["javascript", "typescript"] = "typescript"
    framework: str = "react"
    styling: Literal["tailwind", "css-modules", "chakra"] = "tailwind"
    routes: list[FrontendRoute] = Field(default_factory=list)


class InfraTarget(BaseModel):
    name: str
    environment: Literal["dev", "staging", "prod"]
    description: str
    runtime: Literal["docker", "serverless", "kubernetes"] = "docker"


class InfraSpec(BaseModel):
    ci: str = "github-actions"
    cd: str = "fly-io"
    targets: list[InfraTarget] = Field(default_factory=list)


class ProjectMetadata(BaseModel):
    name: str
    summary: str
    owner: str
    version: str = "0.1.0"


class ProjectSpec(BaseModel):
    metadata: ProjectMetadata
    backend: BackendSpec
    frontend: FrontendSpec
    infra: InfraSpec
