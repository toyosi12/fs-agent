"""Architect agent that transforms a request into a project spec."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..models.spec import ProjectSpec
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class ArchitectAgent(BaseAgent):
    role = AgentRole.ARCHITECT

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        request = context.user_request.strip()
        spec = self._build_spec(context)
        context.update_spec(spec)
        spec_payload = spec.model_dump(mode="json")
        spec_json = json.dumps(spec_payload, indent=2)

        attachments = [
            AgentArtifact(
                name="architect_spec.json",
                kind="doc",
                description="Generated JSON specification",
                body=spec_json,
            )
        ]

        summary = (
            f"Generated spec '{spec.metadata.name}' with "
            f"{len(spec.backend.endpoints)} endpoints, "
            f"{len(spec.frontend.routes)} routes, "
            f"{len(spec.frontend.components)} components"
        )
        result = AgentResult(
            role=self.role,
            summary=summary,
            artifacts={
                "architect_spec": spec_payload,
                "architect_summary": {
                    "request": request,
                    "endpoints": len(spec.backend.endpoints),
                    "routes": len(spec.frontend.routes),
                    "components": len(spec.frontend.components),
                    "data_models": len(spec.backend.data_models),
                },
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(summary)
        return result

    def _build_spec(self, context: RunContext) -> ProjectSpec:
        try:
            return self._spec_from_llm(context)
        except Exception as exc:  # pragma: no cover - LLM dependent
            self.logger.warning("LLM spec generation failed: %s", exc)
            raise RuntimeError("Failed to generate project specification") from exc

    def _spec_from_llm(self, context: RunContext) -> ProjectSpec:
        raw_json = self._generate_spec_json(context)
        data = self._parse_json(raw_json)
        return ProjectSpec.model_validate(data)

    def _generate_spec_json(self, context: RunContext) -> str:
        request = context.user_request.strip() or "Full-stack application"
        schema = ProjectSpec.prompt_schema()
        system = (
            "You are an experienced software architect. Given a product brief you produce a "
            "single JSON specification that exactly conforms to the provided JSON Schema. "
            "The spec must be comprehensive: include ALL backend endpoints required by the "
            "application, ALL frontend routes and reusable components, database models with "
            "migration SQL, and infrastructure targets. Every frontend route.consumes and "
            "component.consumes entry must match a declared backend endpoint as "
            "'METHOD /path'. No endpoint should be orphaned. "
            "The backend and frontend languages must both be set to 'javascript'."
        )
        prompt = (
            f"User brief:\n{request}\n\n"
            "Respond with valid JSON only (no markdown fences, no commentary).\n\n"
            f"JSON Schema to follow:\n{schema}\n\n"
            "Requirements:\n"
            "- Every backend endpoint must have: name, method, path, description, "
            "request_schema (object mapping field names to types/descriptions), "
            "response_schema (same), and errors (list of strings like '400 invalid payload').\n"
            "- Include data_models with fields as {fieldName: typeString} mappings.\n"
            "- If a database is needed, populate backend.database with provider, models "
            "(with table_name, fields, relationships, indexes), and migrations (with name, "
            "up SQL, down SQL).\n"
            "- frontend.routes must reference backend endpoints via consumes.\n"
            "- frontend.components lists reusable UI components with props and consumes.\n"
            "- frontend.theme must be a flat object where every key AND value is a plain "
            "- infra.targets must be objects with name, environment (dev|staging|test|prod), "
            "description, and runtime (docker|serverless|kubernetes).\n"
            "- Include at least two infra targets (dev and prod).\n"
        )
        # Allow a dedicated provider/model for the architect via
        # FS_AGENT_LLM_PROVIDER_ARCHITECT / FS_AGENT_LLM_MODEL_ARCHITECT.
        response = context.get_llm("architect").generate(
            prompt,
            system=system,
            temperature=0.2,
        )
        return self._strip_code_fences(response).strip()

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(text).strip()
        if not cleaned:
            raise ValueError("Architect LLM returned empty response")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Architect LLM returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Architect LLM response was not a JSON object")
        return data

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            parts = stripped.split("```", 2)
            if len(parts) >= 2:
                cleaned = parts[1]
                for prefix in ("json\n", "yaml\n", "yml\n"):
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):]
                        break
                return cleaned.strip()
        return text