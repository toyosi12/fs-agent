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

    _MAX_SPEC_RETRIES = 3

    def _build_spec(self, context: RunContext) -> ProjectSpec:
        last_error: Exception | None = None
        for attempt in range(1, self._MAX_SPEC_RETRIES + 1):
            try:
                return self._spec_from_llm(context, attempt=attempt, last_error=last_error)
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "LLM spec generation failed (attempt %d/%d): %s",
                    attempt, self._MAX_SPEC_RETRIES, exc,
                )
        raise RuntimeError("Failed to generate project specification") from last_error

    def _spec_from_llm(
        self,
        context: RunContext,
        *,
        attempt: int = 1,
        last_error: Exception | None = None,
    ) -> ProjectSpec:
        raw_json = self._generate_spec_json(context, attempt=attempt, last_error=last_error)
        data = self._parse_json(raw_json)
        return ProjectSpec.model_validate(data)

    def _generate_spec_json(
        self,
        context: RunContext,
        *,
        attempt: int = 1,
        last_error: Exception | None = None,
    ) -> str:
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
            "response_schema (the exact JSON response body shape the endpoint returns, "
            "including any wrapper like {success, data, pagination} — use concrete field "
            "names and types so the frontend can consume them directly), "
            "and errors (list of strings like '400 invalid payload').\n"
            "- Include data_models with fields as {fieldName: typeString} mappings.\n"
            "- If a database is needed, populate backend.database with provider (use 'sqlite'), models "
            "(with table_name, fields, relationships, indexes), and migrations (with name, "
            "up SQL using SQLite syntax, down SQL).\n"
            "- frontend.routes must reference backend endpoints via consumes.\n"
            "- frontend.components lists reusable UI components with props and consumes.\n"
            "- frontend.theme can capture color/styling tokens from the brief.\n"
            "- infra.targets must be objects with name, environment (dev|staging|test|prod), "
            "description, and runtime (docker|serverless|kubernetes).\n"
            "- Include at least two infra targets (dev and prod), both using 'docker' runtime.\n"
        )
        if attempt > 1 and last_error:
            prompt += (
                f"\n\nIMPORTANT: Your previous response had a JSON error:\n"
                f"  {last_error}\n"
                "Please fix this and return ONLY valid, parseable JSON.\n"
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
        # First try parsing as-is
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting the outermost JSON object from the response
            data = self._extract_json_object(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Architect LLM response was not a JSON object")
        return data

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Find and parse the outermost {...} JSON object in a text blob."""
        start = text.find("{")
        if start == -1:
            raise ValueError("Architect LLM returned no JSON object")
        # Walk forward to find the matching closing brace
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Architect LLM returned invalid JSON: {exc}"
                        ) from exc
        raise ValueError("Architect LLM returned unterminated JSON object")

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