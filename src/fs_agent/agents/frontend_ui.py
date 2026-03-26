"""Frontend UI agent — specializes in reusable components, styling, and state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..mcp import apply_filesystem_plan
from ..models.spec import ProjectSpec
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class FrontendUiAgent(BaseAgent):
    """Generates reusable UI components, shared styling, and state management.

    Used by the hierarchical pattern's 3-level topology.  Produces
    components in ``frontend/src/components/``.  Assumes the page-level
    agent has already created the App shell and routing.
    """

    role = AgentRole.FRONTEND_UI

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        frontend = spec.frontend

        component_defs = [comp.model_dump() for comp in frontend.components]

        # Collect all component names referenced by routes
        route_components: list[str] = []
        for route in frontend.routes:
            route_components.extend(route.components)

        ui_code = self._generate_ui_code(context, component_defs, route_components)
        mcp_plan = self._generate_ui_plan(context, spec, component_defs, ui_code)

        projects_root = context.projects_dir
        application = apply_filesystem_plan(
            mcp_plan, projects_root, dry_run=context.settings.dry_run,
        )
        created_files = [str(path) for path in application.created_files]
        mcp_plan = {**mcp_plan, "project_path": str(application.project_path)}

        attachments = [
            AgentArtifact(
                name="frontend_components.jsx",
                kind="code",
                description="Reusable UI components",
                body=ui_code,
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=(
                f"Generated {len(component_defs) or len(route_components)} "
                f"reusable UI components with Tailwind styling."
            ),
            artifacts={
                "frontend_ui_blueprint": {
                    "components": component_defs,
                    "theme": frontend.theme,
                },
                "frontend_ui_source": {"language": "javascript", "body": ui_code},
                "frontend_ui_mcp_plan": mcp_plan,
                "frontend_ui_files": created_files,
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_ui_code(
        self,
        context: RunContext,
        component_defs: list[dict[str, Any]],
        route_components: list[str],
    ) -> str:
        metadata = context.require_spec().metadata
        frontend = context.require_spec().frontend

        comp_lines = []
        for comp in component_defs:
            comp_lines.append(
                f"- {comp.get('name', '?')}: {comp.get('description', '')}"
            )
        # Also include component names from routes that aren't in component_defs
        defined_names = {c.get("name") for c in component_defs}
        for name in route_components:
            if name not in defined_names:
                comp_lines.append(f"- {name}: (referenced in routes, no definition)")

        theme_info = ""
        if frontend.theme:
            theme_info = f"\nTheme: {json.dumps(frontend.theme)}\n"

        prompt = (
            f"Project: {metadata.name} ({metadata.summary})\n\n"
            "Generate REUSABLE UI COMPONENTS for a React application.\n"
            f"Styling: Tailwind CSS{theme_info}\n"
            "Components needed:\n" + ("\n".join(comp_lines) or "- Common UI components") + "\n\n"
            "Focus on:\n"
            "- Reusable, composable components (buttons, cards, forms, modals, tables)\n"
            "- Consistent Tailwind styling with the project theme\n"
            "- Proper prop interfaces and default values\n"
            "- Loading states, error states, empty states\n"
            "- Accessibility basics (aria labels, semantic HTML)\n\n"
            "Each component should be separated with a comment marker:\n"
            "// === COMPONENT: ComponentName ===\n\n"
            "Do NOT generate page-level components or routing — those exist already.\n"
            "Do NOT use TypeScript. Do NOT leave TODO stubs.\n"
        )

        upstream = context.extra_context.get("upstream_context", "")
        if upstream:
            prompt += (
                "\n--- UPSTREAM CONTEXT ---\n"
                f"{upstream}\n"
                "--- END UPSTREAM CONTEXT ---\n"
            )

        system = (
            "You are a senior UI engineer specializing in component libraries. "
            "Generate complete, reusable React components in plain JavaScript (JSX) "
            "with Tailwind CSS. Focus on composition, consistency, and reusability. "
            "Each component must be fully functional with proper props handling. "
            "Use lucide-react for icons. No placeholders, no TODOs, no TypeScript."
        )

        try:
            return context.get_llm("frontend_ui").generate(
                prompt, system=system, temperature=0.2,
            )
        except Exception:
            return self._fallback_ui(component_defs, route_components)

    def _fallback_ui(
        self,
        component_defs: list[dict[str, Any]],
        route_components: list[str],
    ) -> str:
        lines = ["import React from 'react';", ""]
        names = [c.get("name", "Component") for c in component_defs]
        # Add route components not in defs
        defined = set(names)
        for rc in route_components:
            if rc not in defined:
                names.append(rc)
        for name in names:
            lines.append(
                f"export function {name}({{ children, className = '' }}) {{\n"
                f"  return <div className={{`p-4 ${{className}}`}}>{{children}}</div>;\n"
                "}\n"
            )
        return "\n".join(lines)

    def _generate_ui_plan(
        self,
        context: RunContext,
        spec: ProjectSpec,
        component_defs: list[dict[str, Any]],
        ui_code: str,
    ) -> dict[str, Any]:
        """Generate MCP plan for component files only."""
        system = (
            "You are a senior frontend engineer. Produce a JSON plan for "
            "reusable component files in a React project. Place components "
            "in src/components/*.jsx. Also include test files in "
            "src/__tests__/components/. "
            "Do NOT include App.jsx, pages, routing, package.json, or Dockerfile — "
            "those are handled by a separate agent."
        )
        prompt = (
            f"User request: {context.user_request}\n"
            f"Components: {json.dumps(component_defs, indent=2)}\n\n"
            f"Component code:\n<components>\n{ui_code}\n</components>\n\n"
            "Split each component into its own file under src/components/.\n"
            "Return JSON:\n"
            '{"tool": "mcp.fs", "project_root": "frontend", '
            '"files": [{"path": "...", "description": "...", "contents": "..."}]}'
        )
        try:
            raw = context.get_llm("frontend_ui").generate(
                prompt, system=system, temperature=0.2,
            )
            text = raw.strip()
            if text.startswith("```"):
                parts = text.split("```", 2)
                if len(parts) >= 2:
                    text = parts[1]
                    if text.lstrip().startswith("json"):
                        text = text.lstrip()[4:]
            plan = json.loads(text.strip())
            plan.setdefault("tool", "mcp.fs")
            plan.setdefault("project_root", "frontend")
            if isinstance(plan.get("files"), list) and plan["files"]:
                return plan
        except Exception as exc:
            self.logger.warning("UI plan generation failed: %s", exc)

        return {
            "tool": "mcp.fs",
            "project_root": "frontend",
            "instructions": f"UI components for {spec.metadata.name}",
            "files": [
                {
                    "path": "src/components/index.jsx",
                    "description": "All UI components",
                    "contents": ui_code,
                },
            ],
        }
