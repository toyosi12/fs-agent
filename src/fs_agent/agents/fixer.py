"""Fixer agent — analyzes infra errors and patches generated code."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..context import RunContext
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class FixerAgent(BaseAgent):
    role = AgentRole.FIXER

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)

        # Collect infra diagnostics from the most recent infra report
        infra_report = self._latest_infra_report(context)
        diagnostics = infra_report.get("diagnostics", []) if infra_report else []
        infra_log = infra_report.get("infra_log", "") if infra_report else ""

        # Read generated source files from disk
        backend_code = self._read_project_code(context, "backend")
        frontend_code = self._read_project_code(context, "frontend")

        # Ask the LLM to analyze and produce patches
        patches = self._generate_patches(
            context, diagnostics, infra_log, backend_code, frontend_code,
        )

        # Apply patches to files
        applied: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []
        for patch in patches:
            try:
                self._apply_patch(context, patch)
                applied.append(patch)
            except Exception as exc:
                patch["error"] = str(exc)
                failed.append(patch)

        summary = (
            f"Fixer applied {len(applied)} patches, "
            f"{len(failed)} failed, "
            f"from {len(diagnostics)} diagnostics"
        )
        self.logger.info(summary)

        return AgentResult(
            role=self.role,
            summary=summary,
            artifacts={
                "fixer_diagnostics_input": diagnostics,
                "fixer_patches_applied": applied,
                "fixer_patches_failed": failed,
                "fixer_patch_count": len(applied),
            },
            attachments=[
                AgentArtifact(
                    name="fixer_report.json",
                    kind="doc",
                    description="Fixer agent patch report",
                    body=json.dumps({
                        "diagnostics": diagnostics,
                        "applied": applied,
                        "failed": failed,
                    }, indent=2),
                )
            ],
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )

    def _latest_infra_report(self, context: RunContext) -> dict[str, Any] | None:
        """Find the most recent infra agent report from transcripts."""
        for report in reversed(context.transcripts):
            if report.role == "infra":
                return {
                    "diagnostics": report.artifacts.get("diagnostics", []),
                    "infra_log": report.artifacts.get("infra_log", ""),
                    "status": report.status,
                }
        return None

    def _read_project_code(self, context: RunContext, component: str) -> str:
        """Read generated source files for a component (backend/frontend)."""
        try:
            comp_dir = context.projects_dir / component
        except RuntimeError:
            return ""
        if not comp_dir.exists():
            return ""

        parts: list[str] = []
        extensions = {".js", ".jsx", ".ts", ".tsx", ".json", ".sql", ".env"}
        for p in sorted(comp_dir.rglob("*")):
            if p.is_file() and p.suffix in extensions and "node_modules" not in str(p):
                rel = p.relative_to(comp_dir)
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    # Limit per-file to avoid prompt explosion
                    if len(content) > 4000:
                        content = content[:4000] + "\n... (truncated)"
                    parts.append(f"// === FILE: {rel} ===\n{content}")
                except Exception:
                    pass
        return "\n\n".join(parts)

    def _generate_patches(
        self,
        context: RunContext,
        diagnostics: list[str],
        infra_log: str,
        backend_code: str,
        frontend_code: str,
    ) -> list[dict[str, str]]:
        """Ask the LLM to produce file patches for the reported errors."""
        if not diagnostics:
            return []

        diag_text = "\n".join(f"- {d}" for d in diagnostics)
        system = (
            "You are an expert full-stack debugger. You are given:\n"
            "1. Error diagnostics from running npm install / npm run migrate / npm run dev\n"
            "2. The generated backend and frontend source code\n\n"
            "Analyze each error and produce JSON patches to fix them.\n"
            "Respond with a JSON array only (no markdown fences):\n"
            "[\n"
            '  {"file": "backend/src/app.js", "action": "replace", '
            '"description": "Fix missing import", '
            '"search": "exact text to find", "replace": "replacement text"},\n'
            '  {"file": "backend/package.json", "action": "replace", '
            '"description": "Add missing dependency", '
            '"search": "exact text", "replace": "replacement text"},\n'
            "  ...\n"
            "]\n\n"
            "Rules:\n"
            "- Each patch targets one file with an exact search-and-replace.\n"
            "- The 'search' field must match text that EXISTS in the file.\n"
            '- Use "action": "create" with "contents" instead of search/replace '
            "for entirely new files.\n"
            "- Focus on fixing the actual errors — don't refactor unrelated code.\n"
            "- If an error cannot be fixed, skip it.\n"
        )
        prompt = (
            f"## Errors from infrastructure bootstrap\n{diag_text}\n\n"
            f"## Infrastructure log\n{infra_log[:3000]}\n\n"
        )
        if backend_code:
            # Limit to avoid exceeding context
            prompt += f"## Backend source code\n{backend_code[:15000]}\n\n"
        if frontend_code:
            prompt += f"## Frontend source code\n{frontend_code[:15000]}\n\n"
        prompt += "Produce a JSON array of patches to fix the errors above."

        raw = context.get_llm("fixer").generate(prompt, system=system, temperature=0.1)
        return self._parse_patches(raw)

    def _parse_patches(self, raw: str) -> list[dict[str, str]]:
        """Parse the LLM response into a list of patch dicts."""
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # skip opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON array in the text
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return []
            else:
                return []
        if not isinstance(data, list):
            return []
        return [p for p in data if isinstance(p, dict) and "file" in p]

    def _apply_patch(self, context: RunContext, patch: dict[str, str]) -> None:
        """Apply a single patch to the project files."""
        file_rel = patch.get("file", "")
        target = context.projects_dir / file_rel
        action = patch.get("action", "replace")

        if action == "create":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch.get("contents", ""), encoding="utf-8")
            return

        if not target.exists():
            raise FileNotFoundError(f"Patch target not found: {file_rel}")

        content = target.read_text(encoding="utf-8")
        search = patch.get("search", "")
        replace = patch.get("replace", "")
        if not search:
            raise ValueError("Patch has empty search string")
        if search not in content:
            raise ValueError(f"Search text not found in {file_rel}")

        new_content = content.replace(search, replace, 1)
        target.write_text(new_content, encoding="utf-8")
