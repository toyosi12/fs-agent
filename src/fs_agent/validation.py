"""Post-generation validation for full-stack projects.

Runs structural and integration checks on generated project files:
- Required files exist (package.json, entry points, Dockerfiles)
- No LLM markdown fences left in source files
- package.json is valid JSON with required fields
- Frontend ↔ Backend API wiring (proxy / base URL consistency)
- Database config sync (backend .env DB_PATH, db.js setup)
- Docker build feasibility (devDeps available for build step)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation problem found in the generated project."""

    component: str  # "backend", "frontend", "infra", "integration"
    severity: str  # "error" or "warning"
    message: str
    file: str = ""  # relative path within project


@dataclass
class ValidationResult:
    """Aggregate result of all validation checks."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def errors_for_component(self, component: str) -> list[ValidationIssue]:
        return [i for i in self.issues if i.component == component and i.severity == "error"]

    def summary(self) -> str:
        if self.passed:
            return f"Validation passed ({self.warning_count} warnings)"
        return (
            f"Validation failed: {self.error_count} errors, "
            f"{self.warning_count} warnings"
        )

    def feedback_prompt(self) -> str:
        """Build a prompt snippet describing all errors for agent re-generation."""
        if self.passed:
            return ""
        lines = ["The generated project has the following validation errors that must be fixed:\n"]
        for issue in self.issues:
            if issue.severity == "error":
                loc = f" ({issue.file})" if issue.file else ""
                lines.append(f"- [{issue.component}]{loc}: {issue.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

_MARKDOWN_FENCE_RE = re.compile(r"^```[\w]*\s*$", re.MULTILINE)


def _check_markdown_fences(project_dir: Path, component: str) -> list[ValidationIssue]:
    """Detect leftover LLM markdown fences (```js, ```) in source files."""
    issues: list[ValidationIssue] = []
    src_dir = project_dir / component / "src"
    if not src_dir.exists():
        return issues
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css", ".sql"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _MARKDOWN_FENCE_RE.search(text):
            rel = str(path.relative_to(project_dir))
            issues.append(ValidationIssue(
                component=component,
                severity="error",
                message="File contains LLM markdown fence (```) — will cause syntax errors",
                file=rel,
            ))
    return issues


def _check_package_json(project_dir: Path, component: str) -> list[ValidationIssue]:
    """Verify package.json exists and is valid JSON with required fields."""
    issues: list[ValidationIssue] = []
    pkg_path = project_dir / component / "package.json"
    if not pkg_path.exists():
        issues.append(ValidationIssue(
            component=component,
            severity="error",
            message="Missing package.json",
            file=f"{component}/package.json",
        ))
        return issues
    try:
        data = json.loads(pkg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(ValidationIssue(
            component=component,
            severity="error",
            message=f"Invalid package.json: {exc}",
            file=f"{component}/package.json",
        ))
        return issues
    if not data.get("name"):
        issues.append(ValidationIssue(
            component=component,
            severity="warning",
            message="package.json missing 'name' field",
            file=f"{component}/package.json",
        ))
    if not data.get("scripts"):
        issues.append(ValidationIssue(
            component=component,
            severity="error",
            message="package.json missing 'scripts' field",
            file=f"{component}/package.json",
        ))
    return issues


def _check_required_files(project_dir: Path) -> list[ValidationIssue]:
    """Check that essential project files exist."""
    issues: list[ValidationIssue] = []
    required = {
        "backend": [
            "package.json",
            "src/server.js",
            "Dockerfile",
        ],
        "frontend": [
            "package.json",
            "index.html",
            "src/main.jsx",
            "vite.config.js",
            "Dockerfile",
        ],
    }
    for component, files in required.items():
        comp_dir = project_dir / component
        if not comp_dir.exists():
            issues.append(ValidationIssue(
                component=component,
                severity="error",
                message=f"Missing {component}/ directory entirely",
            ))
            continue
        for file_rel in files:
            if not (comp_dir / file_rel).exists():
                issues.append(ValidationIssue(
                    component=component,
                    severity="error",
                    message=f"Missing required file: {file_rel}",
                    file=f"{component}/{file_rel}",
                ))
    return issues


def _check_dockerfile_build(project_dir: Path, component: str) -> list[ValidationIssue]:
    """Check that Dockerfiles don't install only prod deps before a build step."""
    issues: list[ValidationIssue] = []
    dockerfile = project_dir / component / "Dockerfile"
    if not dockerfile.exists():
        return issues
    try:
        text = dockerfile.read_text(encoding="utf-8")
    except OSError:
        return issues

    lines = text.splitlines()
    has_prod_only_install = False
    has_build_step = False
    prod_line_idx = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Detect npm ci --production or npm install --production (before build)
        if re.search(r"npm\s+(ci|install)\s+.*--production", stripped):
            if not has_build_step:
                has_prod_only_install = True
                prod_line_idx = idx + 1
        if re.search(r"npm\s+run\s+build", stripped):
            has_build_step = True

    if has_prod_only_install and has_build_step:
        issues.append(ValidationIssue(
            component=component,
            severity="error",
            message=(
                f"Dockerfile installs production-only deps (line {prod_line_idx}) "
                "before 'npm run build' — devDependencies like vite won't be available"
            ),
            file=f"{component}/Dockerfile",
        ))
    return issues


def _check_frontend_api_wiring(project_dir: Path) -> list[ValidationIssue]:
    """Check that frontend API calls point to the right backend URL/port."""
    issues: list[ValidationIssue] = []
    frontend_dir = project_dir / "frontend"
    if not frontend_dir.exists():
        return issues

    # Check for VITE_API_BASE or hardcoded localhost references in source
    api_base_found = False
    hardcoded_ports: set[str] = set()

    for path in frontend_dir.rglob("*.js"):
        if "node_modules" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "VITE_API_BASE" in text or "import.meta.env" in text:
            api_base_found = True
        # Look for hardcoded localhost with wrong port
        for match in re.finditer(r"http://localhost:(\d+)", text):
            hardcoded_ports.add(match.group(1))

    for path in frontend_dir.rglob("*.jsx"):
        if "node_modules" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "VITE_API_BASE" in text or "import.meta.env" in text:
            api_base_found = True
        for match in re.finditer(r"http://localhost:(\d+)", text):
            hardcoded_ports.add(match.group(1))

    # Backend should be on port 4000
    if hardcoded_ports and "4000" not in hardcoded_ports:
        issues.append(ValidationIssue(
            component="integration",
            severity="error",
            message=(
                f"Frontend references localhost port(s) {hardcoded_ports} "
                "but backend runs on port 4000"
            ),
        ))

    # Check vite.config.js for proxy setup or env-based config
    vite_config = frontend_dir / "vite.config.js"
    if vite_config.exists():
        try:
            vite_text = vite_config.read_text(encoding="utf-8", errors="replace")
        except OSError:
            vite_text = ""
        if _MARKDOWN_FENCE_RE.search(vite_text):
            issues.append(ValidationIssue(
                component="frontend",
                severity="error",
                message="vite.config.js contains markdown fences",
                file="frontend/vite.config.js",
            ))

    return issues


def _check_database_config(project_dir: Path) -> list[ValidationIssue]:
    """Check backend database configuration consistency."""
    issues: list[ValidationIssue] = []
    backend_dir = project_dir / "backend"
    if not backend_dir.exists():
        return issues

    # Check if migrations exist → db.js should also exist
    migrations_dir = backend_dir / "migrations"
    db_js = backend_dir / "src" / "db.js"
    has_migrations = migrations_dir.exists() and any(migrations_dir.glob("*.sql"))

    if has_migrations and not db_js.exists():
        issues.append(ValidationIssue(
            component="backend",
            severity="error",
            message="Migrations exist but src/db.js is missing — database won't be initialised",
            file="backend/src/db.js",
        ))

    # Check that db.js references better-sqlite3
    if db_js.exists():
        try:
            text = db_js.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if "better-sqlite3" not in text and "sqlite" not in text.lower():
            issues.append(ValidationIssue(
                component="backend",
                severity="warning",
                message="db.js does not reference better-sqlite3 or sqlite",
                file="backend/src/db.js",
            ))

    # Check package.json includes better-sqlite3 if migrations exist
    pkg_path = backend_dir / "package.json"
    if has_migrations and pkg_path.exists():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pkg = {}
        deps = pkg.get("dependencies", {})
        if "better-sqlite3" not in deps:
            issues.append(ValidationIssue(
                component="backend",
                severity="error",
                message=(
                    "Migrations exist but better-sqlite3 is not in "
                    "package.json dependencies"
                ),
                file="backend/package.json",
            ))

    # Check docker-compose DB_PATH consistency
    compose_path = project_dir / "docker-compose.yml"
    env_path = backend_dir / ".env"
    if compose_path.exists() and env_path.exists():
        try:
            compose_text = compose_path.read_text(encoding="utf-8", errors="replace")
            env_text = env_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            compose_text = env_text = ""

        compose_db = re.search(r"DB_PATH=(\S+)", compose_text)
        env_db = re.search(r"DB_PATH=(\S+)", env_text)
        if compose_db and env_db:
            # They should reference the same relative filename
            compose_filename = Path(compose_db.group(1)).name
            env_filename = Path(env_db.group(1)).name
            if compose_filename != env_filename:
                issues.append(ValidationIssue(
                    component="integration",
                    severity="error",
                    message=(
                        f"DB_PATH mismatch: docker-compose uses '{compose_db.group(1)}' "
                        f"but .env uses '{env_db.group(1)}'"
                    ),
                ))

    return issues


def _check_docker_compose(project_dir: Path) -> list[ValidationIssue]:
    """Check docker-compose.yml structure."""
    issues: list[ValidationIssue] = []
    compose_path = project_dir / "docker-compose.yml"
    if not compose_path.exists():
        issues.append(ValidationIssue(
            component="infra",
            severity="warning",
            message="Missing docker-compose.yml",
            file="docker-compose.yml",
        ))
        return issues
    try:
        text = compose_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return issues

    if "backend" not in text:
        issues.append(ValidationIssue(
            component="infra",
            severity="error",
            message="docker-compose.yml does not define a backend service",
            file="docker-compose.yml",
        ))
    if "frontend" not in text:
        issues.append(ValidationIssue(
            component="infra",
            severity="error",
            message="docker-compose.yml does not define a frontend service",
            file="docker-compose.yml",
        ))

    return issues


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_project(project_dir: Path) -> ValidationResult:
    """Run all validation checks on a generated project directory.

    Parameters
    ----------
    project_dir:
        Path to the project root (contains backend/, frontend/,
        docker-compose.yml).

    Returns
    -------
    ValidationResult with all issues found.
    """
    result = ValidationResult()

    if not project_dir.exists():
        result.issues.append(ValidationIssue(
            component="project",
            severity="error",
            message=f"Project directory does not exist: {project_dir}",
        ))
        return result

    # Structural checks
    result.issues.extend(_check_required_files(project_dir))
    result.issues.extend(_check_package_json(project_dir, "backend"))
    result.issues.extend(_check_package_json(project_dir, "frontend"))

    # Code quality checks
    result.issues.extend(_check_markdown_fences(project_dir, "backend"))
    result.issues.extend(_check_markdown_fences(project_dir, "frontend"))

    # Docker checks
    result.issues.extend(_check_dockerfile_build(project_dir, "backend"))
    result.issues.extend(_check_dockerfile_build(project_dir, "frontend"))
    result.issues.extend(_check_docker_compose(project_dir))

    # Integration checks
    result.issues.extend(_check_frontend_api_wiring(project_dir))
    result.issues.extend(_check_database_config(project_dir))

    logger.info(
        "Validation complete for %s: %s",
        project_dir.name,
        result.summary(),
    )
    return result
