"""Docker lifecycle management for runtime evaluation.

Handles building, starting, health-checking, and tearing down
generated full-stack projects via ``docker compose``.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default Dockerfiles for generated projects that are missing them
# ---------------------------------------------------------------------------

_BACKEND_DOCKERFILE = """\
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
EXPOSE 4000
CMD ["node", "src/server.js"]
"""

_BACKEND_DOCKERIGNORE = """\
node_modules
.env
data/
*.sqlite
"""

_FRONTEND_DOCKERFILE = """\
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
"""

_FRONTEND_DOCKERIGNORE = """\
node_modules
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProjectInstance:
    """Represents a running project with accessible service URLs."""

    project_dir: Path
    backend_port: int = 4000
    frontend_port: int = 3000
    backend_healthy: bool = False
    frontend_healthy: bool = False

    @property
    def backend_url(self) -> str:
        return f"http://localhost:{self.backend_port}"

    @property
    def frontend_url(self) -> str:
        return f"http://localhost:{self.frontend_port}"


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


def ensure_dockerfiles(project_dir: Path) -> None:
    """Generate missing Dockerfiles so ``docker compose build`` succeeds."""
    backend_dir = project_dir / "backend"
    frontend_dir = project_dir / "frontend"

    if backend_dir.exists() and not (backend_dir / "Dockerfile").exists():
        logger.info("Generating missing backend Dockerfile in %s", backend_dir)
        (backend_dir / "Dockerfile").write_text(_BACKEND_DOCKERFILE)
        ignore = backend_dir / ".dockerignore"
        if not ignore.exists():
            ignore.write_text(_BACKEND_DOCKERIGNORE)

    if frontend_dir.exists() and not (frontend_dir / "Dockerfile").exists():
        logger.info("Generating missing frontend Dockerfile in %s", frontend_dir)
        (frontend_dir / "Dockerfile").write_text(_FRONTEND_DOCKERFILE)
        ignore = frontend_dir / ".dockerignore"
        if not ignore.exists():
            ignore.write_text(_FRONTEND_DOCKERIGNORE)


def start_project(
    project_dir: Path,
    *,
    build_timeout: int = 300,
    healthy_timeout: int = 60,
) -> ProjectInstance | None:
    """Build and start a project via ``docker compose up``.

    Returns a :class:`ProjectInstance` on success, or ``None`` if the
    build/start failed.
    """
    compose_file = project_dir / "docker-compose.yml"
    if not compose_file.exists():
        logger.error("No docker-compose.yml in %s", project_dir)
        return None

    ensure_dockerfiles(project_dir)

    # -- Build ---------------------------------------------------------------
    print(f"[executor] Building Docker images for {project_dir.name} ...")
    try:
        result = subprocess.run(
            ["docker", "compose", "build"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=build_timeout,
        )
    except FileNotFoundError:
        logger.error("'docker' command not found — is Docker installed?")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Docker build timed out after %ds", build_timeout)
        return None

    if result.returncode != 0:
        logger.error("Docker build failed:\n%s", result.stderr[:3000])
        return None

    # -- Start ---------------------------------------------------------------
    print("[executor] Starting containers ...")
    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.error("docker compose up timed out")
        stop_project(project_dir)
        return None

    if result.returncode != 0:
        logger.error("docker compose up failed:\n%s", result.stderr[:3000])
        stop_project(project_dir)
        return None

    instance = ProjectInstance(project_dir=project_dir)

    # -- Health-check --------------------------------------------------------
    print("[executor] Waiting for services to become healthy ...")
    instance.backend_healthy = _wait_for_url(
        instance.backend_url, timeout=healthy_timeout
    )
    instance.frontend_healthy = _wait_for_url(
        instance.frontend_url, timeout=healthy_timeout
    )

    if instance.backend_healthy:
        print(f"[executor] Backend healthy at {instance.backend_url}")
    else:
        logger.warning("Backend did NOT become healthy within %ds", healthy_timeout)
        _dump_container_logs(project_dir, "backend")

    if instance.frontend_healthy:
        print(f"[executor] Frontend healthy at {instance.frontend_url}")
    else:
        logger.warning("Frontend did NOT become healthy within %ds", healthy_timeout)
        _dump_container_logs(project_dir, "frontend")

    return instance


def stop_project(project_dir: Path) -> None:
    """Tear down containers and volumes for a project."""
    print(f"[executor] Stopping containers for {project_dir.name} ...")
    try:
        subprocess.run(
            ["docker", "compose", "down", "-v", "--remove-orphans"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Failed to cleanly stop containers in %s", project_dir)


def find_project_dir(artifact_root: Path, task_id: str, pattern: str) -> Path | None:
    """Locate the project directory (contains docker-compose.yml).

    Layout: ``<artifact_root>/<task_id>/<pattern>/projects/<slug>/``
    """
    projects_root = artifact_root / task_id / pattern / "projects"
    if not projects_root.exists():
        return None
    dirs = [d for d in projects_root.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return dirs[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wait_for_url(url: str, *, timeout: int = 60, interval: float = 2.0) -> bool:
    """Poll *url* until it returns a non-5xx response."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status < 500:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(interval)
    return False


def _dump_container_logs(project_dir: Path, service: str) -> None:
    """Print the last 30 lines of a service's Docker logs for debugging."""
    try:
        result = subprocess.run(
            ["docker", "compose", "logs", "--tail=30", service],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            logger.warning("Last logs for %s:\n%s", service, result.stdout[-2000:])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
