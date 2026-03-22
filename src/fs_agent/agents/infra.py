"""Infrastructure agent — bootstraps SQLite, runs migrations, starts services."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..context import RunContext
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class InfraAgent(BaseAgent):
    role = AgentRole.INFRA

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        slug = context._slugify(spec.metadata.name)
        projects_dir = context.projects_dir

        # Locate sub-projects written by backend/frontend agents
        backend_dir = projects_dir / "backend"
        frontend_dir = projects_dir / "frontend"

        log_lines: list[str] = []
        errors: list[str] = []

        # --- 1. Ensure SQLite data directory exists -------------------------
        db_name = slug.replace("-", "_") + "_db"
        if backend_dir.exists():
            data_dir = backend_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            log_lines.append(f"✓ Created data directory {data_dir}")
        else:
            errors.append(f"Backend directory not found: {backend_dir}")

        # --- 2. Write a .env so the backend can find the SQLite DB ----------
        env_path = backend_dir / ".env"
        if backend_dir.exists():
            env_contents = (
                f"PORT=4000\n"
                f"DB_PATH=./data/{db_name}.db\n"
            )
            env_path.write_text(env_contents)
            log_lines.append(f"✓ Wrote {env_path}")
        else:
            errors.append(f"Backend directory not found: {backend_dir}")

        # --- 3. npm install + migrate in backend ----------------------------
        if backend_dir.exists():
            self._run_step(
                "npm install (backend)",
                ["npm", "install"],
                log_lines,
                errors,
                cwd=backend_dir,
            )
            self._run_step(
                "npm run migrate (backend)",
                ["npm", "run", "migrate"],
                log_lines,
                errors,
                cwd=backend_dir,
            )

        # --- 4. npm install in frontend -------------------------------------
        if frontend_dir.exists():
            self._run_step(
                "npm install (frontend)",
                ["npm", "install"],
                log_lines,
                errors,
                cwd=frontend_dir,
            )

        # --- 5. Start backend (background) ----------------------------------
        backend_proc = None
        if backend_dir.exists():
            backend_proc = self._start_background(
                "backend dev server",
                ["npm", "run", "dev"],
                log_lines,
                errors,
                cwd=backend_dir,
            )

        # --- 6. Start frontend (background) ---------------------------------
        frontend_proc = None
        if frontend_dir.exists():
            frontend_proc = self._start_background(
                "frontend dev server",
                ["npm", "run", "dev"],
                log_lines,
                errors,
                cwd=frontend_dir,
            )

        # --- 7. Write docker-compose.yml at project root --------------------
        compose_path = projects_dir / "docker-compose.yml"
        compose_contents = self._render_docker_compose(db_name)
        compose_path.write_text(compose_contents)
        log_lines.append(f"✓ Wrote {compose_path}")

        # --- Build result ---------------------------------------------------
        body = "\n".join(log_lines) or "No steps executed."
        status = "success" if not errors else "error"
        attachments = [
            AgentArtifact(
                name="infra_log.md",
                kind="doc",
                description="Infrastructure bootstrap log",
                body=body,
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=(
                f"Bootstrapped infra: prepared SQLite DB '{db_name}', ran migrations, "
                f"started backend (port 4000) and frontend dev servers."
            ),
            artifacts={
                "infra_log": body,
                "db_name": db_name,
                "backend_dir": str(backend_dir),
                "frontend_dir": str(frontend_dir),
                "backend_pid": backend_proc.pid if backend_proc else None,
                "frontend_pid": frontend_proc.pid if frontend_proc else None,
            },
            attachments=attachments,
            diagnostics=errors,
            status=status,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_docker_compose(self, db_name: str) -> str:
        return (
            "version: '3.8'\n\n"
            "services:\n"
            "  backend:\n"
            "    build: ./backend\n"
            "    ports:\n"
            "      - '4000:4000'\n"
            "    environment:\n"
            "      - PORT=4000\n"
            f"      - DB_PATH=/app/data/{db_name}.db\n"
            "    volumes:\n"
            "      - backend-data:/app/data\n"
            "    restart: unless-stopped\n\n"
            "  frontend:\n"
            "    build: ./frontend\n"
            "    ports:\n"
            "      - '3000:80'\n"
            "    depends_on:\n"
            "      - backend\n"
            "    restart: unless-stopped\n\n"
            "volumes:\n"
            "  backend-data:\n"
        )

    def _run_step(
        self,
        label: str,
        cmd: list[str],
        log: list[str],
        errors: list[str],
        *,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        """Run a blocking shell command and capture output."""
        self.logger.info("Running: %s", label)
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                log.append(f"✓ {label}")
                if result.stdout.strip():
                    log.append(f"  stdout: {result.stdout.strip()[:500]}")
            else:
                msg = f"✗ {label} (exit {result.returncode})"
                log.append(msg)
                if result.stderr.strip():
                    log.append(f"  stderr: {result.stderr.strip()[:500]}")
                errors.append(msg)
            return result
        except FileNotFoundError:
            msg = f"✗ {label}: command not found ({cmd[0]})"
            log.append(msg)
            errors.append(msg)
            return None
        except subprocess.TimeoutExpired:
            msg = f"✗ {label}: timed out after 120s"
            log.append(msg)
            errors.append(msg)
            return None

    def _start_background(
        self,
        label: str,
        cmd: list[str],
        log: list[str],
        errors: list[str],
        *,
        cwd: Path | None = None,
    ) -> subprocess.Popen[str] | None:
        """Start a long-running process in the background."""
        self.logger.info("Starting background: %s", label)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.append(f"✓ Started {label} (PID {proc.pid})")
            return proc
        except FileNotFoundError:
            msg = f"✗ {label}: command not found ({cmd[0]})"
            log.append(msg)
            errors.append(msg)
            return None
        except Exception as exc:
            msg = f"✗ {label}: {exc}"
            log.append(msg)
            errors.append(msg)
            return None
