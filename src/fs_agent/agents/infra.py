"""Infrastructure agent — bootstraps MySQL, runs migrations, starts services."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..context import RunContext
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class InfraAgent(BaseAgent):
    role = AgentRole.INFRA

    # MySQL credentials (localhost dev defaults)
    _DB_HOST = "localhost"
    _DB_USER = "root"
    _DB_PASS = ""

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        slug = context._slugify(spec.metadata.name)
        projects_dir = context.projects_dir

        # Locate sub-projects written by backend/frontend agents
        backend_dir = projects_dir / f"{slug}-backend"
        frontend_dir = projects_dir / f"{slug}-frontend"

        log_lines: list[str] = []
        errors: list[str] = []

        # --- 1. Create the MySQL database -----------------------------------
        db_name = slug.replace("-", "_") + "_db"
        self._run_step(
            f"Create MySQL database '{db_name}'",
            [
                "mysql",
                f"--host={self._DB_HOST}",
                f"--user={self._DB_USER}",
                "-e",
                f"CREATE DATABASE IF NOT EXISTS `{db_name}`;",
            ],
            log_lines,
            errors,
        )

        # --- 2. Write a .env so the backend can connect ---------------------
        env_path = backend_dir / ".env"
        if backend_dir.exists():
            env_contents = (
                f"PORT=4000\n"
                f"DB_HOST={self._DB_HOST}\n"
                f"DB_PORT=3306\n"
                f"DB_USER={self._DB_USER}\n"
                f"DB_PASSWORD={self._DB_PASS}\n"
                f"DB_NAME={db_name}\n"
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
                f"Bootstrapped infra: created DB '{db_name}', ran migrations, "
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
