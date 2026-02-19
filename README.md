# fs-agent

Python-based multi-agent orchestrator that drafts full-stack JavaScript applications by coordinating specialized agents (architect, backend, frontend, infra) through a sequential workflow. The initial goal is to prove out a deterministic stage-gate pipeline while leaving room to plug in richer orchestration patterns later.

## Why it exists
- Capture product requirements once and let agents specialize without stepping on each other.
- Emit JavaScript/TypeScript backend APIs that the generated frontend consumes.
- Produce infra plans (Docker, CI/CD, deployment targets) so code can ship.
- Keep orchestration logic in Python to experiment with different coordination strategies.

## Components
1. **Orchestrator** – Applies an orchestration pattern (sequential for now) and hands off context between agents.
2. **Architect Agent** – Converts a natural-language product request into a validated YAML spec.
3. **Backend Agent** – Plans Node/Express APIs and emits MCP filesystem operations for the full project.
4. **Frontend Agent** – Plans UI flows, generates React code, and publishes MCP instructions for the SPA.
5. **Infra Agent** – Suggests deployment architecture, environments, and automation steps.
6. **Shared Context** – pydantic spec + artifact tracker that every agent can read/append.

## Quickstart
```bash
# create a virtual environment however you prefer, then
pip install -e .[dev]

# describe the product you want
fs-agent "Build a collaborative task manager" --artifact-dir artifacts/demo

# optional: point agents at a live LLM (defaults to a deterministic placeholder)
FS_AGENT_LLM_PROVIDER=openai FS_AGENT_OPENAI_API_KEY=sk-... \
	fs-agent "Build a collaborative task manager" --llm-model gpt-4o-mini
```
The CLI prints a Rich-formatted summary and writes each agent's outputs into the requested artifact directory. Expect the stack below:
- `architect_spec.yaml` – generated YAML spec plus JSON copies for downstream automations.
- `backend_*.json` / `frontend_*.json` – MCP-friendly filesystem plans alongside the generated TypeScript/React code.
- `infra_plan.md` / `infra_runbook.md` – deployment guidance that references every upstream artifact.
The CLI prints a Rich-formatted summary and writes each agent's outputs into the requested artifact directory (JSON blueprints plus Markdown/code artifacts such as `backend_backend_source.json`, `backend_plan.md`, and `backend_service.ts`).

- **.env loading** – The CLI automatically loads environment variables from `.env` in the workspace root via `python-dotenv`.
- **.env loading** – The CLI automatically loads environment variables from `.env` in the workspace root (or from a custom path via `--env-file`).
- **Default** – `FS_AGENT_LLM_PROVIDER` defaults to `dummy`, so agents emit deterministic placeholder code without leaving the machine.
- **OpenAI** – export `FS_AGENT_LLM_PROVIDER=openai` and `FS_AGENT_OPENAI_API_KEY=<your key>`. Adjust the target model via `FS_AGENT_LLM_MODEL`.
- **Per-run overrides** – Set the env vars above inline with the command (e.g. `FS_AGENT_LLM_MODEL=gpt-4.1 fs-agent "..."`).

## Orchestration Pattern
The sequential pattern enforces the order `Architect → Backend → Frontend → Infra`. Each agent receives:
- The latest project spec (architect writes it, downstream agents require it)
- Aggregated artifacts from prior agents
- A lightweight execution transcript to keep track of decisions

The orchestrator persists each agent's summary, artifacts, and status so that other patterns (parallel or dynamic) can reuse the results later.

## Extensibility Hooks
- **Agent Registry** – Register new roles or swap implementations without touching the orchestrator.
- **Pattern Factory** – Choose orchestration strategies based on config (`sequential` today; hybrid/parallel later).
- **Artifact Contracts** – Every agent returns structured artifacts (metadata + suggested file paths) so downstream automation can materialize real source files.
- **Spec Schema** – The architect agent emits YAML compatible with the `ProjectSpec` pydantic schema; extend it as requirements evolve.

## Roadmap Ideas
- Parallelize frontend/backend once contracts are strong.
- Add reviewer/QA agents that lint, test, and simulate deployments.
- Integrate real LLM calls behind the agent skeletons (currently deterministic placeholders).
- Execute MCP filesystem operations automatically after validating the generated plans.
- Wire deployment steps into Terraform/Pulumi or platform CLIs.

## Contributing
1. Create/activate a Python 3.11+ environment.
2. `pip install -e .[dev]`
3. Run `ruff check .` and `pytest` before opening pull requests.

This repo is intentionally small to make experimentation fast—feel free to change the orchestration pattern or add new agents as your research progresses.
