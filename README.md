# fs-agent

Python-based multi-agent orchestrator that drafts full-stack JavaScript applications by coordinating specialized agents (backend, frontend, infra) through a sequential workflow. The initial goal is to prove out a deterministic stage-gate pipeline while leaving room to plug in richer orchestration patterns later.

## Why it exists
- Capture product requirements once and let agents specialize without stepping on each other.
- Emit JavaScript/TypeScript backend APIs that the generated frontend consumes.
- Produce infra plans (Docker, CI/CD, deployment targets) so code can ship.
- Keep orchestration logic in Python to experiment with different coordination strategies.

## Components
1. **Orchestrator** – Applies an orchestration pattern (sequential for now) and hands off context between agents.
2. **Backend Agent** – Plans Node/Express (or similar) API implementations and testing hooks.
3. **Frontend Agent** – Plans UI flows that consume the generated APIs.
4. **Infra Agent** – Suggests deployment architecture, environments, and automation steps.
5. **Shared Context** – pydantic spec + artifact tracker that every agent can read/append.

## Quickstart
```bash
# create a virtual environment however you prefer, then
pip install -e .[dev]

# run the orchestrator with the sample specification
fs-agent examples/specs/todo_app.yaml --artifact-dir artifacts/demo
```

The CLI prints a Rich-formatted summary and writes each agent's outputs into the requested artifact directory (JSON blueprints plus Markdown plans such as `backend_backend_blueprint.json` and `backend_plan.md`).

## Orchestration Pattern
The sequential pattern enforces the order `Backend → Frontend → Infra`. Each agent receives:
- The validated project spec
- Aggregated artifacts from prior agents
- A lightweight execution transcript to keep track of decisions

The orchestrator persists each agent's summary, artifacts, and status so that other patterns (parallel or dynamic) can reuse the results later.

## Extensibility Hooks
- **Agent Registry** – Register new roles or swap implementations without touching the orchestrator.
- **Pattern Factory** – Choose orchestration strategies based on config (`sequential` today; hybrid/parallel later).
- **Artifact Contracts** – Every agent returns structured artifacts (metadata + suggested file paths) so downstream automation can materialize real source files.
- **Spec Schema** – `examples/specs/todo_app.yaml` shows how to describe endpoints, UI routes, and infra targets; extend it as requirements evolve.

## Roadmap Ideas
- Parallelize frontend/backend once contracts are strong.
- Add reviewer/QA agents that lint, test, and simulate deployments.
- Integrate real LLM calls behind the agent skeletons (currently deterministic placeholders).
- Emit actual JavaScript files instead of textual plans.
- Wire deployment steps into Terraform/Pulumi or platform CLIs.

## Contributing
1. Create/activate a Python 3.11+ environment.
2. `pip install -e .[dev]`
3. Run `ruff check .` and `pytest` before opening pull requests.

This repo is intentionally small to make experimentation fast—feel free to change the orchestration pattern or add new agents as your research progresses.
