# fs-agent

Python-based multi-agent orchestrator that drafts full-stack JavaScript applications by coordinating specialized agents (architect, backend, frontend, infra) through configurable orchestration patterns. Built as a research platform to benchmark how different coordination strategies affect code-generation quality, token efficiency, and time.

## Why it exists
- Capture product requirements once and let agents specialize without stepping on each other.
- Emit JavaScript backend APIs (Express) and React/Vite frontends that consume them.
- Produce infra automation that creates databases, runs migrations, and starts dev servers.
- Keep orchestration logic in Python to experiment with different coordination strategies.
- Benchmark six orchestration patterns head-to-head on the same task dataset.

## Components
1. **Orchestrator** – Applies a chosen orchestration pattern and hands off context between agents.
2. **Architect Agent** – Converts a natural-language product request into a validated project spec.
3. **Backend Agent** – Plans Node/Express APIs, generates code with Jest tests, and emits MCP filesystem operations.
4. **Frontend Agent** – Plans UI flows, generates React code with Vitest tests, and publishes MCP instructions.
5. **Infra Agent** – Creates MySQL databases, runs migrations, and starts backend/frontend dev servers.
6. **Shared Context** – Pydantic spec + artifact tracker that every agent can read/append.
7. **Benchmark Runner** – Reads tasks from a dataset, runs every pattern, and produces metrics reports.

## Orchestration Patterns
Four patterns are available, selectable via the `--pattern` flag:

| Pattern | Description |
|---|---|
| `sequential` | Fixed order: Architect → Backend → Frontend → Infra |
| `centralized` | An LLM coordinator decides which agent runs next in a loop |
| `decentralized` | Each agent decides who should handle the work next (handoff routing) |
| `parallel` | Fan-out/fan-in — independent agents run concurrently via ThreadPoolExecutor |

### Post-Generation Validation
Each pattern runs a validation loop after agents complete. The loop checks:
- Required files exist (package.json, entry points, Dockerfiles)
- No LLM markdown fences left in source files
- Frontend ↔ Backend API wiring (port consistency, proxy setup)
- Database config sync (better-sqlite3 dependency, DB_PATH consistency)
- Dockerfile build feasibility (devDeps available for build step)

If validation fails, the responsible agents are re-run with error feedback. Retries are configurable via `--max-retries` (default: 3, set to 0 to disable).

## Setup

### Prerequisites
- Python 3.11+
- MySQL server (for the infra agent to create databases and run migrations)
- Node.js 18+ (for the generated JavaScript projects)

### Installation
```bash
# Clone and enter the project
git clone <repo-url> && cd fs-agent

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

### Environment variables
Create a `.env` file in the project root (auto-loaded via `python-dotenv`):

```env
FS_AGENT_LLM_PROVIDER=ollama     # or "openai", "qwen", "dummy"
FS_AGENT_OPENAI_API_KEY=sk-...   # required when provider is openai/qwen, optional for ollama
FS_AGENT_LLM_MODEL=llama3.1:8b   # any Ollama model or OpenAI model
FS_AGENT_LLM_BASE_URL=http://localhost:11434/v1  # optional, overrides default
LLM_PROVIDER=ollama              # alternative env var name
LLM_MODEL=llama3.1:8b           # alternative env var name
```

If no API key is set, the system falls back to a deterministic dummy LLM.

## Running a Single Project

```bash
# Generate a project using the default sequential pattern
fs-agent run "Build a collaborative task manager"

# Specify an output directory
fs-agent run "Build a stock portfolio tracker" --artifact-dir artifacts/demo

# Choose a specific orchestration pattern
fs-agent run "Build a recipe sharing platform" --pattern centralized

# Dry run (skip infra side-effects like DB creation and server startup)
fs-agent run "Build a blog engine" --pattern parallel --dry-run
```

### Output structure
Each generated project is self-enclosed under `artifacts/projects/<slug>/`:
```
artifacts/projects/collaborative-task-manager/
├── backend/          # Express API with Jest tests
├── frontend/         # React/Vite SPA with Vitest tests
└── metadata/         # Agent artifacts, specs, and attachments
```

## Running the Benchmark

The benchmark runner reads tasks from `dataset/tasks_with_difficulty.json`, executes each task through every orchestration pattern, and records detailed metrics.

### Quick start
```bash
# Run all 100 tasks × all 6 patterns (600 runs total)
fs-agent benchmark dataset/tasks_with_difficulty.json

# Run a quick test with just 1 task and 2 patterns
fs-agent benchmark dataset/tasks_with_difficulty.json --max-tasks 1 --patterns sequential,parallel

# Run specific task IDs only
fs-agent benchmark dataset/tasks_with_difficulty.json --task-ids 000001,000005,000010

# Custom output directory
fs-agent benchmark dataset/tasks_with_difficulty.json --artifact-root artifacts/my_benchmark
```

### Benchmark CLI options
| Flag | Description |
|---|---|
| `DATASET` (required) | Path to the tasks JSON file |
| `--patterns` | Comma-separated subset of patterns (default: all four) |
| `--task-ids` | Comma-separated task IDs to run (default: all) |
| `--max-tasks` | Cap on number of tasks to process |
| `--artifact-root` | Root directory for outputs (default: `artifacts/benchmark/`) |

### Benchmark output structure
```
artifacts/benchmark/
├── 000001/
│   ├── sequential/
│   │   ├── run.log              # Detailed DEBUG-level log for this run
│   │   └── projects/…           # Generated project files
│   ├── centralized/
│   ├── decentralized/
│   └── parallel/
├── 000002/
│   └── …
└── results/
    ├── benchmark_results.json   # Full per-run metrics with agent breakdown
    ├── benchmark_results.csv    # Flat CSV for spreadsheet/notebook analysis
    └── benchmark_summary.json   # Per-pattern aggregated statistics
```

### Metrics collected
| Metric | Description |
|---|---|
| `wall_clock_seconds` | Total elapsed time for the run |
| `agent_total_seconds` | Sum of individual agent durations |
| `orchestration_overhead_seconds` | Wall clock − agent time (coordination cost) |
| `llm_call_count` | Total LLM API calls (including coordinator/critic calls) |
| `prompt_tokens` | Input tokens sent to the LLM |
| `completion_tokens` | Output tokens received from the LLM |
| `total_tokens` | Combined token usage |
| `agent_count` | Number of agents dispatched |
| `coordinator_calls` | Extra LLM calls for routing/critique overhead |
| Per-agent breakdown | Role, status, duration, artifact and attachment counts |

## Extensibility Hooks
- **Agent Registry** – Register new roles or swap implementations without touching the orchestrator.
- **Pattern Factory** – Add new orchestration strategies by subclassing `OrchestrationPattern`.
- **Artifact Contracts** – Every agent returns structured artifacts so downstream automation can materialize real source files.
- **Spec Schema** – The architect agent emits a spec compatible with the `ProjectSpec` Pydantic schema; extend it as requirements evolve.

## Contributing
1. Create/activate a Python 3.11+ environment.
2. `pip install -e .[dev]`
3. Run `ruff check .` and `pytest` before opening pull requests.
