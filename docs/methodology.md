# Methodology

## 1. Overview

This study presents a comparative empirical evaluation of six multi-agent orchestration patterns for LLM-driven full-stack code generation. We designed and implemented a unified experimental platform — **fs-agent** — in which the same set of four specialist agents (architect, backend, frontend, infrastructure) can be coordinated through any of the six patterns without modification to the agents themselves. Each pattern is evaluated against a standardised task dataset, and we collect fine-grained metrics covering execution time, token consumption, communication overhead, and structural output completeness.

---

## 2. System Architecture

### 2.1 Agent Design

The system comprises four domain-specialist agents, each implemented as a subclass of a common `BaseAgent` abstract class. Every agent receives a shared `RunContext` object containing the user's natural-language brief, the current project specification, accumulated artifacts from prior agents, and a reference to the LLM client. Each agent returns a structured `AgentResult` containing a summary, typed artifacts, file attachments, execution status, and timestamps.

| Agent | Role | Inputs | Outputs |
|-------|------|--------|---------|
| **Architect** | Translates the user brief into a structured project specification | User request, JSON Schema of the `ProjectSpec` model | Validated `ProjectSpec` (endpoints, routes, components, data models, infra targets) |
| **Backend** | Generates Express.js API server code | Project spec (backend slice), architect artifacts | MCP filesystem plan, route handlers, migration files, `package.json`, Jest test suite |
| **Frontend** | Generates React/Vite single-page application code | Project spec (frontend slice), backend endpoint signatures | MCP filesystem plan, page components, API hooks, `package.json`, Vitest test suite |
| **Infrastructure** | Provisions runtime environment | Project spec (infra slice), generated backend/frontend directories | MySQL database creation, dependency installation, migration execution, dev server startup |

Agents are registered in a central `AgentRegistry` (a role → class mapping), enabling any orchestration pattern to instantiate agents by role without coupling to concrete implementations. A shared `execute_agent()` helper function provides the canonical dispatch sequence: invoke the agent, persist its output to the metadata directory, record an `AgentReport` on the shared context, and log timing information.

### 2.2 Project Specification Model

The architect agent produces a `ProjectSpec` — a deeply nested Pydantic model that serves as the contract between all downstream agents. The specification includes:

- **Metadata**: project name, summary, owner, version.
- **Backend**: language (JavaScript), framework (Express), API style (REST), endpoint definitions (method, path, request/response schemas, error codes), data models (fields, relationships, indexes), and database configuration (MySQL, migrations).
- **Frontend**: framework (React), styling (Tailwind), route definitions (path, consumed endpoints, components), component declarations (props, consumed endpoints), and theme tokens.
- **Infrastructure**: CI/CD provider, deployment targets (environment, runtime).

A built-in cross-referencing validator checks that every frontend route's `consumes` references resolve to declared backend endpoints, and flags orphaned endpoints not consumed by any route or component. The model exposes its JSON Schema so the architect agent can embed it in the LLM prompt, guiding the model to produce structurally valid output.

### 2.3 LLM Integration

All LLM interactions are mediated through a `BaseLLMClient` abstraction with two implementations:

- **`OpenAILLMClient`**: Makes real HTTP requests to the OpenAI Chat Completions API via `httpx`. After each call, it parses the API-reported `usage` object and records prompt tokens, completion tokens, and total tokens.
- **`DummyLLMClient`**: Returns deterministic placeholder text and estimates token counts heuristically (1 token ≈ 4 characters) for offline testing.

The base class accumulates per-call token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`) and a call counter across all invocations. A `reset_usage()` method zeroes these counters between benchmark runs, ensuring each run's metrics reflect only its own LLM consumption. The `usage_stats` property exposes the accumulated counters as a dictionary for the benchmark harness to read.

### 2.4 Output Structure

Each generated project is self-enclosed under a single directory:

```
artifacts/projects/<project-slug>/
├── backend/      # Express API with Jest tests
├── frontend/     # React/Vite SPA with Vitest tests
└── metadata/     # Agent artifacts, specs, attachments
```

---

## 3. Orchestration Patterns

All six patterns implement a common `OrchestrationPattern` abstract base class with a single method `run(context: RunContext) → Sequence[AgentReport]`. This ensures the benchmark harness can treat every pattern uniformly. The patterns differ only in **how** and **when** agents are dispatched.

### 3.1 Sequential

The sequential pattern enforces a deterministic linear pipeline:

$$\text{Architect} \rightarrow \text{Backend} \rightarrow \text{Frontend} \rightarrow \text{Infra}$$

Each agent runs to completion before the next begins. There are no LLM coordination calls; the execution order is statically defined. This pattern serves as the **baseline** against which all other patterns are compared, as it represents the simplest possible multi-agent workflow with zero coordination overhead.

### 3.2 Centralised Coordinator

A single LLM-driven coordinator loop governs agent dispatch. At each iteration, the coordinator receives a prompt containing:

- The original user request.
- The list of available agents and their completion status.
- Dependency rules (e.g., architect must run first; infra requires backend and frontend).

The coordinator returns a JSON decision: `{"action": "run", "agent": "<name>", "reason": "..."}` or `{"action": "done", ...}`. The chosen agent is dispatched through the standard `execute_agent()` helper. The loop continues until the coordinator signals completion or a safety guard of 10 iterations is reached. If the LLM call fails, the system falls back to canonical sequential order. The coordinator uses a temperature of 0.0 to maximise determinism in routing decisions.

### 3.3 Decentralised Handoff

Routing intelligence is distributed across agents. After each agent completes, the orchestrator queries the LLM — framed as the outgoing agent — to decide who should handle the work next. The handoff prompt includes the just-completed agent's status and output summary, the remaining agent list, and dependency rules. The LLM responds with `{"next": "<agent>", "reason": "..."}` or `{"next": "done", ...}`.

The seed agent is always the architect (since it depends only on the user request). Execution continues until an agent hands off to `"done"`, all agents have run, or the 10-iteration safety guard fires. Unvisited agents are swept up in canonical order as a fallback.

### 3.4 Hierarchical (Two-Level Supervisor Tree)

This pattern introduces a two-level control hierarchy:

1. **Root supervisor** (LLM): selects which *phase* to execute next.
2. **Phase supervisor** (LLM): within the selected phase, decides which *agent* runs next.

The default topology defines two phases:

| Phase | Agents |
|-------|--------|
| Planning | Architect |
| Build | Backend, Frontend, Infra |

Both supervisors receive dependency-aware prompts and respond with JSON. The root supervisor has a 10-iteration guard; the phase supervisor also has a 10-iteration guard per phase. Custom topologies can be injected via `PhaseGroup` dataclass configurations.

### 3.5 Parallel (Fan-Out / Fan-In)

Agents are grouped into *stages* that execute in sequence, but agents **within** a stage run concurrently using Python's `ThreadPoolExecutor` with up to 4 worker threads. The default stage configuration is:

| Stage | Agents | Execution |
|-------|--------|-----------|
| Planning | Architect | Serial |
| Build | Backend, Frontend | **Parallel** |
| Deploy | Infra | Serial |

There are no LLM coordination calls; stage membership is statically defined. Thread safety is ensured by collecting `AgentReport` objects from futures and calling `context.record()` exclusively on the main thread after all futures in a stage complete. This pattern tests whether concurrency in the build phase yields wall-clock improvements over the sequential baseline.

### 3.6 Iterative Refinement (Critic-Driven Retry Loop)

Each agent runs in the canonical order (architect → backend → frontend → infra), but after each execution an LLM *critic* evaluates the output against role-specific quality criteria. The critic scores the output on a 1–10 scale. If the score falls below a pass threshold of 7, the agent is re-run (up to 2 retries, for a maximum of 3 attempts per agent). Critic feedback is stored in the agent report's metadata so it is available in the context on retry.

The quality criteria are tailored per role:

- **Architect** (5 criteria): spec completeness, frontend–backend cross-references, migration presence, infra targets, no orphaned endpoints.
- **Backend** (5 criteria): correct dependencies, route handler coverage, database connection, environment documentation, error handling middleware.
- **Frontend** (5 criteria): correct dependencies, route coverage, API call correctness, theme adherence, navigation implementation.
- **Infra** (4 criteria): database creation, backend startup, frontend startup, environment file configuration.

The critic uses temperature 0.0 and a structured JSON response format. If the critic LLM call fails, the agent auto-passes (score = 10) to avoid blocking the pipeline.

---

## 4. Benchmark Design

### 4.1 Dataset

The benchmark uses a curated dataset of 100 full-stack web application tasks (`dataset/tasks.json`). Each task contains:

- **`id`**: A unique six-digit identifier (e.g., `000001`).
- **`instruction`**: A natural-language product brief describing the desired application, its features, and UI styling preferences.
- **`Category`**: A taxonomy with a primary category (e.g., Data Management, E-commerce) and subcategories (e.g., CRUD Operations, API Integration, Data Visualisation).
- **`application_type`**: The broad application class (e.g., Analytics Platforms/Dashboards, Social Media Applications).
- **`ui_instruct`**: A list of UI acceptance tests, each with a task description, expected result, and test category.
- **`data_structures`**: Key data entities the application must manage.
- **`backend_test_cases`**: API-level acceptance tests with instructions and expected results.

The tasks span diverse domains and complexity levels, ensuring the benchmark evaluates pattern performance across a representative workload.

### 4.2 Experimental Procedure

The benchmark runner (`benchmark.py`) executes a full factorial design: every task is run through every orchestration pattern, yielding up to $100 \times 6 = 600$ individual runs. The procedure for each run is:

1. **Isolation**: A dedicated output directory is created at `artifacts/benchmark/<task_id>/<pattern>/`, ensuring outputs from different runs cannot interfere.
2. **Logging**: A per-run file logger (`run.log`) is attached at DEBUG level, capturing all orchestrator decisions, agent outputs, LLM prompts, and error traces.
3. **Token counter reset**: The shared LLM client's accumulated counters are zeroed via `reset_usage()`.
4. **Context initialisation**: A fresh `RunContext` is created with no prior spec, transcripts, or artifacts.
5. **Pattern dispatch**: The selected orchestration pattern is instantiated and invoked with the fresh context.
6. **Metrics collection**: Upon completion (or failure), the harness reads wall-clock time, LLM usage stats, per-agent timing breakdowns, and coordinator call counts.

All runs share a single `BaseLLMClient` instance (to reuse HTTP connection pools), but token counters are reset between runs. The LLM model is held constant across all runs (configurable, default `gpt-4o-mini`) to control for model-level variance.

### 4.3 Metrics

We collect the following metrics for each run:

#### 4.3.1 Temporal Metrics

| Metric | Definition |
|--------|-----------|
| **Wall-clock time** ($T_{\text{wall}}$) | Total elapsed time from context creation to final agent completion, measured via `time.perf_counter()`. |
| **Agent total time** ($T_{\text{agent}}$) | Sum of individual agent durations: $T_{\text{agent}} = \sum_{i=1}^{n} (t_{\text{finish},i} - t_{\text{start},i})$ where $n$ is the number of agents dispatched. |
| **Orchestration overhead** ($T_{\text{overhead}}$) | Time spent on coordination logic (LLM routing calls, thread management, decision parsing): $T_{\text{overhead}} = \max(T_{\text{wall}} - T_{\text{agent}},\; 0)$. For the parallel pattern, $T_{\text{wall}} < T_{\text{agent}}$ is expected due to concurrency, in which case overhead is clamped to zero. |

#### 4.3.2 Token Usage Metrics

| Metric | Definition |
|--------|-----------|
| **LLM call count** ($C_{\text{llm}}$) | Total number of LLM API invocations, including both agent-internal calls (spec generation, code generation) and orchestrator-level calls (coordinator decisions, critic evaluations, handoff routing). |
| **Prompt tokens** ($\tau_{\text{prompt}}$) | Total input tokens sent to the LLM across all calls, as reported by the API's `usage.prompt_tokens` field. |
| **Completion tokens** ($\tau_{\text{completion}}$) | Total output tokens received from the LLM, as reported by `usage.completion_tokens`. |
| **Total tokens** ($\tau_{\text{total}}$) | Combined token consumption: $\tau_{\text{total}} = \tau_{\text{prompt}} + \tau_{\text{completion}}$. |

#### 4.3.3 Communication and Coordination Metrics

| Metric | Definition |
|--------|-----------|
| **Agent count** ($n$) | Number of agents successfully dispatched during the run. For patterns with retry logic (iterative), this may exceed 4. |
| **Coordinator calls** ($C_{\text{coord}}$) | Number of LLM calls specifically for routing/coordination (distinct from agent-internal LLM calls). Derived from `coordinator_calls` and `handoff_calls` metadata recorded by patterns. |
| **Communication overhead ratio** | Proportion of total tokens consumed by coordination: $\rho = C_{\text{coord}} / C_{\text{llm}}$. |

#### 4.3.4 Per-Agent Breakdown

For each dispatched agent, we record:
- **Role** (architect, backend, frontend, infra).
- **Status** (success or error).
- **Duration** in seconds.
- **Artifact count** (structured data objects produced).
- **Attachment count** (files written to disk).

### 4.4 Report Generation

Upon completion of all runs, the benchmark harness produces three report files:

1. **`benchmark_results.json`**: Full per-run metrics including nested per-agent breakdowns. Suitable for programmatic analysis.
2. **`benchmark_results.csv`**: A flat tabular export with 15 columns (one row per run), suitable for import into spreadsheet tools, pandas, or statistical software.
3. **`benchmark_summary.json`**: Per-pattern aggregate statistics including mean, minimum, and maximum wall-clock time; mean token counts (total, prompt, completion); mean LLM call count; mean orchestration overhead; mean agent count; and mean coordinator calls.

---

## 5. Controlled Variables

To ensure fair comparison across patterns, the following variables are held constant:

| Variable | Value |
|----------|-------|
| Agent implementations | Identical across all patterns (same 4 agents) |
| LLM model | Configurable, held constant within a benchmark run (default: `gpt-4o-mini`) |
| LLM temperature (agents) | 0.15 (backend), 0.2 (architect, frontend) |
| LLM temperature (coordinators/critics) | 0.0 |
| Agent dispatch helper | Shared `execute_agent()` function |
| Output persistence | Identical artifact writer, metadata directory structure |
| Dataset | Same 100 tasks for all patterns |
| Dependency rules | Architect first; infra after backend and frontend (encoded identically in all coordinator/handoff prompts) |

---

## 6. Independent and Dependent Variables

**Independent variable**: The orchestration pattern (6 levels: sequential, centralised, decentralised, hierarchical, parallel, iterative).

**Dependent variables**:
- Wall-clock execution time ($T_{\text{wall}}$)
- Total token consumption ($\tau_{\text{total}}$)
- Orchestration overhead time ($T_{\text{overhead}}$)
- LLM call count ($C_{\text{llm}}$)
- Coordinator call count ($C_{\text{coord}}$)
- Agent success rate
- Per-agent execution time distribution

---

## 7. Implementation Details

The system is implemented in Python 3.11+ using the following key libraries:

| Library | Purpose |
|---------|---------|
| **Pydantic v2** | ProjectSpec schema definition, validation, and JSON Schema generation |
| **Typer** | CLI framework for the `run` and `benchmark` commands |
| **Rich** | Formatted console output, logging, and results tables |
| **httpx** | HTTP client for OpenAI API calls |
| **python-dotenv** | Environment variable loading from `.env` files |
| **concurrent.futures** | `ThreadPoolExecutor` for the parallel pattern |

Generated projects use:
| Technology | Role |
|------------|------|
| **Express.js** | Backend API framework |
| **React + Vite** | Frontend SPA framework and build tool |
| **MySQL (mysql2)** | Database with migration support |
| **Jest + supertest** | Backend unit testing |
| **Vitest + React Testing Library** | Frontend unit testing |
| **Tailwind CSS** | Frontend styling |

The full source code, dataset, and benchmark harness are available in the project repository.

---

## 8. Limitations

- **LLM non-determinism**: Despite using low temperatures (0.0 for coordinators, 0.15–0.2 for agents), LLM outputs are inherently stochastic. Multiple runs of the same (task, pattern) pair may yield different results.
- **Single LLM provider**: The evaluation uses a single model (OpenAI `gpt-4o-mini`). Results may not generalise to other models or providers.
- **Dummy LLM fallback**: When no API key is configured, the system uses a deterministic placeholder that produces non-functional code. Benchmark results are only meaningful with a real LLM.
- **Infrastructure side effects**: The infra agent creates real MySQL databases and starts Node.js processes, requiring a local development environment. This introduces external variability (database latency, npm registry speed).
- **Coordinator call tracking**: The `coordinator_calls` metric relies on patterns explicitly recording these counts in report metadata. Patterns that do not propagate this metadata will report zero coordinator calls even if coordination LLM calls occurred; however, such calls are still captured in the total `llm_call_count`.
- **No functional correctness evaluation**: The current metrics focus on process efficiency (time, tokens, overhead). Evaluating whether the generated code is *functionally correct* (e.g., tests pass, endpoints behave as specified) is left for future work.
