# From Fixed Role Pipelines to Iterative Multi-Agent Software Development

## A Small Proposed Adjustment to Our Research Direction

### Motivation

Our initial experiments have raised a methodological question that may be more important than the comparison among the orchestration patterns themselves: are the current tasks and agent interactions sufficiently demanding to expose meaningful differences between orchestration strategies?

Using a strong model, currently Gemini 3.1 Pro, we observed that a single full-stack agent could successfully complete the benchmark tasks we tested. The multi-agent configurations also completed many of the same tasks, but they frequently converged to nearly the same execution sequence: Architect, Backend, Frontend, Infrastructure, and, only when needed, Fixer. Centralized, decentralized, sequential, and parallel configurations therefore appeared different at the controller level, but their realized trajectories were often very similar. This made the final outcomes broadly equivalent and left relatively little evidence that the orchestration pattern itself was responsible for success.

This does not mean that multi-agent orchestration is ineffective. A more cautious interpretation is that the combination of a highly capable model, relatively tractable tasks, and a strongly predefined development workflow leaves little room for the orchestration strategy to matter. If one model can solve the task alone, additional agents may primarily introduce coordination cost. Likewise, if every configuration begins with the same global architectural decomposition and assigns work to the same specialist roles, nominally different patterns may still implement almost the same decision process.

We briefly considered moving directly to substantially harder benchmarks. In particular, we examined benchmarks based on larger projects and sequences of evolving tasks, including settings in which a project changes over multiple requirement increments. These benchmarks remain promising, especially for studying long-term adaptation and technical debt. However, adopting one of them immediately would introduce considerable implementation and evaluation work. It could also change several variables at once: repository scale, task format, build environment, evaluation procedure, context length, and orchestration mechanism. A closer and more controlled next step may be to retain our current benchmark initially, vary model capability, and make the internal development process more genuinely iterative.

### What Our Current Results Suggest

The strong single-agent result provides a useful starting point rather than a negative result. Gemini 3.1 Pro can serve as a high-capability baseline: it approximates a setting in which model competence is sufficient for the tested tasks and coordination is not required for basic task completion. The questions of whether multi-agent systems improve quality, introduce coordination overhead, or trade cost for performance are already central to this research area and overlap with research questions considered in the study that motivated our work. We should therefore not present them as new research questions by themselves.

The proposed extension is to make **model capability a systematic experimental dimension** in the comparison of orchestration patterns. Our current results describe one point in that space: a highly capable model operating on tasks that it can often solve as a single agent. We can add faster, less expensive, and less capable models while holding the tasks, tools, budgets, and orchestration protocols constant. This would allow us to examine whether the relative effect of orchestration changes as individual-agent capability decreases. The objective is not simply to demonstrate that a group of smaller models can replace a stronger model, but to identify whether there is a capability region in which specialization, review, or task decomposition becomes more consequential than the coordination overhead they introduce. Throughput remains part of this dimension because smaller models may generate more tokens per second and support more concurrent calls, even when each call is individually less capable.

The experiment should therefore include at least three capability tiers where API availability permits: a strong model as the quality baseline, a mid-sized model, and a smaller or faster model. All orchestration configurations within a tier should use the same tools, limits, and evaluation procedure. We should measure task success, test results, wall-clock time, prompt and completion tokens, API cost, number of agent turns, repair iterations, and duplicated or reverted work. The most interesting result may be a Pareto frontier rather than a single winner: different configurations may optimize quality, cost, or time.

### Relationship to the Original Study

Our observations resemble the findings of Rizk, Khatoonabadi, and Shihab in *Bridging Design and Implementation: A Study of Multi-Agent LLM Architectures for Automated Front-End Generation*. That study organizes front-end generation around Builder, Validator, and Fixer agents and compares Supervisor with tool-calling, Hierarchical, and Custom deterministic architectures. Although the controllers differ, all three execute variants of an underlying Build–Validate–Fix process. The authors report only modest quality differences across architectures but substantial differences in token consumption. The deterministic Custom architecture achieves the best cost–performance trade-off, while more elaborate coordination duplicates context without producing a corresponding quality improvement.

Our implementation extends the domain from multimodal front-end generation to full-stack project generation and introduces Architect, Backend, Frontend, Infrastructure, and Fixer roles. Nevertheless, it retains an important property of the original design: the overall development process is largely known in advance. In practice, the Architect is not only an architectural specialist but also the global planner. It interprets the entire project, decomposes the work, and produces an artifact that conditions the behavior of all downstream agents. Consequently, even the decentralized configuration begins from a centralized account of the problem. Parallelism changes when predefined work executes, but not necessarily how the work is discovered or revised.

This helps explain why our patterns converged. They use different routing mechanisms around a shared decomposition. In this sense, our preliminary results are compatible with the original paper: when the task has a clear dependency chain and the agents perform similar amounts of generation, streamlined coordination may be as effective as more complex orchestration. The main difference may appear in efficiency rather than quality.

### Are the Current Components Actually Acting as Agents?

A second concern is the granularity of agency in the current implementation. Our specialized components often receive a broad assignment, generate a large amount of code in one call, and pass the result to the next component. They have limited opportunity to inspect the repository incrementally, run tools, observe failures, revise their own code, and communicate discoveries before committing to a large solution.

This behavior is closer to a pipeline of role-conditioned LLM generators than to an iterative software engineering agent. A coding agent would normally alternate among inspection, planning, editing, testing, and revision. It would use the result of each tool call to decide what to do next. Human development is similarly cyclic: architects, backend engineers, frontend engineers, and testers do not normally complete isolated blocks of work once and then disappear. Requirements and designs are revised as implementation reveals constraints.

The importance of the Fixer in our system may partly follow from this coarse interaction model. When an agent produces a large code artifact without being able to compile, test, and repair it incrementally, defects accumulate and must be delegated to a downstream repair stage. The Fixer is valuable as an independent verification mechanism, but it may also be compensating for insufficient self-validation by the generating agents. A useful experiment would distinguish these functions: all implementation agents should be able to test and revise their own changes, while an independent validator or fixer remains available for issues they fail to detect or resolve.

### Proposed Adjustment: Development as a Sequence of Small Cycles

We propose a small change in direction: preserve the comparison of orchestration strategies, but redefine the unit of agent work as a small, testable development cycle rather than a complete role-level deliverable. A cycle would consist of inspecting the current state, selecting a small step, implementing it, validating it, recording discoveries, and proposing or scheduling the next step. The output of a cycle would include code changes, validation evidence, unresolved issues, and suggested follow-up tasks.

This creates an internal task trajectory similar to staged software engineering benchmarks without requiring an immediate migration to a new dataset. A large benchmark requirement can be decomposed into internal subtasks, while datasets containing multiple evolving requirements can later provide external project cycles. The distinction is useful: benchmark tasks describe what changes over the lifetime of the project, whereas agent-created subtasks describe how the team chooses to implement each change.

The three principal coordination strategies would then have genuinely different decision rules:

1. **Centralized coordination.** A Manager inspects the project, creates an initial backlog of small deliverables, identifies dependencies, assigns agents, and continuously revises the plan from execution feedback. The Manager may call the same specialist multiple times. It is the only component with authority over the global backlog.

2. **Decentralized coordination.** No agent creates an authoritative plan for the whole project. An initial agent is selected, chooses a small step for itself, executes and validates it, and recommends both the next step and the next agent. The plan emerges through peer-to-peer handoffs. An agent may call another specialist or return work to a previous specialist when new information requires it.

3. **Sequential coordination.** Agents operate in a fixed or rotating order over a shared backlog. Each agent selects and performs a small step, leaves observations and suggestions, and hands control to the next position in the sequence. Unlike the centralized strategy, there is no permanent manager; unlike the decentralized strategy, the current agent does not freely select its successor. This provides an intermediate form of coordination with distributed planning but controlled scheduling.

Parallelism should be treated as a separate experimental factor rather than as a fourth organizational strategy. Each of the three strategies can run in a serial mode or a parallel-enabled mode. In centralized–parallel execution, the Manager may dispatch independent backlog items concurrently. In decentralized–parallel execution, an agent may create a bounded number of independent handoffs. In sequential–parallel execution, agents can mark independent work for a batch that executes between synchronization barriers. This yields a clear factorial comparison between coordination structure and concurrency policy.

A single iterative agent remains essential as the baseline. It should receive the same repository access, tools, total budget, and validation capabilities as the multi-agent systems. Otherwise, an observed multi-agent advantage could simply result from granting the team more tools or more inference. Planning and coordination tokens must also count toward the total cost.

### Suggested Research Questions and Next Step

This adjustment supports a focused set of research questions:

- How does model capability affect the relative value of single- and multi-agent development?
- Can coordinated smaller models approach the quality of a strong single-model baseline at lower cost or latency?
- Does global planning outperform emergent task decomposition when implementation reveals unexpected constraints?
- Does parallelism improve only elapsed time, or can it improve quality through independent exploration?
- How much overhead arises from duplicated context, coordination messages, conflicting changes, and rework?
- When agents can test and revise their own work, how often is an independent Fixer still needed?

The proposed change is deliberately incremental. We do not need to abandon the current pipeline or immediately adopt a much larger benchmark. The current implementation can remain a fixed-workflow baseline closely related to the original study. We can add an iterative execution protocol, begin with centralized and decentralized serial variants plus the single-agent baseline, and validate the design on a small subset of existing tasks. Once the trajectories are observably different and the instrumentation is reliable, we can cross the coordination strategies with parallelism, model capability, and eventually a staged-project benchmark.

The intended contribution would therefore move from a simple ranking of orchestration patterns toward identifying the conditions under which orchestration becomes valuable. The central question is not whether multi-agent systems are universally better, but where their coordination overhead is justified by task complexity, model limitations, iterative feedback, or opportunities for parallel work. Our preliminary single-agent and pattern-convergence results provide a concrete motivation for this refinement and a strong baseline against which it can be evaluated.

## Reference

C. Rizk, S. Khatoonabadi, and E. Shihab, “Bridging Design and Implementation: A Study of Multi-Agent LLM Architectures for Automated Front-End Generation,” *Proceedings of the 23rd International Conference on Mining Software Repositories (MSR 2026)*, 2026. DOI: [10.1145/3793302.3793371](https://doi.org/10.1145/3793302.3793371). [Author-hosted paper](https://das.encs.concordia.ca/pdf/rizk_MSR2026.pdf).
