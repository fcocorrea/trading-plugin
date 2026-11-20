# Chapter 24: Autonomous Agents

## Core Idea
Agents are treated as engineering systems (not chatbots) for decision support: they gather evidence, use tools, maintain state, and produce replayable, auditable artifacts under bounded, read-only autonomy (no order execution; that is Ch 25). The value comes from evidence + tool contracts + provenance + point-in-time integrity + evaluation, not from replacing statistical models.

## Frameworks Introduced
- **5-phase agent loop**: Perception/Reasoning, Planning, Action, Observation, with tools and memory. "Action" = information action (data API, index query, artifact write), never order execution.
  - When to use: evidence-rich tasks (event forecasting, due diligence, research support). Do not use when labels are stable and structured, the task is latency-critical (microseconds), or it does not require external retrieval.
- **Autonomy levels L0-L4**: L0 passive summary; L1 decision support with probabilities; L2 restricted actions with human approval; L3-L4 increasing autonomy. Practical deployments sit at L1-L2 (a forecast error is expensive, an execution error without policy is catastrophic). The notebooks operate at L1.
- **ReAct (Thought, Action, Observation)**: baseline for evidence-anchored tasks; every claim is linked to a tool call.
  - How: iteration limits, evidence-sufficiency checks and state validators.
  - Why it works / failure mode: traceability/audit trail. Failures: repetitive tool loops, premature synthesis (a single observation), brittle behavior from vague schemas, unreconciled hidden contradictions.
- **Tree of Thoughts (ToT)**: explores candidate branches in parallel (e.g., hawkish vs dovish ahead of a rate decision) before choosing. Cost = multiplier by branching factor + state management. Reserve it for strategic decisions where committing late changes the decision.
- **Reflexion**: post-run critique with compact lessons; requires provenance, validity horizon, decay/pruning and rollback; without them it turns temporary heuristics into durable biases.
- **Composition rule**: ReAct baseline; ToT only in branch-heavy decisions (e.g., conflicting observations); Reflexion only with a stable evaluation pipeline. Shallow composition (one layer of each); define a per-run reasoning budget (cap on tool calls, branches, reconciliation rounds) and return a bounded output with explicit uncertainty when it is exhausted.
- **Memory hierarchy**: working (step context), short-term (session), long-term (indexes, artifacts, outcomes, calibration history). Typed state: run/question id, as_of_date, cutoff_date, evidence with metadata, open sub-tasks, tool-call trace, intermediate probabilities, artifact status (draft/review/final).
- **Checkpoint + replay**: checkpoints at decision boundaries (after initial evidence, after resolving contradictions, before final synthesis, before a high-impact handoff). Replay: freeze tool outputs, restore state, rerun with new prompts/policies, compare trace/eval deltas, accept only if metrics improve.
  - Failure mode: confusing evidence drift with prompt improvements.
- **Tool contract**: purpose/scope, typed arguments with ranges, error semantics (transient vs permanent), provenance fields. The description states what it returns, when to call it and when not to.
- **Context engineering**: each step exposes only the state fields, tools and sources relevant to its phase, and requires an output schema.
- **Source policy**: domain allowlists + date constraints tied to the cutoff, enforced at the tool boundary (do not trust the model to police itself).
- **Migration sequence**: baseline with a native SDK and typed schemas; add explicit state and traces; checkpoints/replay; multi-agent only if evaluation shows a measurable improvement over single-agent.
- **AIA-style multi-agent pipeline (6 layers)**: intake, research (parallel agents with the same cutoff), aggregation (+optional debate), supervisor (override only if confidence exceeds a policy threshold), calibration, persistence/scoring.
- **Extremization (Neyman)**: p_extreme = base + d*(p_mean - base); d = sqrt(n / (1 + (n-1)*rho)), n = forecasters, rho = average pairwise correlation. rho=0 gives d = sqrt(n) (aggressive); rho=1 gives d = 1 (passes the mean through).
  - Failure mode: Platt scaling on top of extremization compounds overconfidence unless the calibration window is disjoint.
- **Calibration workflow (Platt scaling)**: post hoc logistic transformation; reserve a resolved historical window, fit only there, evaluate on a disjoint holdout, monitor drift with a refit cadence, version the model.
- **Retention rubric**: does it improve resolved-outcome metrics? Is it stable across subsets? Is the cost/latency acceptable? Does it increase policy risk or fragility? If not, demote or remove it.
- **Operator pattern (ML4T research agent)**: a small orchestrator (~880 lines, 10 tools: read/write files, bash, SQLite registry, Parquet, list_skills, read_skill) that hands general tools to the LLM; the methodology lives in libraries and in a skills repo (56 Markdown files in 9 categories: problem, WRONG/CORRECT pair, Production Implementation) consulted on demand.
- **Warden pattern**: policy proxy between agent output and execution: validates against an allowlist, blocks patterns, validates arguments/ranges, timeouts, immutable allow/deny logs.

## Key Concepts
- **Information action**: a sandboxed read-only/compute/artifact-write action.
- **Point-in-time discipline (as_of_date / cutoff_date)**: a cutoff that propagates to every tool call.
- **Quality gates**: coverage (required evidence types), freshness (time window), consistency (ticker mismatch, post-cutoff leakage).
- **Forced default**: a forecast recorded with explicit uncertainty when the step limit is exhausted (default 5).
- **AgentForecastArtifact / AgentTrace**: artifact with probability, confidence (inferred |p_yes - 0.5|), sentiment, evidence quality, traces and tokens.
- **Model cascading**: route sub-tasks to the cheapest model that meets quality (extraction/formatting to small models; synthesis and supervision to frontier models).
- **Brier / log score / ECE / sharpness**: probability quality metrics; sharpness is only desirable with acceptable calibration.
- **Contamination**: the LLM memorizes pre-cutoff outcomes; pre-cutoff evaluation is not identifiable (Lopez-Lira, Tang, Zhu 2025).
- **Retrieval poisoning / tool shadowing / prompt injection**: attacks on the corpus, on tool identity and on instructions.
- **MCP (Model Context Protocol)**: standard for tool access; adopt selectively.

## Mental Models
- Use ReAct when you need auditability and sequential evidence; ToT only if branches change the decision; Reflexion when evaluation is already stable.
- Use debate/specialized roles when the question is two-sided; on one-directional questions they add cost without changing the answer.
- Use a single tool (search) with a 2-action schema (search/forecast) to isolate reasoning quality from routing quality.
- Use this debugging order: tool contracts and search quality, then JSON parsing/extraction, then prompt wording.
- Use the Investment Policy Statement as the operating boundary of the multi-agent pipeline (Ang et al. 2026), instead of inventing a governance document.
- Use identical uniformity across agents as an alarm (broken temperature or a bug), not similarity on questions with one-directional evidence.

## Anti-patterns
- **Chat history as state**: mixes instructions, evidence and tool outputs; not replayable.
- **Sending the full accumulated context to every step**: wastes tokens and adds noise.
- **Adopting multi-agent before a single-agent baseline**: complexity without evidence of benefit.
- **Relying on framework defaults for security**, postponing observability, treating framework migration as a substitute for evaluation.
- **Long-term memory without validation/expiration**: accumulates obsolete assumptions.
- **Platt scaling over extremized aggregation without a holdout**: overconfidence.
- **Celebrating improved ECE with collapsed sharpness**; recalibrating too often without drift; calibrating with few events.
- **Declaring a "win" on a single panel**: contamination and leakage; evaluate on disjoint windows, report ranges.
- **Declaring outperformance vs the market on unresolved questions**: both are estimates.
- **Operator failures**: confabulated improvement, done() before finishing, overinterpreting inconclusive results.
- **Excessive write/exec permissions in a read-only runtime**.

## Code Examples
Minimal agent state (reconstructed from the description of 03_state_and_memory):

```python
@dataclass
class AgentState:
    run_id: str
    question: str
    cutoff_date: date
    evidence: list[EvidenceRecord]
    tool_trace: list[ToolCall]
    open_subquestions: list[str]
    gate_results: dict[str, bool]
    synthesis_status: str
```
What it demonstrates: typed state with cutoff and gates that enables JSON serialization, checkpoint round-trip and replay.

## Reference Tables
**Reasoning frameworks (Table 24.1)**

| Framework | Mechanism | Strength | Best for |
|---|---|---|---|
| ReAct | Thought-Action-Observation | grounded, auditable | interactive research, due diligence |
| Tree of Thoughts | parallel exploration | strategic depth | scenarios, portfolio alternatives |
| Reflexion | post-run critique + memory | adaptive behavior | iterative refinement |

**Design-choice matrix (Table 24.2)**

| Need | Style | Fit |
|---|---|---|
| Minimal overhead, full control | Native SDK + typed schemas | small agents |
| Fast ReAct/tool | lightweight framework | prototypes |
| Retrieval-centric | retrieval-first abstractions | RAG-heavy |
| Explicit, durable state | state-graph with persistence/replay (LangGraph-style) | production pipelines |
| Multi-role collaboration | multi-agent orchestration (CrewAI-style) | specialized tasks |

**Security controls**

| Category | Control | Verification |
|---|---|---|
| Input | sanitization | adversarial suite |
| Permissions | least privilege, scoped credentials | per-deployment audit |
| Retrieval | allowlist + dates | post-cutoff attempt rate |
| Execution | Warden proxy | review of the block log |
| High-stakes | HITL above thresholds | approval queue latency |
| Observability | full tracing | completeness checks |
| Audit | immutable logs | integrity verification |

**Security metrics**: prompt-injection success rate (ideal 0), unsupported-claim rate, policy-violation attempt rate, blocked privileged-call rate, false-positive rejection rate, mean time to incident diagnosis. **Evaluation**: agent quality (task success, citation faithfulness, tool-call validity, abstention, cost/latency), forecasting (Brier, log, ECE, reliability, sharpness), operational (incident rate, policy violations, replay pass/fail, drift alerts).

## Worked Example
Chapter ablation and runs. (a) US recession by end-2026: three Claude Sonnet 4 agents gave 0.12, 0.22, 0.22 vs a market price of 0.175 (one-directional evidence, agreement expected). Fed hike 2026 (two-sided): 0.35, 0.58, 0.38 and a bull/bear gap of ~0.4 with no consensus in 3 rounds. (b) Operator, ETFs case: ensembling GB + tabular DL + conv autoencoder via cross-sectional z-score gave IC 0.065 vs 0.052 for the LSTM, but validation Sharpe 0.56 vs 0.92; 3 of 9 folds with negative IC amplified by the top-k score-weighted allocator; the registry only has the LSTM's holdout, so it cannot be promoted; it reported a clean negative (39 turns, ~$1.25). (c) US firm characteristics case: filtering to the top 3 market-cap quartiles cut Sharpe from 4.27 to 2.24 (-48%), MaxDD from -15% to -52%, IC 0.074 to 0.048, turnover unchanged; the alpha persists (PBO near 0) but ~half of the Sharpe is small-cap residual (27 turns, ~$0.95). Warden: 4 calls, 2 benign ones pass, "hack admin credentials" and a trade order are blocked; 5 injection payloads detected.

## Key Takeaways
- Keep the agent read-only and at L1-L2; order execution belongs to another layer with stricter controls.
- Start with ReAct, a step limit (default 5), a minimal action schema and a forced default with explicit uncertainty.
- Model typed state with cutoff_date propagated to the tool boundary; checkpoints at decision boundaries and replay with frozen evidence.
- Evaluate with Brier/log/ECE/sharpness and ablations against a market baseline, on time-disjoint windows; contamination takes precedence over demo performance.
- Calibrate with disjoint windows and take care when combining with extremization.
- Apply least privilege, Warden, threshold-based HITL, sanitization/allowlists, immutable logs, and separate observability views (engineering vs research).
- Use model cascading and caching with cutoff-aware keys; an optimization that degrades calibration is not valid.

## Connects To
Ch 22 (RAG, MCP, ReAct, FinDER); Ch 21 (RL, policy learning); Ch 20 (case studies, Sec 20.9 next experiments); Ch 16-19 (simulation/backtest); Ch 25 (live trading, execution); Ch 26 (MLOps/governance); deflated Sharpe, PBO, point-in-time.

## Code Reference

Official repository: [24_autonomous_agents/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/24_autonomous_agents). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_react_reasoning.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/01_react_reasoning.ipynb) + `.py` | 24.1, 24.2 | Replays a captured ReAct session and exposes provider selection, bounded search, and raw message traces. |
| [`02_tool_contracts.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/02_tool_contracts.ipynb) + `.py` | 24.1, 24.4 | Demonstrates typed tool schemas, validated arguments, provenance, and returned-domain filtering with a labeled synthetic search fixture. |
| [`03_state_and_memory.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/03_state_and_memory.ipynb) + `.py` | 24.1, 24.3 | Demonstrates explicit state, checkpoint round-trips, and coverage, freshness, and consistency gates on labeled teaching fixtures. |
| [`04_research_agent.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/04_research_agent.ipynb) + `.py` | 24.1, 24.6 | Accepts a finite probability only with retained evidence and a cited rationale. |
| [`05_aggregation_math.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/05_aggregation_math.ipynb) + `.py` | 24.1, 24.7 | Demonstrates aggregation, linear Neyman and log-odds extremization, and held-out Platt fitting on labeled mathematical examples. |
| [`06_multi_agent_research.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/06_multi_agent_research.ipynb) + `.py` | 24.7 | Replays parallel research agents and compares aggregation sensitivity. |
| [`07_adversarial_debate.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/07_adversarial_debate.ipynb) + `.py` | 24.7 | Replays bull/bear debate, tracks the probability gap, and demonstrates consensus detection. |
| [`08_forecasting_pipeline.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/08_forecasting_pipeline.ipynb) + `.py` | 24.1, 24.7 | Composes accepted-agent averaging, optional debate, high-confidence cited supervisor updates, and the fixed AIA log-odds correction. |
| [`09_evaluation_and_governance.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/09_evaluation_and_governance.ipynb) + `.py` | 24.1, 24.7, 24.9, 24.10 | Preserves and recovers a historical panel, scores eight of ten questions, and distinguishes proper scores from diagnostics. |
| [`10_framework_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/10_framework_comparison.ipynb) + `.py` | 24.1, 24.5, 24.7 | Executes native Python and LangGraph over shared stages, including a synthetic abstention test. |
| [`11_research_operator.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/24_autonomous_agents/11_research_operator.ipynb) + `.py` | 24.8 | Replays two retained operator experiments over case-study artifacts. |
