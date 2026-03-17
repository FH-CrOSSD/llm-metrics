# CrOSSD Health Analyser

A [LangGraph](https://github.com/langchain-ai/langgraph)-powered pipeline that evaluates the health of open-source GitHub repositories using local LLMs via [Ollama](https://ollama.com/).

Repository data is fetched from the [CrOSSD API](https://health.crossd.tech) (or loaded from a local JSON export) and scored across a configurable set of metrics using either a single-agent or multi-agent debate strategy.

---

## How it works

```
CrOSSD API / local JSON
        │
        ▼
  PipelineState
        │
        ▼ (conditional fan-out via Send)
  ┌─────┴──────┐
  │            │
run_single  run_debate
  │            │
  │     agent_turn ◄─┐
  │            │     │ (loop until all rounds done)
  │     _should_synthesize?
  │            │
  │       synthesize
  │            │
  └────────────┘
        │
  results (merged via reducer)
        │
        ▼
  JSON report / CLI output
```

**`single` pipeline** — one LLM call scores the metric directly and returns a JSON object.

**`debate` pipeline** — N configurable agents take turns in round-robin order for K rounds. Each agent sees only a short, score-free summary of its peers' most recent arguments (to prevent social-contagion score drift). A synthesizer then reads the full transcript and emits the final score, anchored to the Python-computed median of all stated estimates.

### LangGraph topology

| Level | Structure |
|---|---|
| Top-level graph | `START` → conditional fan-out (`Send`) → `run_single` / `run_debate` → `END` |
| Single sub-graph | `START → run_single → END` |
| Debate sub-graph | `START → agent_turn ⇄ (conditional) ⇄ synthesize → END` |

Results from parallel branches are merged into `PipelineState.results` via an `Annotated` reducer.

---

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com/) running locally (default: `http://localhost:11434`)
- A pulled Ollama model (default: `gemma3:27b`)

### Python dependencies

```
langgraph>=1.0
requests>=2.31
```

Install with:

```bash
pip install langgraph requests
```

---

## Quick start

```bash
# Analyse a repo from the CrOSSD API (newest snapshot)
python main.py lorabridge/lorabridge

# Use a specific crawl-snapshot timestamp
python main.py lorabridge/lorabridge --timestamp 1758139805.6506386

# Analyse from a local JSON export
python main.py --file example.json

# Run only specific metrics
python main.py google/flax --metrics friendliness documentation_quality

# Use a different Ollama model or host
python main.py google/flax --model llama3.1:8b --llm-host http://localhost:11434

# Output raw JSON instead of a formatted report
python main.py google/flax --json

# Save debate logs to a custom directory
python main.py google/flax --log-dir my_logs/
```

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `repo` | — | GitHub slug, e.g. `owner/repo` |
| `--file`, `-f` | — | Path to a local JSON export (alternative to live fetch) |
| `--metrics`, `-m` | all | Space-separated list of metric names to evaluate |
| `--model` | `gemma3:27b` | Ollama model name |
| `--llm-host` | `http://localhost:11434` | Ollama API base URL |
| `--crossd-host` | `https://health.crossd.tech/api` | CrOSSD API base URL |
| `--timestamp`, `-t` | newest | Crawl-snapshot timestamp |
| `--json`, `-j` | false | Emit raw JSON output |
| `--log-dir` | `debate_logs/` | Directory for saved debate transcripts |

---

## Metrics

Metrics are declared in `config.py` and can be freely added, removed, or reordered.

| Name | Pipeline | Agents | Description |
|---|---|---|---|
| `friendliness` | single | analyst | Tone and welcomingness of developer interactions |
| `documentation_quality` | single | analyst | README, contributing guides, issue/PR templates |
| `development_efficiency` | debate (2 rounds) | analyst, reviewer | Issue/PR turnaround, branching, release cadence |
| `project_maturity` | debate (3 rounds) | analyst, developer advocate, user advocate, security auditor, reviewer | Holistic maturity across governance, community, CI/CD, security |

### Adding a metric

```python
# config.py
from config import MetricDefinition, AGENT_ANALYST, AGENT_REVIEWER

DEFAULT_METRICS.append(
    MetricDefinition(
        name="test_coverage_culture",
        display_name="Test Coverage Culture",
        description="Evidence of automated testing practices in the repository.",
        prompt=(
            "Evaluate the testing culture of this repository. "
            "Look for test directories, CI configuration, and PR comments about tests. "
            "Give a score from {min} to {max} where {min} is no evidence of testing "
            "and {max} is a comprehensive, enforced testing culture."
        ),
        pipeline="debate",
        data_keys=["commits", "pulls", "community_profile"],
        debate_rounds=2,
        debate_agents=[AGENT_ANALYST, AGENT_REVIEWER],
    )
)
```

### Built-in debate agents

| Constant | Role |
|---|---|
| `AGENT_ANALYST` | Meticulous software analyst — objective, evidence-grounded |
| `AGENT_REVIEWER` | Peer reviewer — challenges and cross-checks other agents |
| `AGENT_USER_ADVOCATE` | End-user perspective — usability, onboarding, barriers to entry |
| `AGENT_DEVELOPER_ADVOCATE` | Maintainer perspective — engineering practices, resource constraints |
| `AGENT_SECURITY_AUDITOR` | Security posture — advisories, dependency hygiene, disclosure process |

---

## Project structure

```
├── main.py          # CLI entry-point
├── config.py        # Metric definitions and debate agent presets
├── graph.py         # LangGraph pipeline (single & debate sub-graphs)
├── agents.py        # Ollama LLM wrapper and prompt templates
├── rag.py           # Context retrieval from the repo JSON
└── debate_logs/     # Saved debate transcripts (JSON, auto-generated)
```

---

## Debate logs

Every `debate` metric run writes a structured JSON transcript to `debate_logs/` (configurable via `--log-dir`):

```
debate_logs/project_maturity_google_flax_1765317472.json
            <metric>_<owner>_<repo>_<timestamp>.json
```

Each file contains:
- `agents` — list of participant names
- `debate_rounds` — number of rounds completed
- `final_score` — the synthesizer's output
- `transcript_median` — Python-computed median of all stated estimates (used to anchor the synthesizer)
- `log` — full turn-by-turn transcript

---

## Score integrity safeguards

The pipeline includes several mechanisms to keep scores grounded in evidence:

1. **Peer summaries strip scores** — agents see only short prose snippets of peers' arguments, never their numeric estimates, preventing herd-following.
2. **Transcript median** — after all rounds, the Python-computed median is injected into the synthesizer prompt as an explicit anchor.
3. **Hard clamp** — the final score is always clamped to the declared `scale` range regardless of LLM output.
4. **JSON fallback chain** — if the synthesizer produces malformed output, the pipeline falls back to the transcript median rather than returning `null`.
