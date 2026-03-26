"""
LangGraph pipeline for repository health analysis.

Two sub-graphs, selected per metric by the ``pipeline`` field:

  * **single** – one LLM call produces the score directly.
  * **debate** – N configurable agents take turns for K rounds → synthesizer.

The top-level graph uses LangGraph's Send API to fan-out across all requested
metrics in parallel.  Each metric runs in its own sub-graph whose internal
debate loop is also modelled as a proper LangGraph conditional-edge cycle.

Graph topology
──────────────
Top-level:
  START → fan_out → [per-metric sub-graph via Send] → collect_results → END

Single sub-graph (per metric):
  START → run_single → END

Debate sub-graph (per metric):
  START → agent_turn → should_synthesize?
                          ├─ "continue" → agent_turn   (round-robin loop)
                          └─ "synthesize" → synthesize → END
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from llm_metrics.agents import (
    SINGLE_AGENT_SYSTEM,
    call_llm,
)
from llm_metrics.config import DEFAULT_METRICS, DebateAgent, MetricDefinition
from llm_metrics.rag import build_repo_summary, retrieve_context

# ═══════════════════════════════════════════════════════════════════════════
# State definitions
# ═══════════════════════════════════════════════════════════════════════════


class MetricResult(TypedDict, total=False):
    name: str
    display_name: str
    metric: float | None
    explanation: str
    pipeline: str
    debate_log: list[dict[str, str]]


def _merge_results(left: list[MetricResult], right: list[MetricResult]) -> list[MetricResult]:
    """Reducer: accumulate per-metric results sent back from parallel sub-graphs."""
    return left + right


class PipelineState(TypedDict, total=False):
    """Top-level state shared across the full graph."""
    repo_data: dict[str, Any]
    metrics: list[MetricDefinition]
    results: Annotated[list[MetricResult], _merge_results]
    llm_model: str
    llm_host: str
    llm_auth_token: str | None
    # Provenance – used to name saved debate logs
    repo_slug: str          # e.g. "lorabridge/lorabridge"
    snapshot_timestamp: float | None  # crawl timestamp, or None for local files
    log_dir: str            # directory where debate logs are written


# ── Single-metric state (used inside per-metric sub-graphs) ────────────────

class SingleMetricState(TypedDict, total=False):
    """Scoped state for evaluating one metric."""
    metric: MetricDefinition
    repo_data: dict[str, Any]
    llm_model: str
    llm_host: str
    llm_auth_token: str | None
    repo_slug: str
    snapshot_timestamp: float | None
    log_dir: str
    result: MetricResult


# ── Debate state (used inside the debate sub-graph loop) ──────────────────

class DebateState(TypedDict, total=False):
    """Scoped state for one debate run."""
    metric: MetricDefinition
    repo_data: dict[str, Any]
    llm_model: str
    llm_host: str
    llm_auth_token: str | None
    repo_slug: str
    snapshot_timestamp: float | None
    log_dir: str
    # Debate runtime
    debate_log: list[dict[str, str]]
    current_round: int       # 0-based round index
    current_agent_idx: int   # index into metric.get_debate_agents()
    result: MetricResult


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_response(text: str) -> dict[str, Any]:
    """Best-effort extraction of the JSON object from an LLM response.

    Attempts, in order:
      1. Direct JSON parse of the full text.
      2. JSON object inside a markdown code fence.
      3. First ``{…}`` block found anywhere in the text.
      4. Numeric fallback: scan prose for an explicit score mention.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    score_match = re.search(
        r"(?:score|rating|metric|give(?:\s+this)?(?:\s+a)?|rank)\D{0,10}(\d+(?:\.\d+)?)"
        r"|(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
        r"|\b(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if score_match:
        raw = next(g for g in score_match.groups() if g is not None)
        return {"metric": float(raw), "explanation": text}
    return {"metric": None, "explanation": text}


def _build_user_prompt(metric: MetricDefinition, context: str) -> str:
    prompt_text = metric.prompt.format(min=metric.scale[0], max=metric.scale[1])
    return (
        f"Metric: {metric.display_name}\n"
        f"Description: {metric.description}\n"
        f"Scale: {metric.scale[0]}–{metric.scale[1]}\n\n"
        f"Instructions: {prompt_text}\n\n"
        f"Repository data:\n{context}"
    )


def _build_scoring_user_prompt(metric: MetricDefinition, context: str) -> str:
    """User prompt for agents that must output final JSON (single + synthesizer)."""
    base = _build_user_prompt(metric, context)
    return (
        base + "\n\n"
        "IMPORTANT: your entire response must be a single JSON object with no "
        "surrounding text, markdown, or code fences. Required format:\n"
        f'{{"metric": <number {metric.scale[0]}–{metric.scale[1]}>, "explanation": "<string>"}}'
    )


def _extract_stated_score(text: str) -> float | None:
    """Pull the *last* explicit score mention from an agent turn."""
    matches = re.findall(
        r"(?:current estimate|my estimate|i.{0,8}give|i.{0,8}rate|i.{0,8}score|score|rating|estimate)"
        r"[^\d]{0,15}(\d+(?:\.\d+)?)"
        r"|(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    for prose_val, slash_val in reversed(matches):
        raw = prose_val or slash_val
        if raw:
            return float(raw)
    return None


def _extract_peer_summaries(
    debate_log: list[dict[str, str]],
    current_agent_name: str,
    agents: list[DebateAgent],
) -> str:
    """Return a short bullet-point summary of each OTHER agent's most recent turn.

    Score estimates are intentionally omitted to prevent social-contagion
    cascades.  Only the synthesizer receives the complete transcript.
    """
    name_to_display = {a.name: a.display_name for a in agents}

    latest: dict[str, str] = {}
    for entry in debate_log:
        role = entry["role"]
        if role != current_agent_name and role in name_to_display:
            latest[role] = entry["content"]

    if not latest:
        return ""

    lines = ["── Key points from other participants (most recent turn, no scores shown) ──"]
    for role, content in latest.items():
        display = name_to_display.get(role, role)
        stripped = re.sub(
            r"(?:my current estimate|current estimate|score estimate)[^\n]{0,60}",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()
        snippet = stripped[:300].replace("\n", " ").strip()
        if len(stripped) > 300:
            snippet += "…"
        lines.append(f"• {display}: {snippet}")
    return "\n".join(lines)


def _compute_transcript_median(
    debate_log: list[dict[str, str]],
    scale: tuple[int, int],
) -> float | None:
    """Collect each non-synthesizer agent's last stated score and return the median."""
    lo, hi = scale
    all_scores: list[float] = []
    for entry in debate_log:
        if entry["role"] == "synthesizer":
            continue
        val = _extract_stated_score(entry.get("content", ""))
        if val is not None and lo <= val <= hi:
            all_scores.append(val)

    if not all_scores:
        return None
    all_scores.sort()
    n = len(all_scores)
    mid = n // 2
    return all_scores[mid] if n % 2 else (all_scores[mid - 1] + all_scores[mid]) / 2


def _save_debate_log(
    debate_log: list[dict[str, str]],
    metric: MetricDefinition,
    result: MetricResult,
    repo_slug: str,
    snapshot_timestamp: float | None,
    log_dir: str,
) -> str:
    """Write *debate_log* to a JSON file and return the path."""
    os.makedirs(log_dir, exist_ok=True)
    ts = int(snapshot_timestamp) if snapshot_timestamp is not None else int(
        datetime.now(timezone.utc).timestamp()
    )
    slug_safe = repo_slug.replace("/", "_")
    filename = f"{metric.name}_{slug_safe}_{ts}.json"
    path = os.path.join(log_dir, filename)

    median_score = _compute_transcript_median(debate_log, metric.scale)
    payload = {
        "metric": metric.name,
        "display_name": metric.display_name,
        "repo": repo_slug,
        "snapshot_timestamp": snapshot_timestamp,
        "agents": [a.name for a in metric.get_debate_agents()],
        "debate_rounds": metric.debate_rounds,
        "final_score": result.get("metric"),
        "transcript_median": median_score,
        "log": debate_log,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Single sub-graph
# ═══════════════════════════════════════════════════════════════════════════

def _single_node(state: SingleMetricState) -> dict:
    """One-shot LLM call that scores the metric and stores the result."""
    metric = state["metric"]
    context = retrieve_context(state["repo_data"], metric)
    user_msg = _build_scoring_user_prompt(metric, context)
    messages = [
        {"role": "system", "content": SINGLE_AGENT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    reply = call_llm(messages, model=state["llm_model"], llm_host=state["llm_host"], llm_auth_token=state.get("llm_auth_token"))
    parsed = _parse_json_response(reply)
    return {
        "result": MetricResult(
            name=metric.name,
            display_name=metric.display_name,
            metric=parsed.get("metric"),
            explanation=parsed.get("explanation", ""),
            pipeline="single",
            debate_log=[],
        )
    }


def _build_single_graph() -> Any:
    g = StateGraph(SingleMetricState)
    g.add_node("run_single", _single_node)
    g.add_edge(START, "run_single")
    g.add_edge("run_single", END)
    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Debate sub-graph  —  proper LangGraph conditional-edge loop
# ═══════════════════════════════════════════════════════════════════════════

def _agent_turn_node(state: DebateState) -> dict:
    """One agent speaks; advance the round-robin cursors."""
    metric: MetricDefinition = state["metric"]
    agents = metric.get_debate_agents()
    agent_idx = state.get("current_agent_idx", 0)
    round_idx = state.get("current_round", 0)
    debate_log: list[dict[str, str]] = state.get("debate_log", [])

    agent = agents[agent_idx]
    context = retrieve_context(state["repo_data"], metric)
    base_user_msg = _build_user_prompt(metric, context)
    round_label = f"Round {round_idx + 1} of {metric.debate_rounds}"

    if not debate_log:
        agent_user_content = (
            f"[{round_label}]\n\n"
            f"{base_user_msg}\n\n"
            "Analyse the repository data above from your perspective and "
            "give your initial score estimate with reasoning. "
            "Do NOT invent any data — only cite what is present above. "
            "State your current score estimate as a plain number at the "
            "very end of your message, e.g. 'My current estimate: 6/10'."
        )
    else:
        peer_summary = _extract_peer_summaries(debate_log, agent.name, agents)
        agent_user_content = (
            f"[{round_label}]\n\n"
            f"{base_user_msg}\n\n"
            f"{peer_summary}\n\n"
            "Respond to any peer points that cite real repository data, "
            "and update your score estimate ONLY if new evidence from "
            "the repository data warrants it. "
            "Do NOT copy or mirror another agent's score — anchor yours "
            "to the specific evidence you cite. "
            "Do NOT invent any data. "
            "State your current score estimate as a plain number at the "
            "very end of your message, e.g. 'My current estimate: 6/10'."
        )

    system = (
        f"You are the {agent.display_name} in a structured evaluation panel.\n\n"
        + agent.system_prompt
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": agent_user_content},
    ]
    reply = call_llm(
        messages,
        model=state["llm_model"],
        llm_host=state["llm_host"],
        llm_auth_token=state.get("llm_auth_token"),
        max_tokens=800,
    )

    new_log = debate_log + [{"role": agent.name, "content": reply}]

    # Advance cursors
    next_agent_idx = agent_idx + 1
    next_round = round_idx
    if next_agent_idx >= len(agents):
        next_agent_idx = 0
        next_round = round_idx + 1

    return {
        "debate_log": new_log,
        "current_agent_idx": next_agent_idx,
        "current_round": next_round,
    }


def _synthesize_node(state: DebateState) -> dict:
    """Read the full debate transcript and produce the final JSON score."""
    metric: MetricDefinition = state["metric"]
    debate_log: list[dict[str, str]] = state["debate_log"]
    agents = metric.get_debate_agents()

    transcript = "\n\n".join(
        f"[{e['role'].upper()}]:\n{e['content']}" for e in debate_log
    )
    agent_names = ", ".join(a.display_name for a in agents)

    median_score = _compute_transcript_median(debate_log, metric.scale)
    median_hint = (
        f"The median of all numeric score estimates in the transcript is "
        f"{median_score:.1f} (scale {metric.scale[0]}–{metric.scale[1]}). "
        "Your final score MUST be close to this median unless you can cite a "
        "specific, concrete reason from the repository data to deviate significantly.\n\n"
        if median_score is not None
        else ""
    )

    synth_messages = [
        {"role": "system", "content": metric.get_synthesizer_prompt()},
        {
            "role": "user",
            "content": (
                f"Metric: {metric.display_name} (scale {metric.scale[0]}–{metric.scale[1]})\n"
                f"Participants: {agent_names}\n\n"
                f"Debate transcript:\n{transcript}\n\n"
                f"{median_hint}"
                "Produce a final consensus score and explanation. "
                "Disregard any score estimates in the transcript that were based on "
                "invented, simulated, or assumed data not present in the repository JSON.\n\n"
                "IMPORTANT: your entire response must be a single JSON object with no "
                "surrounding text, markdown, or code fences. Required format:\n"
                f'{{"metric": <number {metric.scale[0]}–{metric.scale[1]}>, "explanation": "<string>"}}'
            ),
        },
    ]
    synth_reply = call_llm(
        synth_messages,
        model=state["llm_model"],
        llm_host=state["llm_host"],
        llm_auth_token=state.get("llm_auth_token"),
        max_tokens=600,
    )
    final_log = debate_log + [{"role": "synthesizer", "content": synth_reply}]

    parsed = _parse_json_response(synth_reply)
    if parsed.get("metric") is None and median_score is not None:
        parsed["metric"] = median_score
        parsed.setdefault("explanation", synth_reply)

    lo, hi = metric.scale
    raw_metric = parsed.get("metric")
    if raw_metric is not None:
        parsed["metric"] = max(lo, min(hi, float(raw_metric)))

    result = MetricResult(
        name=metric.name,
        display_name=metric.display_name,
        metric=parsed.get("metric"),
        explanation=parsed.get("explanation", ""),
        pipeline="debate",
        debate_log=final_log,
    )

    log_path = _save_debate_log(
        final_log,
        metric,
        result,
        state.get("repo_slug", "unknown_repo"),
        state.get("snapshot_timestamp"),
        state.get("log_dir", "debate_logs"),
    )
    print(f"  💾  Debate log saved: {log_path}")

    return {"debate_log": final_log, "result": result}


def _should_synthesize(state: DebateState) -> Literal["agent_turn", "synthesize"]:
    """Conditional edge: keep looping until all rounds are exhausted."""
    if state.get("current_round", 0) >= state["metric"].debate_rounds:
        return "synthesize"
    return "agent_turn"


def _build_debate_graph() -> Any:
    g = StateGraph(DebateState)
    g.add_node("agent_turn", _agent_turn_node)
    g.add_node("synthesize", _synthesize_node)
    g.add_edge(START, "agent_turn")
    g.add_conditional_edges("agent_turn", _should_synthesize)
    g.add_edge("synthesize", END)
    return g.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Top-level graph — fan-out via Send, accumulate via reducer
# ═══════════════════════════════════════════════════════════════════════════

_single_graph = _build_single_graph()
_debate_graph = _build_debate_graph()


def _fan_out(state: PipelineState) -> list[Send]:
    """Return one Send per metric — used as a conditional-edges function off START."""
    metrics = state.get("metrics", DEFAULT_METRICS)
    sends = []
    for metric in metrics:
        agents_desc = ""
        if metric.pipeline == "debate":
            agent_list = metric.get_debate_agents()
            agents_desc = f" — agents: {[a.name for a in agent_list]}"
        print(f"  ⏳  Queuing: {metric.display_name} ({metric.pipeline}{agents_desc})")

        scoped: dict[str, Any] = {
            "metric": metric,
            "repo_data": state["repo_data"],
            "llm_model": state.get("llm_model", "gemma3:27b"),
            "llm_host": state.get("llm_host", "http://localhost:11434"),
            "llm_auth_token": state.get("llm_auth_token", None),
            "repo_slug": state.get("repo_slug", "unknown_repo"),
            "snapshot_timestamp": state.get("snapshot_timestamp"),
            "log_dir": state.get("log_dir", "debate_logs"),
        }
        if metric.pipeline == "debate":
            scoped.update({"debate_log": [], "current_round": 0, "current_agent_idx": 0})
            sends.append(Send("run_debate", scoped))
        else:
            sends.append(Send("run_single", scoped))
    return sends


def _run_single_wrapper(state: SingleMetricState) -> dict:
    """Invoke the single sub-graph and surface the result into top-level state."""
    final = _single_graph.invoke(state)
    result: MetricResult = final["result"]
    print(f"  ✅  {result['display_name']}: {result.get('metric', '?')}")
    return {"results": [result]}


def _run_debate_wrapper(state: DebateState) -> dict:
    """Invoke the debate sub-graph and surface the result into top-level state."""
    final = _debate_graph.invoke(state)
    result: MetricResult = final["result"]
    print(f"  ✅  {result['display_name']}: {result.get('metric', '?')}")
    return {"results": [result]}


def build_graph() -> Any:
    """Construct and compile the top-level LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("run_single", _run_single_wrapper)
    graph.add_node("run_debate", _run_debate_wrapper)

    # _fan_out is a conditional-edges function: it returns Send objects that
    # route each metric to run_single or run_debate without going through a node.
    graph.add_conditional_edges(START, _fan_out, ["run_single", "run_debate"])
    graph.add_edge("run_single", END)
    graph.add_edge("run_debate", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Public API (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def analyse_repo(
    repo_data: dict[str, Any],
    metrics: list[MetricDefinition] | None = None,
    *,
    model: str = "gemma3:27b",
    llm_host: str = "http://localhost:11434",
    llm_auth_token: str | None = None,
    repo_slug: str = "unknown_repo",
    snapshot_timestamp: float | None = None,
    log_dir: str = "debate_logs",
) -> list[MetricResult]:
    """Run the full analysis pipeline and return a list of ``MetricResult``s."""
    app = build_graph()
    initial_state: PipelineState = {
        "repo_data": repo_data,
        "metrics": metrics or DEFAULT_METRICS,
        "llm_model": model,
        "llm_host": llm_host,
        "llm_auth_token": llm_auth_token,
        "repo_slug": repo_slug,
        "snapshot_timestamp": snapshot_timestamp,
        "log_dir": log_dir,
        "results": [],
    }
    final = app.invoke(initial_state)
    return final["results"]
