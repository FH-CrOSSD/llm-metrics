"""
LangGraph pipeline for repository health analysis.

Two sub-graphs, selected per metric by the ``pipeline`` field:

  * **single** – one LLM call produces the score directly.
  * **debate** – N configurable agents take turns for K rounds → synthesizer.

The top-level graph fans-out across all requested metrics, collects results,
and returns a list of ``MetricResult``.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    SINGLE_AGENT_SYSTEM,
    call_llm,
)
from config import DEFAULT_METRICS, DebateAgent, MetricDefinition
from rag import build_repo_summary, retrieve_context

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


class PipelineState(TypedDict, total=False):
    """Top-level state shared across the full graph."""
    repo_data: dict[str, Any]
    metrics: list[MetricDefinition]
    results: list[MetricResult]
    llm_model: str
    llm_host: str
    # Provenance – used to name saved debate logs
    repo_slug: str          # e.g. "lorabridge/lorabridge"
    snapshot_timestamp: float | None  # crawl timestamp, or None for local files
    log_dir: str            # directory where debate logs are written


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
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Markdown fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. First bare { … }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 4. Numeric prose fallback — look for patterns like "score: 7", "a 7/10",
    #    "I'd give this a 7", "rating of 7", "7 out of 10"
    score_match = re.search(
        r"(?:score|rating|metric|give(?:\s+this)?(?:\s+a)?|rank)\D{0,10}(\d+(?:\.\d+)?)"
        r"|(\d+(?:\.\d+)?)\s*/\s*10"
        r"|\b(\d+(?:\.\d+)?)\s+out\s+of\s+10",
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


# ═══════════════════════════════════════════════════════════════════════════
# Single-agent sub-graph
# ═══════════════════════════════════════════════════════════════════════════

def _run_single(
    metric: MetricDefinition,
    repo_data: dict,
    model: str,
    llm_host: str,
) -> MetricResult:
    context = retrieve_context(repo_data, metric)
    user_msg = _build_scoring_user_prompt(metric, context)  # strict JSON reminder
    messages = [
        {"role": "system", "content": SINGLE_AGENT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    reply = call_llm(messages, model=model, llm_host=llm_host)
    parsed = _parse_json_response(reply)
    return MetricResult(
        name=metric.name,
        display_name=metric.display_name,
        metric=parsed.get("metric"),
        explanation=parsed.get("explanation", ""),
        pipeline="single",
        debate_log=[],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-agent debate sub-graph (generic N-agent round-robin)
# ═══════════════════════════════════════════════════════════════════════════

def _format_transcript(debate_log: list[dict[str, str]]) -> str:
    """Render the debate log into a readable transcript string."""
    return "\n\n".join(
        f"[{entry['role'].upper()}]:\n{entry['content']}" for entry in debate_log
    )


def _extract_stated_score(text: str) -> float | None:
    """Pull the *last* explicit score mention from an agent turn.

    Searching from the end of the text reduces the risk of picking up a
    score mentioned in passing (e.g. quoting a peer) rather than the
    agent's own current estimate, which is typically stated at the end.

    Returns the numeric value, or None if nothing was found.
    """
    matches = re.findall(
        r"(?:current estimate|my estimate|i.{0,8}give|i.{0,8}rate|i.{0,8}score|score|rating|estimate)"
        r"[^\d]{0,15}(\d+(?:\.\d+)?)"
        r"|(\d+(?:\.\d+)?)\s*/\s*10",
        text,
        re.IGNORECASE,
    )
    # Walk matches in reverse to find the agent's final stated estimate
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

    Each bullet contains only a brief prose snippet (≤300 chars) of the
    agent's reasoning.  Score estimates are intentionally OMITTED here to
    prevent social-contagion cascades; numeric anchoring is handled solely
    by the Python-computed median injected into the synthesizer prompt.

    We deliberately do NOT pass the full transcript to discussion agents so
    that hallucinations from earlier turns cannot compound.  Only the
    synthesizer receives the complete transcript.
    """
    name_to_display = {a.name: a.display_name for a in agents}

    # Keep only the most recent entry per peer agent
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
        # Strip any score lines at the tail to avoid implicit anchoring
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
    """Collect each non-synthesizer agent's *last* stated score per turn and
    return the median.

    Using the last score per turn (via ``_extract_stated_score`` which searches
    from the tail) means we capture the agent's final estimate for that round
    rather than a score mentioned in passing while quoting a peer.

    Returns None if no valid scores were found.
    """
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


def _run_debate(
    metric: MetricDefinition,
    repo_data: dict,
    model: str,
    llm_host: str,
) -> MetricResult:
    """Run a configurable multi-agent debate.

    Each agent takes turns in round-robin order for ``metric.debate_rounds``
    full rounds.  On follow-up turns an agent sees only a short bullet summary
    of the *other* agents' most recent positions — NOT the full transcript —
    to prevent hallucination compounding.  The synthesizer alone receives the
    complete transcript.

    Each agent's identity is embedded in the system prompt wrapper produced by
    ``_agent_system_prompt()`` so the model cannot misattribute its own role.
    """
    agents: list[DebateAgent] = metric.get_debate_agents()
    context = retrieve_context(repo_data, metric)
    base_user_msg = _build_user_prompt(metric, context)
    debate_log: list[dict[str, str]] = []

    agent_names = ", ".join(a.display_name for a in agents)

    for round_idx in range(metric.debate_rounds):
        for agent in agents:
            round_label = f"Round {round_idx + 1} of {metric.debate_rounds}"

            if not debate_log:
                # First turn — repository data only; no prior discussion yet
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

            # Identity is locked in the system prompt so the model cannot confuse roles
            system = (
                f"You are the {agent.display_name} in a structured evaluation panel.\n\n"
                + agent.system_prompt
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": agent_user_content},
            ]
            reply = call_llm(messages, model=model, llm_host=llm_host, max_tokens=800)
            debate_log.append({"role": agent.name, "content": reply})

    # ── Synthesizer: receives full transcript, produces final JSON ────────
    transcript = _format_transcript(debate_log)

    # Compute the median of all scores stated in the transcript in Python so
    # the LLM has an explicit anchor and cannot drift to 0 or 10.
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
    synth_reply = call_llm(synth_messages, model=model, llm_host=llm_host, max_tokens=600)
    debate_log.append({"role": "synthesizer", "content": synth_reply})

    # If the LLM still produced a bad score, fall back to the Python-computed median
    parsed = _parse_json_response(synth_reply)
    if parsed.get("metric") is None and median_score is not None:
        parsed["metric"] = median_score
        parsed.setdefault("explanation", synth_reply)

    # Hard clamp: ensure the final score is always within the declared scale,
    # regardless of what the LLM returned.  This prevents values like -1 or 11
    # from slipping through the JSON parser's numeric fallback.
    lo, hi = metric.scale
    raw_metric = parsed.get("metric")
    if raw_metric is not None:
        parsed["metric"] = max(lo, min(hi, float(raw_metric)))

    return MetricResult(
        name=metric.name,
        display_name=metric.display_name,
        metric=parsed.get("metric"),
        explanation=parsed.get("explanation", ""),
        pipeline="debate",
        debate_log=debate_log,
    )


def _save_debate_log(
    debate_log: list[dict[str, str]],
    metric: MetricDefinition,
    result: "MetricResult",
    repo_slug: str,
    snapshot_timestamp: float | None,
    log_dir: str,
) -> str:
    """Write *debate_log* to a JSON file and return the path.

    Filename pattern: ``{metric}_{repo_slug_safe}_{timestamp}.json``
    """
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
# LangGraph wiring
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_metrics_node(state: PipelineState) -> dict:
    """Evaluate all requested metrics (sequentially for Ollama)."""
    repo_data = state["repo_data"]
    metrics = state.get("metrics", DEFAULT_METRICS)
    model = state.get("llm_model", "gemma3:27b")
    llm_host = state.get("llm_host", "http://localhost:11434")
    repo_slug = state.get("repo_slug", "unknown_repo")
    snapshot_timestamp = state.get("snapshot_timestamp")
    log_dir = state.get("log_dir", "debate_logs")

    results: list[MetricResult] = []
    for metric in metrics:
        agents_desc = ""
        if metric.pipeline == "debate":
            agent_list = metric.get_debate_agents()
            agents_desc = f" — agents: {[a.name for a in agent_list]}"
        print(f"  ⏳  Evaluating: {metric.display_name} ({metric.pipeline}{agents_desc}) …")
        if metric.pipeline == "debate":
            result = _run_debate(metric, repo_data, model, llm_host)
            log_path = _save_debate_log(
                result["debate_log"], metric, result, repo_slug, snapshot_timestamp, log_dir
            )
            print(f"  💾  Debate log saved: {log_path}")
        else:
            result = _run_single(metric, repo_data, model, llm_host)
        results.append(result)
        score = result.get("metric", "?")
        print(f"  ✅  {metric.display_name}: {score}")

    return {"results": results}


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("evaluate_metrics", _evaluate_metrics_node)

    graph.set_entry_point("evaluate_metrics")
    graph.add_edge("evaluate_metrics", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def analyse_repo(
    repo_data: dict[str, Any],
    metrics: list[MetricDefinition] | None = None,
    *,
    model: str = "gemma3:27b",
    llm_host: str = "http://localhost:11434",
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
        "repo_slug": repo_slug,
        "snapshot_timestamp": snapshot_timestamp,
        "log_dir": log_dir,
        "results": [],
    }
    final = app.invoke(initial_state)
    return final["results"]
