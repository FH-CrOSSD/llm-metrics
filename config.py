"""
Metric definitions for the repository health analysis pipeline.

Each metric is declarative and specifies:
  - name: unique identifier
  - display_name: human-readable name
  - description: what the metric measures
  - prompt: the analysis prompt sent to LLMs
  - pipeline: "single" for one-shot scoring, "debate" for multi-agent discussion
  - scale: (min, max) score range
  - data_keys: which sections of the repo JSON are most relevant (for RAG)
  - debate_rounds: (only for "debate" pipeline) how many rounds of discussion
  - debate_agents: (optional) custom list of debate participants
  - synthesizer_prompt: (optional) custom synthesizer system prompt

Add / remove / reorder entries to customise the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Debate agent definition ─────────────────────────────────────────────


@dataclass
class DebateAgent:
    """A single participant in a multi-agent debate.

    Attributes:
        name: Short identifier shown in the debate log (e.g. "analyst").
        display_name: Human-readable label (e.g. "Software Analyst").
        system_prompt: The LLM system prompt that defines this agent's persona.
    """
    name: str
    display_name: str
    system_prompt: str


# ── Built-in agent presets ──────────────────────────────────────────────

_DEBATE_RULES = (
    "STRICT RULES — violations will corrupt the final score:\n"
    "1. ONLY cite facts that appear verbatim in the repository JSON provided. "
    "NEVER invent, simulate, assume, or extrapolate data that is not explicitly "
    "present (e.g. do NOT say 'I ran a scan and found X vulnerabilities' or "
    "'simulated data shows…').\n"
    "2. If a piece of information is absent from the data, say so explicitly "
    "('the data does not include…') rather than filling the gap.\n"
    "3. Keep your score estimate within the stated scale and anchor it to the "
    "evidence you cite. Do NOT lower (or raise) your score because another "
    "participant did — only revise your estimate when new repository data "
    "justifies it. Scores of 0 or 10 require extraordinary data-backed reasons; "
    "never assign them just because someone else did.\n"
    "4. Write only in YOUR OWN voice. Do NOT write role headers such as "
    "'Response from the Analyst', 'Round N Response – Security Auditor', etc., "
    "and never speak as or for another participant.\n"
    "5. A synthesizer will produce the final JSON score — focus on concise "
    "plain-text reasoning and state your current score estimate as a single "
    "number at the very end of your message, e.g. 'My current estimate: 6/10'."
)

AGENT_ANALYST = DebateAgent(
    name="analyst",
    display_name="Software Analyst",
    system_prompt=(
        "You are a meticulous open-source software analyst participating in a "
        "structured multi-agent discussion to evaluate a repository metric. "
        "Ground every claim in the data provided. Be objective and concise.\n\n"
        + _DEBATE_RULES
    ),
)

AGENT_REVIEWER = DebateAgent(
    name="reviewer",
    display_name="Peer Reviewer",
    system_prompt=(
        "You are a critical peer reviewer participating in a structured multi-agent "
        "discussion to evaluate a repository metric. "
        "Challenge, refine, or validate the other participants' assessments. "
        "Point out data that may have been missed, over-weighted, or under-weighted.\n\n"
        + _DEBATE_RULES
    ),
)

AGENT_USER_ADVOCATE = DebateAgent(
    name="user_advocate",
    display_name="User Advocate",
    system_prompt=(
        "You are an advocate for end-users and newcomers participating in a "
        "structured multi-agent discussion to evaluate a repository metric. "
        "Evaluate the repository from the perspective of someone trying to use, "
        "install, or contribute for the first time. Highlight usability gaps, "
        "missing guides, unclear issue responses, and barriers to entry.\n\n"
        + _DEBATE_RULES
    ),
)

AGENT_DEVELOPER_ADVOCATE = DebateAgent(
    name="developer_advocate",
    display_name="Developer Advocate",
    system_prompt=(
        "You are an advocate for the core development team participating in a "
        "structured multi-agent discussion to evaluate a repository metric. "
        "Evaluate the repository from the perspective of maintainers under "
        "resource constraints. Acknowledge good engineering practices and give "
        "credit where due while noting areas that could realistically be improved.\n\n"
        + _DEBATE_RULES
    ),
)

AGENT_SECURITY_AUDITOR = DebateAgent(
    name="security_auditor",
    display_name="Security Auditor",
    system_prompt=(
        "You are a security-focused auditor participating in a structured multi-agent "
        "discussion to evaluate a repository metric. "
        "Evaluate the project's security posture: security policy, advisories, "
        "dependency hygiene, vulnerability disclosure process, and code review "
        "practices.\n\n"
        + _DEBATE_RULES
    ),
)

# Default agent pair used when no custom agents are specified
DEFAULT_DEBATE_AGENTS: list[DebateAgent] = [AGENT_ANALYST, AGENT_REVIEWER]

DEFAULT_SYNTHESIZER_PROMPT = (
    "You are a senior evaluator. Given a multi-round discussion between "
    "several agents about a repository metric, produce a final consensus "
    "score and explanation that incorporates the strongest evidence-backed "
    "arguments from all participants.\n\n"
    "CRITICAL SCORING RULES:\n"
    "1. Base the score ONLY on facts present in the repository data. Ignore "
    "any agent claims that cite invented, simulated, or assumed data.\n"
    "2. Scores of 0 (or minimum) and 10 (or maximum) are EXTRAORDINARY claims "
    "that require overwhelming, specific, concrete evidence from the repository "
    "data — not consensus or social pressure among agents. If agents converged "
    "toward 0 or 10 by following each other rather than citing new data, "
    "disregard that drift and score based on evidence alone.\n"
    "3. Your score MUST stay close to the median of the evidence-backed "
    "estimates provided in the transcript. Significant deviation requires a "
    "specific, named data point from the repository JSON as justification.\n\n"
    "You MUST respond with ONLY a JSON object — no markdown, no prose before "
    "or after it. The object must have exactly two keys:\n"
    '  "metric": a single number within the stated scale\n'
    '  "explanation": a concise string summarising the reasoning\n'
    "Example of the required output format:\n"
    '{"metric": 7, "explanation": "The project demonstrates ..."}'
)


# ── Metric definition ──────────────────────────────────────────────────


@dataclass
class MetricDefinition:
    name: str
    display_name: str
    description: str
    prompt: str
    pipeline: Literal["single", "debate"] = "single"
    scale: tuple[int, int] = (1, 10)
    data_keys: list[str] = field(default_factory=list)
    debate_rounds: int = 2  # only used when pipeline == "debate"
    # Custom debate participants – defaults to [analyst, reviewer]
    debate_agents: list[DebateAgent] | None = None
    # Custom synthesizer system prompt – defaults to DEFAULT_SYNTHESIZER_PROMPT
    synthesizer_prompt: str | None = None

    def get_debate_agents(self) -> list[DebateAgent]:
        """Return the agents for this metric, falling back to the defaults."""
        return self.debate_agents if self.debate_agents is not None else DEFAULT_DEBATE_AGENTS

    def get_synthesizer_prompt(self) -> str:
        """Return the synthesizer prompt, falling back to the default."""
        return self.synthesizer_prompt if self.synthesizer_prompt is not None else DEFAULT_SYNTHESIZER_PROMPT


# ---------------------------------------------------------------------------
# Default metric catalogue – edit this list to customise the analysis
# ---------------------------------------------------------------------------

DEFAULT_METRICS: list[MetricDefinition] = [
    # ── Simple single-agent metrics ──────────────────────────────────────
    MetricDefinition(
        name="friendliness",
        display_name="Developer Friendliness",
        description="How friendly and welcoming are the developers in issues and comments?",
        prompt=(
            "Evaluate how friendly and welcoming the repository developers are "
            "when interacting with contributors and users in issues and comments. "
            "Consider tone, responsiveness, helpfulness, and encouragement. "
            "Give a score from {min} to {max} where {min} is hostile/unwelcoming "
            "and {max} is exceptionally friendly and supportive."
        ),
        pipeline="single",
        data_keys=["issues", "community_profile"],
    ),
    MetricDefinition(
        name="documentation_quality",
        display_name="Documentation Quality",
        description="Quality and professionalism of documentation, READMEs, and guides.",
        prompt=(
            "Analyse the quality of documentation in this repository. Consider the "
            "README structure, whether there are contributing guides, code of conduct, "
            "issue/PR templates, and external docs links. "
            "Give a score from {min} to {max} where {min} is poor and {max} is excellent."
        ),
        pipeline="single",
        data_keys=["community_profile", "readme", "contributing"],
    ),
    # ── Multi-agent debate metrics (using default analyst + reviewer) ────
    MetricDefinition(
        name="development_efficiency",
        display_name="Development Efficiency",
        description="How efficiently is development conducted? Issue turnaround, branching, PRs, releases.",
        prompt=(
            "Evaluate how efficiently development is conducted in this repository. "
            "Consider issue resolution time, PR turnaround, branching strategy, "
            "release cadence, commit activity, and contributor productivity. "
            "Give a score from {min} to {max} where {min} is very inefficient "
            "and {max} is highly efficient."
        ),
        pipeline="debate",
        data_keys=["issues", "pulls", "branches", "releases", "commits", "contributors"],
        debate_rounds=2,
        # uses default [analyst, reviewer] — no need to specify
    ),
    # ── Multi-agent debate with custom agents ────────────────────────────
    MetricDefinition(
        name="project_maturity",
        display_name="Project Maturity",
        description="Overall maturity of the project considering governance, processes, and community.",
        prompt=(
            "Evaluate the overall maturity of this open-source project. "
            "Score holistically across ALL of the following dimensions — "
            "no single dimension should dominate:\n"
            "  • Governance: code of conduct, contributing guide, issue/PR templates\n"
            "  • Community: contributor count, bus-factor, PR activity, issue responsiveness\n"
            "  • Release & versioning: release history, cadence, tagging\n"
            "  • Documentation: README quality, external docs, changelogs\n"
            "  • CI/CD & automation: automated tests, build pipelines\n"
            "  • Security posture: security policy presence, advisory count/severity, "
            "patch responsiveness — treat this as ONE of the six dimensions above, "
            "not a veto over the others\n\n"
            "Give a score from {min} to {max} in increments of 0.1 where {min} is very immature "
            "and {max} is a fully mature project."
        ),
        pipeline="debate",
        data_keys=["issues", "pulls", "branches", "releases", "commits", "contributors",
            "community_profile",
            "dependencies",
            "advisories",
        ],
        debate_rounds=3,
        # Three-way debate: user advocate, developer advocate, security auditor
        debate_agents=[AGENT_ANALYST, AGENT_DEVELOPER_ADVOCATE, AGENT_USER_ADVOCATE, AGENT_SECURITY_AUDITOR, AGENT_REVIEWER],
    ),
]
