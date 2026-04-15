"""
RAG module – extracts and ranks relevant chunks from the repository JSON.

The repo JSON is large and contains many sections. For each metric we only
want to feed the LLMs the sections that are actually relevant, plus a compact
summary of the overall repo for context.

Because the data is structured (not free-text) we use a lightweight keyword /
section-matching approach rather than a full vector-store.  This keeps the
pipeline self-contained and avoids an external embedding service.
"""

from __future__ import annotations

import json
from typing import Any

from llm_metrics.config import MetricDefinition

# ── Section mapping ─────────────────────────────────────────────────────────
# Maps logical data_key names (used in MetricDefinition.data_keys) to
# JSONPath-like accessors into the repo dict.

_SECTION_MAP: dict[str, list[str]] = {
    "issues": [
        "repository.repository.issues",
        # Also grab individual issue detail keys (issue1, issue2, …)
        "repository.repository.issue*",
    ],
    "pulls": [
        "repository.repository.pullRequests",
    ],
    "branches": [
        "repository.repository.branches",
    ],
    "releases": [
        "repository.repository.releases",
    ],
    "commits": [
        "repository.commits",
        "repository.repository.defaultBranchRef",
    ],
    "contributors": [
        "repository.contributors",
        "repository.organizations",
    ],
    "community_profile": [
        "repository.community_profile",
    ],
    "readme": [
        "repository.repository.README_md",
        "repository.repository.README_txt",
        "repository.repository.README",
        "repository.repository.docs_README_md",
    ],
    "contributing": [
        "repository.repository.contributing_md",
        "repository.repository.contributing_txt",
        "repository.repository.contributing_raw",
        "repository.repository.contributingGuidelines",
    ],
    "dependencies": [
        "repository.dependencies",
        "repository.dependents",
    ],
    "advisories": [
        "repository.advisories",
    ],
}


def _resolve_path(data: dict, path: str) -> Any | None:
    """Walk *data* using a dot-separated *path*.

    Supports a trailing ``*`` wildcard which collects all keys that start
    with the prefix (e.g. ``repository.repository.issue*`` matches
    ``issue1``, ``issue19``, …).
    """
    parts = path.split(".")
    current: Any = data
    for i, part in enumerate(parts):
        if current is None or not isinstance(current, dict):
            return None
        if part.endswith("*"):
            prefix = part[:-1]
            remaining = ".".join(parts[i + 1 :]) if i + 1 < len(parts) else ""
            collected = {}
            for key in current:
                if key.startswith(prefix):
                    if remaining:
                        val = _resolve_path(current[key], remaining)
                    else:
                        val = current[key]
                    if val is not None:
                        collected[key] = val
            return collected or None
        current = current.get(part)
    return current


def _summarise_advisories(advisories: Any) -> dict:
    """Replace a raw advisory list with a compact statistical summary.

    Dumping full CVE descriptions into the context causes every agent to
    discuss security at length regardless of its assigned persona, drowning
    out governance/maturity signals.  A counts-plus-severity summary gives
    agents the signal they need without flooding the token budget.

    Returns a dict like::

        {
          "total": 7,
          "by_severity": {"HIGH": 3, "MEDIUM": 2, "LOW": 2},
          "has_patches": true,
          "types": ["RCE", "XSS"]
        }
    """
    if not isinstance(advisories, list) or not advisories:
        return {"total": 0}

    severity_counts: dict[str, int] = {}
    has_patches = False
    vuln_types: list[str] = []

    for adv in advisories:
        if not isinstance(adv, dict):
            continue
        sev = str(adv.get("severity") or adv.get("cvss", {}).get("vectorString") or "UNKNOWN").upper()
        # Normalise to broad buckets
        if any(s in sev for s in ("CRITICAL",)):
            sev = "CRITICAL"
        elif any(s in sev for s in ("HIGH",)):
            sev = "HIGH"
        elif any(s in sev for s in ("MODERATE", "MEDIUM")):
            sev = "MEDIUM"
        elif any(s in sev for s in ("LOW",)):
            sev = "LOW"
        else:
            sev = "UNKNOWN"
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Any patched version present?
        if adv.get("patchedVersions") or adv.get("firstPatchedVersion"):
            has_patches = True

        # Collect short vulnerability type labels (CWE name / summary first sentence)
        summary = adv.get("summary") or adv.get("description") or ""
        if summary:
            first_sentence = summary.split(".")[0].strip()[:60]
            if first_sentence and first_sentence not in vuln_types:
                vuln_types.append(first_sentence)

    return {
        "total": len(advisories),
        "by_severity": severity_counts,
        "has_patches": has_patches,
        # Keep only up to 5 type labels to avoid re-flooding the context
        "types": vuln_types[:5],
    }


def _compact_json(obj: Any, max_depth: int = 6, _depth: int = 0) -> Any:
    """Return a depth-limited, compacted version of *obj*."""
    if _depth >= max_depth:
        if isinstance(obj, dict):
            return f"{{...{len(obj)} keys}}"
        if isinstance(obj, list):
            return f"[...{len(obj)} items]"
        return obj
    if isinstance(obj, dict):
        return {k: _compact_json(v, max_depth, _depth + 1) for k, v in obj.items() if k not in ("pageInfo", "cursor", "id")}
    if isinstance(obj, list):
        # For very long lists keep only first + last to save tokens
        if len(obj) > 6:
            return (
                [_compact_json(x, max_depth, _depth + 1) for x in obj[:3]]
                + [f"... ({len(obj) - 6} more items) ..."]
                + [_compact_json(x, max_depth, _depth + 1) for x in obj[-3:]]
            )
        return [_compact_json(x, max_depth, _depth + 1) for x in obj]
    return obj


def build_repo_summary(repo_data: dict) -> str:
    """Return a short context summary of the repository for all prompts."""
    repo = repo_data.get("repository", {}).get("repository", {})
    info = {
        "name": repo.get("nameWithOwner", "unknown"),
        "description": repo.get("description", ""),
        "created": repo.get("createdAt"),
        "updated": repo.get("updatedAt"),
        "archived": repo.get("archivedAt"),
        "stars": repo.get("stargazerCount"),
        "license": (repo.get("licenseInfo") or {}).get("spdxId"),
        "total_issues": (repo.get("issues") or {}).get("totalCount"),
        "total_prs": (repo.get("pullRequests") or {}).get("totalCount"),
        "total_releases": (repo.get("releases") or {}).get("totalCount"),
        "total_branches": (repo.get("branches") or {}).get("totalCount"),
        "health_percentage": repo_data.get("repository", {})
        .get("community_profile", {})
        .get("health_percentage"),
    }
    return json.dumps(info, indent=2)


def retrieve_context(repo_data: dict, metric: MetricDefinition) -> str:
    """Build the RAG context string for a specific *metric*.

    Returns a JSON string containing:
      1. A compact repo summary (always included).
      2. The relevant sections indicated by ``metric.data_keys``.

    Advisory data is intentionally replaced with a compact statistical
    summary (total count + severity breakdown) rather than raw CVE text, to
    prevent security language from drowning out governance/maturity signals
    when those advisories happen to be present.
    """
    sections: dict[str, Any] = {}
    for key in metric.data_keys:
        paths = _SECTION_MAP.get(key, [])
        for path in paths:
            resolved = _resolve_path(repo_data, path)
            if resolved is not None:
                # Replace raw advisory lists with a compact summary so they
                # don't flood the context and bias all agents toward security.
                if path == "repository.advisories" and isinstance(resolved, list):
                    resolved = _summarise_advisories(resolved)
                sections[path] = resolved

    compacted = _compact_json(sections)
    summary = build_repo_summary(repo_data)
    return (
        "=== Repository Summary ===\n"
        + summary
        + "\n\n=== Relevant Data ===\n"
        + json.dumps(compacted, indent=2, default=str)
    )
