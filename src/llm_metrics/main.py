#!/usr/bin/env python3
"""
Entry-point for the CrOSSD LangGraph repo-analysis pipeline.

Usage:
    # Analyse repo from the CrOSSD API (auto-selects newest snapshot)
    python main.py lorabridge/lorabridge

    # Use a specific crawl-snapshot timestamp
    python main.py lorabridge/lorabridge --timestamp 1758139805.6506386

    # Analyse repo from a local JSON export
    python main.py --file example.json

    # Use only specific metrics
    python main.py lorabridge/lorabridge --metrics friendliness documentation_quality

    # Custom Ollama model / host
    python main.py lorabridge/lorabridge --model gemma3:27b --llm-host http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import sys

from dateutil.relativedelta import relativedelta
import datetime
from typing import Callable
import requests

from llm_metrics.config import DEFAULT_METRICS, MetricDefinition
from llm_metrics.graph import analyse_repo


def fetch_snapshots(repo_slug: str, crossd_host: str) -> list[float]:
    """Fetch the list of available crawl-snapshot timestamps for *repo_slug*.

    Calls ``POST /snapshots`` and returns the timestamps sorted ascending.
    """
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    resp = session.post(
        crossd_host.rstrip("/") + "/snapshots",
        json={"term": repo_slug},
    )
    resp.raise_for_status()
    timestamps: list[float] = resp.json()
    return sorted(timestamps)


def fetch_repo_data(
    repo_slug: str,
    crossd_host: str,
    timestamp: float | None = None,
) -> dict:
    """Fetch repository data from the CrOSSD API.

    Parameters
    ----------
    repo_slug:
        GitHub owner/repo, e.g. ``"lorabridge/lorabridge"``.
    crossd_host:
        Base URL for the CrOSSD API.
    timestamp:
        A specific crawl-snapshot timestamp.  When ``None`` the newest
        available snapshot is looked up automatically via ``/snapshots``.
    """
    if timestamp is None:
        snapshots = fetch_snapshots(repo_slug, crossd_host)
        if not snapshots:
            raise RuntimeError(
                f"No snapshots found for '{repo_slug}' – "
                "the repository may not have been crawled yet."
            )
        timestamp = snapshots[-1]  # newest
        print(f"   ℹ️  Using newest snapshot: {timestamp}")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    payload = {"term": repo_slug, "timestamp": timestamp}
    resp = session.post(crossd_host.rstrip("/") + "/repo", json=payload)
    resp.raise_for_status()
    return resp.json()


def load_repo_data(path: str) -> dict:
    """Load repository data from a local JSON file."""
    with open(path) as f:
        return json.load(f)


def select_metrics(names: list[str] | None) -> list[MetricDefinition]:
    """Filter ``DEFAULT_METRICS`` by name. Returns all if *names* is ``None``."""
    if names is None:
        return DEFAULT_METRICS
    lookup = {m.name: m for m in DEFAULT_METRICS}
    selected = []
    for n in names:
        if n not in lookup:
            print(f"⚠️  Unknown metric '{n}', skipping. Available: {list(lookup)}")
            continue
        selected.append(lookup[n])
    if not selected:
        print("No valid metrics selected – using all defaults.")
        return DEFAULT_METRICS
    return selected


def print_report(results: list[dict]) -> None:
    """Pretty-print the analysis results."""
    print("\n" + "=" * 70)
    print("  REPOSITORY HEALTH REPORT")
    print("=" * 70)
    for r in results:
        score = r.get("metric", "N/A")
        pipeline_tag = "🗣️ debate" if r["pipeline"] == "debate" else "📝 single"
        print(f"\n  [{pipeline_tag}] {r['display_name']}: {score}")
        print(f"  {'─' * 60}")
        explanation = r.get("explanation", "")
        # Wrap long explanations
        for line in explanation.split("\n"):
            print(f"    {line}")
        if r.get("debate_log"):
            print(f"\n    💬 Debate had {len(r['debate_log'])} turns.")
    print("\n" + "=" * 70)


def date_filter(
    data: list[dict],
    selector: Callable[[dict], str],
    since: datetime.datetime | relativedelta,
) -> list[dict]:
    """
    Filter a list of dictionaries based on a date.

    Args:
        data: list[dict]: The list of dictionaries to be filtered.
        selector: Callable: A function that takes a dictionary and returns a date ISO8601 string.
        since: datetime.datetime | datetime.timedelta: The date or time duration to filter the data. (Default value = datetime.timedelta(days=30 * 6))
    Returns:
        list[dict]: A list of dictionaries that match the date filter.
    """
    if type(since) == datetime.datetime:
        # if since is a datetime object, use it as is
        past = since
    elif type(since) == relativedelta:
        # if since is a timedelta object, use it to calculate the past date
        past = datetime.datetime.now(datetime.UTC) - since
    else:
        raise TypeError("since not of type datetime.datetime, dateutil.relativedelta.relativedelta")

    res = []
    for elem in data:
        if datetime.datetime.fromisoformat(selector(elem)) > past:
            res.append(elem)
    return res

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse GitHub repository health using LLM agents."
    )
    parser.add_argument(
        "repo",
        nargs="?",
        help='GitHub repo slug, e.g. "lorabridge/lorabridge"',
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to a local JSON export instead of fetching from the API.",
    )
    parser.add_argument(
        "--metrics", "-m",
        nargs="*",
        help="Metric names to evaluate (default: all).",
    )
    parser.add_argument(
        "--model",
        default="gemma3:27b",
        help="Ollama model name (default: gemma3:27b).",
    )
    parser.add_argument(
        "--llm-host",
        default="http://localhost:11434",
        help="Ollama API base URL.",
    )
    parser.add_argument(
        "--llm-auth-token",
        help="Authentication token for the Ollama API.",
    )
    parser.add_argument(
        "--crossd-host",
        default="https://health.crossd.tech/api",
        help="CrOSSD API base URL.",
    )
    parser.add_argument(
        "--timestamp", "-t",
        type=float,
        default=None,
        help=(
            "Crawl-snapshot timestamp to use.  If omitted the newest "
            "available snapshot is selected automatically."
        ),
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON instead of a formatted report.",
    )
    parser.add_argument(
        "--log-dir",
        default="debate_logs",
        help="Directory where debate logs are saved (default: debate_logs/).",
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    snapshot_timestamp: float | None = None
    if args.file:
        print(f"📂 Loading repo data from {args.file} …")
        repo_data = load_repo_data(args.file)
        # Best-effort: extract repo name from the JSON itself
        repo_slug = (
            repo_data.get("repository", {})
            .get("repository", {})
            .get("nameWithOwner", args.file)
        )
    elif args.repo:
        print(f"🌐 Fetching repo data for {args.repo} …")
        repo_data = fetch_repo_data(args.repo, args.crossd_host, args.timestamp)
        repo_slug = args.repo
        snapshot_timestamp = args.timestamp  # may still be None if auto-selected
        # If auto-selected, recover the timestamp from the returned data
        if snapshot_timestamp is None:
            snapshot_timestamp = (
                repo_data.get("repository", {}).get("timestamp")
                or repo_data.get("timestamp")
            )
    else:
        parser.error("Provide either a repo slug or --file path.")

    # ── Select metrics ───────────────────────────────────────────────────
    metrics = select_metrics(args.metrics)
    print(f"📊 Evaluating {len(metrics)} metric(s): {[m.name for m in metrics]}\n")

    issues_data = date_filter(
        repo_data["repository"]["repository"]["issues"]["edges"],
        lambda x: x["node"]["updatedAt"],
        # datetime.timedelta(days=30 * 3),
        relativedelta(weeks=12),
    )
    # print(issues_data)
    print(f"   ℹ️  {len(issues_data)} issues updated in the last week.")
    # exit()
    # print(f"   ℹ️  {len(repo_data['repository']['repository']['issues']['edges'])} total issues.")
    numbers = tuple("issue"+str(x['node']['number']) for x in issues_data)
    repo_data['repository']['repository']['issues']['edges'] = issues_data
    for key in list(repo_data['repository']['repository'].keys()):
        if key != "issues" and key.startswith("issue") and key not in numbers:
            del repo_data['repository']['repository'][key]
    repo_data["repository"]["organizations"] = [{"login":repo_data["repository"]["organizations"][u]["login"],"organizations":repo_data["repository"]["organizations"][u]["organizations"]["nodes"]} for u in repo_data["repository"]["organizations"] if "login" in repo_data["repository"]["organizations"][u]]

    repo_data["repository"]["repository"]["releases"]["edges"] = [e["node"] for e in repo_data["repository"]["repository"]["releases"]["edges"]]

    repo_data["repository"]["repository"]["branches"]["edges"] = [e["branch"] for e in repo_data["repository"]["repository"]["branches"]["edges"]]

    repo_data["repository"]["repository"]["pullRequests"]["edges"] = [e["node"] for e in repo_data["repository"]["repository"]["pullRequests"]["edges"]]

    repo_data["repository"]["repository"]["issues"]["edges"] = [e["node"] for e in repo_data["repository"]["repository"]["issues"]["edges"]]


    # ── Run pipeline ─────────────────────────────────────────────────────
    results = analyse_repo(
        repo_data,
        # [DEFAULT_METRICS[3]],
        metrics,
        model=args.model,
        llm_host=args.llm_host,
        llm_auth_token=args.llm_auth_token,
        repo_slug=repo_slug,
        snapshot_timestamp=snapshot_timestamp,
        log_dir=args.log_dir,
    )

    # ── Output ───────────────────────────────────────────────────────────
    if args.json:
        res = {elem["name"]: elem for elem in results}
        print(json.dumps(res, indent=2, default=str))
        # print(json.dumps(results, indent=2, default=str))
    else:
        print_report(results)


if __name__ == "__main__":
    main()
