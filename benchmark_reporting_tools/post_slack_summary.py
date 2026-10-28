#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post a summary of a nightly benchmark to Slack.

Reads {nightly_dir}/{run}/benchmark_result.json for each requested run,
compares per-query median times against the most recent earlier nightly
directory that has results for the same run, and links to the
BenchmarkingDB run recorded by post_results.py in posted_runs.json.

Expected layout (as written by presto/slurm/presto-nvl72/nightly-benchmark.sh):

    {results_base}/nightly-20261027/tpch-1k/benchmark_result.json
    {results_base}/nightly-20261027/tpch-1k/posted_runs.json
    {results_base}/nightly-20261026/tpch-1k/benchmark_result.json   <- baseline

Usage:
    python benchmark_reporting_tools/post_slack_summary.py results/nightly-20261027 \
        --run tpch-1k --run tpch-3k \
        --note "Worker image: presto-latest-gpu-cuda13.1"

    # Print the message instead of sending it:
    python benchmark_reporting_tools/post_slack_summary.py results/nightly-20261027 \
        --run tpch-1k --run tpch-3k --dry-run

Environment variables:
    SLACK_WEBHOOK_URL: Slack incoming webhook URL (required unless --dry-run)

"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.workflow_status.lib.slack import build_payload, send_webhook

RESULT_FILE = "benchmark_result.json"
# Written by post_results.py after a successful submission.
POSTED_RUNS_FILE = "posted_runs.json"
NIGHTLY_DIR_PATTERN = re.compile(r"^nightly-(\d{8})$")
# Written by the slurm launcher to presto-tpch-run_*.out, e.g.
#   "Starting Presto coordinator on presto-gb200-gcn-05..."
#   "  Starting worker 7 on node presto-gb200-gcn-12 GPU 3"
COORDINATOR_NODE_PATTERN = re.compile(r"Starting Presto coordinator on (\S+?)\.{0,3}\s*$", re.MULTILINE)
WORKER_NODE_PATTERN = re.compile(r"Starting worker \d+ on node (\S+)")


@dataclass
class RunResult:
    name: str
    query_times_ms: dict[str, float] = field(default_factory=dict)
    failed_queries: dict[str, str] = field(default_factory=dict)
    context: dict = field(default_factory=dict)


def _query_sort_key(name: str):
    match = re.match(r"(\d+)(.*)", name.lstrip("Qq"))
    if match:
        return (int(match.group(1)), match.group(2))
    return (float("inf"), name)


def _query_time_ms(query: str, bench: dict) -> float | None:
    """Median time for *query*, preferring the aggregate written by the benchmark runner."""
    agg = bench.get("agg_times_ms", {})
    for key in ("median", "avg"):
        value = agg.get(key, {}).get(query)
        if value is not None:
            return value
    raw = [t for t in bench.get("raw_times_ms", {}).get(query) or [] if t is not None]
    return statistics.median(raw) if raw else None


def load_run(run_dir: Path) -> RunResult | None:
    """Load benchmark_result.json from *run_dir*, or None if it is missing or unreadable."""
    try:
        raw = json.loads((run_dir / RESULT_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    result = RunResult(name=run_dir.name, context=raw.get("context", {}))
    benchmarks = result.context.get("benchmark") or [k for k in raw if k != "context"]
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]
    for bench_name in benchmarks:
        bench = raw.get(bench_name, {})
        result.failed_queries.update(bench.get("failed_queries", {}))
        for query in bench.get("raw_times_ms", {}):
            if query in result.failed_queries:
                continue
            time_ms = _query_time_ms(query, bench)
            if time_ms is not None:
                result.query_times_ms[query] = time_ms
    return result


def load_posted_urls(run_dir: Path) -> list[str]:
    try:
        posted = json.loads((run_dir / POSTED_RUNS_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [entry["url"] for entry in posted.values() if isinstance(entry, dict) and entry.get("url")]


def load_nodes(run_dir: Path) -> tuple[str | None, list[str]]:
    """Return (coordinator node, worker nodes in first-seen order) from the slurm job output in *run_dir*."""
    coordinator = None
    workers: list[str] = []
    for out_file in sorted(run_dir.glob("presto-*.out")):
        try:
            text = out_file.read_text(errors="replace")
        except OSError:
            continue
        match = COORDINATOR_NODE_PATTERN.search(text)
        if match and coordinator is None:
            coordinator = match.group(1)
        for node in WORKER_NODE_PATTERN.findall(text):
            if node not in workers:
                workers.append(node)
    return coordinator, workers


def _fmt_nodes(coordinator: str | None, workers: list[str]) -> str | None:
    nodes = list(workers)
    if coordinator and coordinator not in nodes:
        nodes.insert(0, coordinator)
    if not nodes:
        return None
    return ", ".join(f"`{node}`" + (" (coordinator)" if node == coordinator else "") for node in nodes)


def find_baseline(nightly_dir: Path, run_name: str) -> tuple[str, RunResult] | None:
    """Return (date, result) for the most recent nightly before *nightly_dir* with results for *run_name*."""
    match = NIGHTLY_DIR_PATTERN.match(nightly_dir.name)
    if not match or not nightly_dir.parent.is_dir():
        return None
    current_date = match.group(1)
    earlier = []
    for candidate in nightly_dir.parent.iterdir():
        candidate_match = NIGHTLY_DIR_PATTERN.match(candidate.name)
        if candidate_match and candidate_match.group(1) < current_date:
            earlier.append((candidate_match.group(1), candidate))
    for date, candidate in sorted(earlier, reverse=True):
        result = load_run(candidate / run_name)
        if result is not None and result.query_times_ms:
            return date, result
    return None


def _fmt_seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def _fmt_pct(new: float, old: float) -> str:
    return f"{(new - old) / old * 100:+.1f}%"


def _fmt_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def summarize_run(
    run_name: str,
    run_dir: Path,
    nightly_dir: Path,
    threshold_pct: float,
    min_delta_ms: float,
) -> tuple[str, str]:
    """Return (status, mrkdwn section) for one run; status is "ok", "warn" or "fail"."""
    result = load_run(run_dir)
    if result is None:
        return "fail", f":x: *{run_name}*: no results (`{RESULT_FILE}` missing)"

    ctx = result.context
    shape = ""
    if ctx.get("node_count") and ctx.get("gpu_count"):
        shape = f" ({ctx['node_count']} node(s), {ctx['gpu_count']} GPUs)"

    urls = load_posted_urls(run_dir)
    if urls:
        db_link = " · ".join(f"<{url}|BenchmarkingDB run {url.rstrip('/').rsplit('/', 1)[-1]}>" for url in urls)
    else:
        db_link = "_not posted to BenchmarkingDB_"

    total_queries = len(result.query_times_ms) + len(result.failed_queries)
    status = "warn" if result.failed_queries or not urls else "ok"
    lines = [f"*{run_name}*{shape} · {db_link}"]
    nodes = _fmt_nodes(*load_nodes(run_dir))
    if nodes:
        lines.append(f"Nodes: {nodes}")

    total_ms = sum(result.query_times_ms.values())
    total_line = f"Total: *{_fmt_seconds(total_ms)}* · {len(result.query_times_ms)}/{total_queries} queries passed"

    baseline = find_baseline(nightly_dir, run_name)
    changes = []
    if baseline is not None:
        baseline_date, baseline_result = baseline
        common = [q for q in result.query_times_ms if q in baseline_result.query_times_ms]
        new_total = sum(result.query_times_ms[q] for q in common)
        old_total = sum(baseline_result.query_times_ms[q] for q in common)
        if old_total > 0:
            total_line += f" · {_fmt_pct(new_total, old_total)} vs {_fmt_date(baseline_date)}"
            if len(common) != len(result.query_times_ms):
                total_line += f" (over {len(common)} common queries)"
        for query in common:
            new, old = result.query_times_ms[query], baseline_result.query_times_ms[query]
            if old > 0 and abs(new - old) >= min_delta_ms and abs(new - old) / old * 100 >= threshold_pct:
                changes.append((query, old, new))
    else:
        total_line += " · no earlier nightly to compare against"
    lines.append(total_line)

    if changes:
        if any(new > old for _, old, new in changes):
            status = "warn"
        lines.append(f"Queries that changed by more than {threshold_pct:g}%:")
        rows = [
            f"{'Q' + q.lstrip('Qq'):<6}{_fmt_seconds(old):>10} -> {_fmt_seconds(new):>10}  {_fmt_pct(new, old):>8}"
            for q, old, new in sorted(changes, key=lambda c: _query_sort_key(c[0]))
        ]
        lines.append("```\n" + "\n".join(rows) + "\n```")

    if result.failed_queries:
        lines.append("Failed queries:")
        rows = [
            f"{'Q' + q.lstrip('Qq'):<6}{error}"
            for q, error in sorted(result.failed_queries.items(), key=lambda item: _query_sort_key(item[0]))
        ]
        lines.append("```\n" + "\n".join(rows) + "\n```")

    return status, "\n".join(lines)


def build_message(args: argparse.Namespace) -> str:
    nightly_dir = Path(args.nightly_dir)
    run_names = args.runs or (
        sorted(p.name for p in nightly_dir.iterdir() if p.is_dir()) if nightly_dir.is_dir() else []
    )

    statuses = []
    sections = []
    for run_name in run_names:
        status, section = summarize_run(
            run_name, nightly_dir / run_name, nightly_dir, args.threshold_pct, args.min_delta_ms
        )
        statuses.append(status)
        sections.append(section)

    if not sections:
        statuses.append("fail")
        sections.append(f":x: No benchmark runs found in `{nightly_dir}`")

    if "fail" in statuses:
        emoji = ":x:"
    elif "warn" in statuses:
        emoji = ":warning:"
    else:
        emoji = ":white_check_mark:"

    match = NIGHTLY_DIR_PATTERN.match(nightly_dir.name)
    title = args.title
    if match:
        title += f" · {_fmt_date(match.group(1))}"
    header = [f"{emoji} *{title}*"]
    header.extend(args.notes)

    # Use the first loaded run's context for the commits under test.
    for run_name in run_names:
        result = load_run(nightly_dir / run_name)
        if result is None:
            continue
        shas = [
            f"{repo} `{result.context[key][:12]}`"
            for repo, key in (("presto", "presto_sha"), ("velox", "velox_sha"))
            if result.context.get(key)
        ]
        if shas:
            header.append("Commits: " + " · ".join(shas))
        break

    return "\n---\n".join(["\n".join(header), *sections])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post a nightly benchmark summary to Slack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("nightly_dir", help="Nightly results directory, e.g. results/nightly-20261027")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=[],
        help="Run subdirectory to summarize (repeatable). Defaults to every subdirectory of nightly_dir.",
    )
    parser.add_argument("--title", default="Presto nightly benchmark", help="Message title")
    parser.add_argument(
        "--note", dest="notes", action="append", default=[], help="Extra line for the message header (repeatable)"
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=10.0,
        help="Report queries whose time changed by at least this percentage (default: 10)",
    )
    parser.add_argument(
        "--min-delta-ms",
        type=float,
        default=100.0,
        help="Ignore changes smaller than this many milliseconds, to filter noise on short queries (default: 100)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the message instead of sending it")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    message = build_message(args)
    if args.dry_run:
        print(message)
        return 0
    return 0 if send_webhook(build_payload(message), os.environ.get("SLACK_WEBHOOK_URL", "")) else 1


if __name__ == "__main__":
    sys.exit(main())
