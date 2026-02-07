#!/usr/bin/env python3
"""
Build router vs no-router performance comparison tables from benchmark outputs.

By default it reads:
- outputs/router_matrix/bench_router_g*_p*_c*.json
- outputs/no_router_matrix/bench_g*_p*_c*_aggregate.json

And writes:
- outputs/router_matrix/perf_compare_router_vs_no_router_all.csv
- outputs/router_matrix/perf_compare_router_vs_no_router_matched.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchPoint:
    gpus: int
    prompts: int
    concurrency: int
    throughput_qps: float
    latency_mean_s: float
    completed_requests: int
    failed_requests: int
    source_file: str


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_router_file(path: Path) -> tuple[tuple[int, int, int], BenchPoint] | None:
    m = re.search(r"bench_router_g(\d+)_p(\d+)_c(\d+)\.json$", path.name)
    if not m:
        return None

    g, p, c = map(int, m.groups())
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    point = BenchPoint(
        gpus=g,
        prompts=p,
        concurrency=c,
        throughput_qps=_safe_float(d.get("throughput_qps", d.get("throughput_qps_sum", 0.0))),
        latency_mean_s=_safe_float(d.get("latency_mean", d.get("latency_mean_weighted_s", 0.0))),
        completed_requests=_safe_int(d.get("completed_requests", 0)),
        failed_requests=_safe_int(d.get("failed_requests", 0)),
        source_file=str(path),
    )
    return (g, p, c), point


def _parse_no_router_file(path: Path) -> tuple[tuple[int, int, int], BenchPoint] | None:
    m = re.search(r"bench_g(\d+)_p(\d+)_c(\d+)_aggregate\.json$", path.name)
    if not m:
        return None

    g, p, c = map(int, m.groups())
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    point = BenchPoint(
        gpus=g,
        prompts=p,
        concurrency=c,
        throughput_qps=_safe_float(d.get("throughput_qps_sum", d.get("throughput_qps", 0.0))),
        latency_mean_s=_safe_float(d.get("latency_mean_weighted_s", d.get("latency_mean", 0.0))),
        completed_requests=_safe_int(d.get("completed_requests", 0)),
        failed_requests=_safe_int(d.get("failed_requests", 0)),
        source_file=str(path),
    )
    return (g, p, c), point


def _pct_delta(new_value: float | None, base_value: float | None) -> str:
    if new_value is None or base_value is None:
        return ""
    if base_value == 0:
        return ""
    delta = (new_value - base_value) / base_value * 100.0
    return f"{delta:+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare router vs no-router benchmark metrics.")
    parser.add_argument(
        "--router-dir",
        type=str,
        default="outputs/router_matrix",
        help="Directory containing router benchmark JSON files.",
    )
    parser.add_argument(
        "--no-router-dir",
        type=str,
        default="outputs/no_router_matrix",
        help="Directory containing no-router aggregate JSON files.",
    )
    parser.add_argument(
        "--all-output",
        type=str,
        default="outputs/router_matrix/perf_compare_router_vs_no_router_all.csv",
        help="Output CSV path including matched and missing rows.",
    )
    parser.add_argument(
        "--matched-output",
        type=str,
        default="outputs/router_matrix/perf_compare_router_vs_no_router_matched.csv",
        help="Output CSV path for matched rows only.",
    )
    args = parser.parse_args()

    router_dir = Path(args.router_dir)
    no_router_dir = Path(args.no_router_dir)
    all_output = Path(args.all_output)
    matched_output = Path(args.matched_output)

    router_points: dict[tuple[int, int, int], BenchPoint] = {}
    for p in sorted(router_dir.glob("bench_router_g*_p*_c*.json")):
        parsed = _parse_router_file(p)
        if parsed is None:
            continue
        key, point = parsed
        router_points[key] = point

    no_router_points: dict[tuple[int, int, int], BenchPoint] = {}
    for p in sorted(no_router_dir.glob("bench_g*_p*_c*_aggregate.json")):
        parsed = _parse_no_router_file(p)
        if parsed is None:
            continue
        key, point = parsed
        no_router_points[key] = point

    keys = sorted(set(router_points) | set(no_router_points))
    rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []

    for g, p, c in keys:
        router = router_points.get((g, p, c))
        no_router = no_router_points.get((g, p, c))

        if router and no_router:
            status = "matched"
        elif router and not no_router:
            status = "missing_no_router"
        else:
            status = "missing_router"

        row = {
            "status": status,
            "gpus": g,
            "num_prompts": p,
            "max_concurrency": c,
            "config": f"{p}x{c}",
            "no_router_throughput_qps": f"{no_router.throughput_qps:.6f}" if no_router else "",
            "router_throughput_qps": f"{router.throughput_qps:.6f}" if router else "",
            "router_delta_throughput_pct": _pct_delta(
                router.throughput_qps if router else None,
                no_router.throughput_qps if no_router else None,
            ),
            "no_router_latency_mean_s": f"{no_router.latency_mean_s:.6f}" if no_router else "",
            "router_latency_mean_s": f"{router.latency_mean_s:.6f}" if router else "",
            "router_delta_latency_pct": _pct_delta(
                router.latency_mean_s if router else None,
                no_router.latency_mean_s if no_router else None,
            ),
            "no_router_completed": no_router.completed_requests if no_router else "",
            "router_completed": router.completed_requests if router else "",
            "no_router_failed": no_router.failed_requests if no_router else "",
            "router_failed": router.failed_requests if router else "",
            "no_router_source_file": no_router.source_file if no_router else "",
            "router_source_file": router.source_file if router else "",
        }
        rows.append(row)
        if status == "matched":
            matched_rows.append(row)

    all_output.parent.mkdir(parents=True, exist_ok=True)
    matched_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "status",
        "gpus",
        "num_prompts",
        "max_concurrency",
        "config",
        "no_router_throughput_qps",
        "router_throughput_qps",
        "router_delta_throughput_pct",
        "no_router_latency_mean_s",
        "router_latency_mean_s",
        "router_delta_latency_pct",
        "no_router_completed",
        "router_completed",
        "no_router_failed",
        "router_failed",
        "no_router_source_file",
        "router_source_file",
    ]

    with all_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with matched_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched_rows)

    print(f"wrote: {all_output}")
    print(f"wrote: {matched_output}")
    print(f"rows_all={len(rows)} rows_matched={len(matched_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
