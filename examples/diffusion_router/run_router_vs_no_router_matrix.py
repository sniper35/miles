#!/usr/bin/env python3
"""
Run no-router and router benchmarks into a separate output root, then compare.

This avoids overwriting existing outputs/ folders by default:
  outputs/router_eval_<YYYYmmdd_HHMMSS>/

usage:

  MODEL="${MODEL:-stabilityai/stable-diffusion-3-medium-diffusers}"

  for g in 8; do
    for k in 3; do
      c=$((k*g))
      p=$((80*c))
      python examples/diffusion_router/run_router_vs_no_router_matrix.py \
        --model "$MODEL" \
        --gpu-counts $g \
        --configs ${p}x${c} \
        --output-root outputs/router_eval_k${k}_g${g} \
        --no-router-worker-base-port $((41000 + 500*g + 20*k)) \
        --router-worker-base-port $((11090 + 500*g + 20*k)) \
        --no-router-bench-status-interval 20 \
        --no-router-bench-shard-timeout 7200 \
        --continue-on-error
    done
  done
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _require_non_empty_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError(
            "--model must be a non-empty model ID/path. "
            "Detected an empty value, which often means a shell variable such as "
            "$MODEL was unset."
        )
    return normalized


def _parse_config_entry(entry: str) -> tuple[int, int]:
    for sep in ("x", "X", ":", ","):
        if sep in entry:
            left, right = entry.split(sep, 1)
            p = int(left.strip())
            c = int(right.strip())
            if p < 1 or c < 1:
                raise ValueError(f"Invalid config '{entry}': values must be >= 1")
            return p, c
    raise ValueError(f"Invalid config '{entry}'. Use formats like '20x4' or '50:8'.")


def _run(cmd: list[str], dry_run: bool) -> int:
    print("[run]", " ".join(shlex.quote(x) for x in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run router-vs-no-router diffusion benchmark matrix into an isolated output root."
    )
    parser.add_argument("--model", type=str, required=True, help="Model ID or local model path.")
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[2, 4, 6, 8], help="GPU counts to run.")
    parser.add_argument("--configs", nargs="+", type=str, default=["20x4", "50x8", "100x16"])
    parser.add_argument("--output-root", type=str, default=None, help="Output root directory.")
    parser.add_argument(
        "--allow-existing-output-root",
        action="store_true",
        help="Allow writing into an existing output root directory.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running remaining commands on error.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")

    parser.add_argument("--sglang-root", type=str, default=None, help="Path to sglang repo.")
    parser.add_argument("--num-gpus-per-worker", type=int, default=1)
    parser.add_argument(
        "--no-router-worker-base-port",
        type=int,
        default=31000,
        help="Base worker port for bench_no_router.py",
    )
    parser.add_argument(
        "--router-worker-base-port",
        type=int,
        default=10090,
        help="Base worker port for bench_router.py",
    )
    parser.add_argument(
        "--worker-port-stride",
        type=int,
        default=2,
        help="Worker HTTP port stride for both router and no-router runs.",
    )
    parser.add_argument(
        "--worker-internal-port-stride",
        type=int,
        default=1000,
        help="Internal sglang port stride for both router and no-router runs.",
    )
    parser.add_argument("--dataset", type=str, default="random", choices=["vbench", "random"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--task", type=str, default=None, help="Optional task override passed to benchmark scripts.")
    parser.add_argument("--width", type=int, default=None, help="Image/Video width for benchmark requests.")
    parser.add_argument("--height", type=int, default=None, help="Image/Video height for benchmark requests.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames (video models).")
    parser.add_argument("--fps", type=int, default=None, help="FPS (video models).")
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--wait-timeout", type=int, default=1200)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--no-router-bench-status-interval",
        type=int,
        default=30,
        help="Seconds between no-router shard progress status prints.",
    )
    parser.add_argument(
        "--no-router-bench-shard-timeout",
        type=int,
        default=0,
        help="Per-shard timeout (seconds) for no-router bench stage; 0 disables timeout.",
    )

    parser.add_argument("--worker-extra-args", type=str, default="")
    parser.add_argument("--router-extra-args", type=str, default="")
    parser.add_argument("--bench-extra-args", type=str, default="")
    parser.add_argument("--router-verbose", action="store_true")

    args = parser.parse_args()
    args.model = _require_non_empty_model(args.model)

    parsed_configs = [_parse_config_entry(x) for x in args.configs]

    root = (
        Path(args.output_root)
        if args.output_root
        else Path("outputs") / f"router_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    no_router_dir = root / "no_router_matrix"
    router_dir = root / "router_matrix"
    compare_all = root / "perf_compare_router_vs_no_router_all.csv"
    compare_matched = root / "perf_compare_router_vs_no_router_matched.csv"

    if root.exists() and any(root.iterdir()) and not args.allow_existing_output_root:
        raise RuntimeError(
            f"Output root already exists and is non-empty: {root}. "
            "Use --output-root with a new path, or pass --allow-existing-output-root."
        )
    no_router_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    failures: list[tuple[str, int]] = []

    # 1) Run no-router matrix once.
    no_router_cmd = [
        py,
        "examples/diffusion_router/bench_no_router.py",
        "--model",
        args.model,
        "--gpu-counts",
        *[str(x) for x in args.gpu_counts],
        "--configs",
        *args.configs,
        "--num-gpus-per-worker",
        str(args.num_gpus_per_worker),
        "--worker-base-port",
        str(args.no_router_worker_base_port),
        "--worker-port-stride",
        str(args.worker_port_stride),
        "--worker-internal-port-stride",
        str(args.worker_internal_port_stride),
        "--dataset",
        args.dataset,
        "--request-rate",
        str(args.request_rate),
        "--wait-timeout",
        str(args.wait_timeout),
        "--bench-status-interval",
        str(args.no_router_bench_status_interval),
        "--bench-shard-timeout",
        str(args.no_router_bench_shard_timeout),
        "--log-level",
        args.log_level,
        "--output-dir",
        str(no_router_dir),
    ]
    if args.sglang_root:
        no_router_cmd += ["--sglang-root", args.sglang_root]
    if args.dataset_path:
        no_router_cmd += ["--dataset-path", args.dataset_path]
    if args.task:
        no_router_cmd += ["--task", args.task]
    if args.width:
        no_router_cmd += ["--width", str(args.width)]
    if args.height:
        no_router_cmd += ["--height", str(args.height)]
    if args.num_frames:
        no_router_cmd += ["--num-frames", str(args.num_frames)]
    if args.fps:
        no_router_cmd += ["--fps", str(args.fps)]
    if args.disable_tqdm:
        no_router_cmd.append("--disable-tqdm")
    if args.worker_extra_args:
        no_router_cmd += ["--worker-extra-args", args.worker_extra_args]

    rc = _run(no_router_cmd, args.dry_run)
    if rc != 0:
        failures.append(("bench_no_router_matrix", rc))
        if not args.continue_on_error:
            return rc

    # 2) Run router for each GPU/config pair.
    for g in args.gpu_counts:
        for prompts, concurrency in parsed_configs:
            out_file = router_dir / f"bench_router_g{g}_p{prompts}_c{concurrency}.json"
            router_cmd = [
                py,
                "examples/diffusion_router/bench_router.py",
                "--model",
                args.model,
                "--num-workers",
                str(g),
                "--num-prompts",
                str(prompts),
                "--max-concurrency",
                str(concurrency),
                "--num-gpus-per-worker",
                str(args.num_gpus_per_worker),
                "--worker-base-port",
                str(args.router_worker_base_port),
                "--worker-port-stride",
                str(args.worker_port_stride),
                "--worker-internal-port-stride",
                str(args.worker_internal_port_stride),
                "--dataset",
                args.dataset,
                "--request-rate",
                str(args.request_rate),
                "--wait-timeout",
                str(args.wait_timeout),
                "--log-level",
                args.log_level,
                "--output-file",
                str(out_file),
            ]
            if args.sglang_root:
                router_cmd += ["--sglang-root", args.sglang_root]
            if args.dataset_path:
                router_cmd += ["--dataset-path", args.dataset_path]
            if args.task:
                router_cmd += ["--task", args.task]
            if args.width:
                router_cmd += ["--width", str(args.width)]
            if args.height:
                router_cmd += ["--height", str(args.height)]
            if args.num_frames:
                router_cmd += ["--num-frames", str(args.num_frames)]
            if args.fps:
                router_cmd += ["--fps", str(args.fps)]
            if args.disable_tqdm:
                router_cmd.append("--disable-tqdm")
            if args.router_verbose:
                router_cmd.append("--router-verbose")
            if args.worker_extra_args:
                router_cmd += ["--worker-extra-args", args.worker_extra_args]
            if args.router_extra_args:
                router_cmd += ["--router-extra-args", args.router_extra_args]
            if args.bench_extra_args:
                router_cmd += ["--bench-extra-args", args.bench_extra_args]

            rc = _run(router_cmd, args.dry_run)
            if rc != 0:
                failures.append((f"bench_router_g{g}_p{prompts}_c{concurrency}", rc))
                if not args.continue_on_error:
                    return rc

    # 3) Build comparison CSVs.
    compare_cmd = [
        py,
        "examples/diffusion_router/compare_router_vs_no_router.py",
        "--router-dir",
        str(router_dir),
        "--no-router-dir",
        str(no_router_dir),
        "--all-output",
        str(compare_all),
        "--matched-output",
        str(compare_matched),
    ]
    rc = _run(compare_cmd, args.dry_run)
    if rc != 0:
        failures.append(("compare_router_vs_no_router", rc))
        if not args.continue_on_error:
            return rc

    print(f"[done] output_root={root}", flush=True)
    print(f"[done] no_router_dir={no_router_dir}", flush=True)
    print(f"[done] router_dir={router_dir}", flush=True)
    print(f"[done] compare_all={compare_all}", flush=True)
    print(f"[done] compare_matched={compare_matched}", flush=True)

    if failures:
        print("[done] failures:", flush=True)
        for name, code in failures:
            print(f"  - {name}: exit_code={code}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
