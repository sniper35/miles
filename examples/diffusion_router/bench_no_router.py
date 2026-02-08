#!/usr/bin/env python3
"""
Benchmark sglang diffusion serving without the Miles diffusion router.

Strategy:
1) Launch multiple standalone sglang workers (typically 1 GPU each).
2) For each benchmark config, shard requests/concurrency across workers.
3) Run one bench_serving client per worker in parallel.
4) Aggregate shard metrics into a single JSON.

Example:
  python examples/diffusion_router/bench_no_router.py \
    --model stabilityai/stable-diffusion-3-medium-diffusers \
    --gpu-counts 2 4 6 8 \
    --configs 20x4 50x8 100x16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import requests


def _require_non_empty_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise ValueError(
            "--model must be a non-empty model ID/path. "
            "Detected an empty value, which often means a shell variable such as "
            "$MODEL was unset."
        )
    return normalized


def _default_sglang_root() -> Path:
    # Repo layout: miles/examples/diffusion_router/bench_no_router.py -> miles/ (parents[2])
    return Path(__file__).resolve().parents[2].parent / "sglang"


def _resolve_sglang_root(path: str | None) -> Path | None:
    if path:
        root = Path(path).expanduser().resolve()
        sglang_pkg = root / "python" / "sglang"
        if not sglang_pkg.exists():
            raise FileNotFoundError(f"sglang source not found at {root}. Expected {sglang_pkg}.")
        return root
    default = _default_sglang_root()
    if (default / "python" / "sglang").exists():
        return default
    return None


def _with_pythonpath(env: dict[str, str], extra_path: Path) -> dict[str, str]:
    env = dict(env)
    existing = env.get("PYTHONPATH")
    extra = str(extra_path)
    env["PYTHONPATH"] = f"{extra}{os.pathsep}{existing}" if existing else extra
    return env


def _build_sglang_cli_cmd() -> list[str]:
    sglang_bin = Path(sys.executable).resolve().parent / "sglang"
    if sglang_bin.exists():
        return [str(sglang_bin)]
    return [sys.executable, "-c", "from sglang.cli.main import main; main()"]


def _parse_gpu_id_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _detect_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _resolve_gpu_pool(args: argparse.Namespace, env: dict[str, str]) -> list[str]:
    if args.gpu_ids:
        return [str(x) for x in args.gpu_ids]

    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        parsed = _parse_gpu_id_list(visible)
        if parsed:
            return parsed

    gpu_count = _detect_gpu_count()
    return [str(i) for i in range(gpu_count)]


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _reserve_available_port(host: str, preferred_port: int, used_ports: set[int]) -> int:
    if preferred_port < 1 or preferred_port > 65535:
        raise ValueError(f"Invalid port: {preferred_port}")

    for port in range(preferred_port, 65536):
        if port in used_ports:
            continue
        if _is_port_available(host, port):
            used_ports.add(port)
            return port

    for port in range(1024, preferred_port):
        if port in used_ports:
            continue
        if _is_port_available(host, port):
            used_ports.add(port)
            return port

    raise RuntimeError(
        f"Unable to reserve a free port for host {host}. "
        f"Preferred start={preferred_port}."
    )


def _wait_for_health(
    url: str, timeout: int, label: str, proc: subprocess.Popen | None = None,
) -> None:
    start = time.time()
    last_print = 0.0
    while True:
        elapsed = time.time() - start

        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{label} process exited with code {proc.returncode}. "
                "Check the worker log for details."
            )

        try:
            resp = requests.get(f"{url}/health", timeout=1)
            if resp.status_code == 200:
                print(f"  [no-router] {label} is healthy ({elapsed:.0f}s)", flush=True)
                return
        except requests.RequestException:
            pass

        if elapsed - last_print >= 30:
            print(f"  [no-router] Still waiting for {label}... ({elapsed:.0f}s elapsed)", flush=True)
            last_print = elapsed

        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for {label} at {url}.")
        time.sleep(1)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    except Exception:
        if proc.poll() is None:
            try:
                os.kill(proc.pid, sig)
            except ProcessLookupError:
                pass


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    for proc in reversed(processes):
        _signal_group(proc, signal.SIGTERM)

    for proc in reversed(processes):
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)

    for proc in reversed(processes):
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _parse_config_entry(entry: str) -> tuple[int, int]:
    for sep in ("x", "X", ":", ","):
        if sep in entry:
            left, right = entry.split(sep, 1)
            num_prompts = int(left.strip())
            max_concurrency = int(right.strip())
            if num_prompts < 1 or max_concurrency < 1:
                raise ValueError(f"Invalid config '{entry}': values must be >= 1")
            return num_prompts, max_concurrency
    raise ValueError(f"Invalid config '{entry}'. Use formats like '20x4' or '50:8'.")


def _split_positive(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _build_shards(worker_count: int, num_prompts: int, max_concurrency: int) -> tuple[list[int], list[int]]:
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if num_prompts < 1 or max_concurrency < 1:
        raise ValueError("num_prompts and max_concurrency must be >= 1")

    active_workers = min(worker_count, num_prompts, max_concurrency)
    prompt_splits = [0] * worker_count
    conc_splits = [0] * worker_count

    p_active = _split_positive(num_prompts, active_workers)
    for i, val in enumerate(p_active):
        prompt_splits[i] = val

    conc_splits_active = [1] * active_workers
    extra = max_concurrency - active_workers
    if extra > 0:
        even = extra // active_workers
        rem = extra % active_workers
        for i in range(active_workers):
            conc_splits_active[i] += even + (1 if i < rem else 0)
    for i, val in enumerate(conc_splits_active):
        conc_splits[i] = val

    return prompt_splits, conc_splits


def _weighted_latency_mean(shard_metrics: list[dict]) -> float:
    completed = sum(int(m.get("completed_requests", 0)) for m in shard_metrics)
    if completed <= 0:
        return 0.0
    total = 0.0
    for m in shard_metrics:
        count = int(m.get("completed_requests", 0))
        total += float(m.get("latency_mean", 0.0)) * count
    return total / completed


def _read_log_progress(log_file: Path) -> tuple[int, int] | None:
    """Best-effort parse of tqdm progress from a shard log tail."""
    try:
        size = log_file.stat().st_size
        with log_file.open("rb") as f:
            # Read only the tail to keep polling cheap.
            f.seek(max(0, size - 32768))
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    # Typical tqdm token in logs: "55%|...| 132/240 [..]"
    matches = re.findall(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", tail)
    if matches:
        _, done, total = matches[-1]
        return int(done), int(total)

    # If tqdm line isn't present, try summary line.
    m = re.search(r"Successful requests:\s+(\d+)/(\d+)", tail)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark sglang diffusion serving without the diffusion router.")
    parser.add_argument("--model", type=str, required=True, help="Diffusion model HF ID or local path.")
    parser.add_argument("--sglang-root", type=str, default=None, help="Path to sglang repo (default: ../sglang).")

    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[2, 4, 6, 8], help="Total GPUs per run.")
    parser.add_argument(
        "--configs",
        nargs="+",
        type=str,
        default=["20x4", "50x8", "100x16"],
        help="Benchmark configs as '<num-prompts>x<max-concurrency>'.",
    )
    parser.add_argument("--gpu-ids", nargs="*", default=None, help="Optional GPU IDs/UUIDs to use for workers.")
    parser.add_argument("--num-gpus-per-worker", type=int, default=1, help="GPUs per launched worker.")

    parser.add_argument("--worker-host", type=str, default="127.0.0.1", help="Worker bind host.")
    parser.add_argument("--worker-base-port", type=int, default=31000, help="Base worker port.")
    parser.add_argument(
        "--worker-port-stride",
        type=int,
        default=2,
        help="Port increment between workers. Keep >=2 to avoid sglang internal port collisions.",
    )
    parser.add_argument(
        "--worker-master-port-base",
        type=int,
        default=30005,
        help="Base torch distributed master port for launched workers.",
    )
    parser.add_argument(
        "--worker-scheduler-port-base",
        type=int,
        default=5555,
        help="Base scheduler port for launched workers.",
    )
    parser.add_argument(
        "--worker-internal-port-stride",
        type=int,
        default=1000,
        help=(
            "Stride used between workers for master/scheduler base ports. "
            "Use >= 101 because sglang randomizes each by +[0,100]."
        ),
    )
    parser.add_argument("--worker-extra-args", type=str, default="", help="Extra args for `sglang serve`.")

    parser.add_argument("--dataset", type=str, default="random", choices=["vbench", "random"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--wait-timeout", type=int, default=1200)
    parser.add_argument(
        "--bench-status-interval",
        type=int,
        default=30,
        help="Seconds between progress status prints while shard benchmark jobs run.",
    )
    parser.add_argument(
        "--bench-shard-timeout",
        type=int,
        default=0,
        help="Per-shard timeout in seconds (0 disables timeout).",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/no_router_matrix")

    args = parser.parse_args()
    args.model = _require_non_empty_model(args.model)

    if args.worker_port_stride < 1:
        raise ValueError("--worker-port-stride must be >= 1")
    if args.worker_internal_port_stride < 101:
        raise ValueError("--worker-internal-port-stride must be >= 101")
    if args.num_gpus_per_worker < 1:
        raise ValueError("--num-gpus-per-worker must be >= 1")

    parsed_configs = [_parse_config_entry(entry) for entry in args.configs]

    sglang_root = _resolve_sglang_root(args.sglang_root)
    if sglang_root is not None:
        env = _with_pythonpath(os.environ, sglang_root / "python")
    else:
        try:
            import sglang  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "sglang is not installed and no source repo found at ../sglang.\n"
                "Install with: uv pip install \"sglang[diffusion]\" --prerelease=allow\n"
                "Or point to the source repo with: --sglang-root /path/to/sglang"
            ) from e
        env = dict(os.environ)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_pool = _resolve_gpu_pool(args, env)
    if not gpu_pool:
        raise RuntimeError("No GPUs available. Check CUDA visibility and --gpu-ids.")

    sglang_cli_cmd = _build_sglang_cli_cmd()

    for total_gpus in args.gpu_counts:
        if total_gpus < 1:
            raise ValueError(f"Invalid gpu count: {total_gpus}")
        if total_gpus > len(gpu_pool):
            raise ValueError(
                f"Requested {total_gpus} GPUs but only {len(gpu_pool)} available from the configured pool."
            )
        if total_gpus % args.num_gpus_per_worker != 0:
            raise ValueError(
                f"gpu-count {total_gpus} must be divisible by --num-gpus-per-worker {args.num_gpus_per_worker}."
            )

        worker_count = total_gpus // args.num_gpus_per_worker
        selected = gpu_pool[:total_gpus]
        worker_gpu_groups = [
            selected[i * args.num_gpus_per_worker:(i + 1) * args.num_gpus_per_worker]
            for i in range(worker_count)
        ]
        worker_ports = [args.worker_base_port + i * args.worker_port_stride for i in range(worker_count)]
        worker_urls = [f"http://{args.worker_host}:{port}" for port in worker_ports]
        reserved_ports: set[int] = set(worker_ports)
        worker_internal_ports: list[tuple[int, int]] = []

        print(
            f"[no-router] Starting run for total_gpus={total_gpus} "
            f"(workers={worker_count}, gpus_per_worker={args.num_gpus_per_worker})",
            flush=True,
        )

        for port in worker_ports:
            if not _is_port_available(args.worker_host, port):
                raise RuntimeError(
                    f"Port {port} is already in use on {args.worker_host}. "
                    "Stop existing servers or change --worker-base-port."
                )

        for i in range(worker_count):
            preferred_master = args.worker_master_port_base + i * args.worker_internal_port_stride
            preferred_scheduler = (
                args.worker_scheduler_port_base + i * args.worker_internal_port_stride
            )
            master_port = _reserve_available_port(args.worker_host, preferred_master, reserved_ports)
            scheduler_port = _reserve_available_port(args.worker_host, preferred_scheduler, reserved_ports)
            worker_internal_ports.append((master_port, scheduler_port))
            if master_port != preferred_master or scheduler_port != preferred_scheduler:
                print(
                    "[no-router] Adjusted internal worker ports due to conflict: "
                    f"worker={i} master={master_port} scheduler={scheduler_port}",
                    flush=True,
                )

        worker_log_files: list[object] = []
        workers: list[subprocess.Popen] = []
        try:
            for i, (url, port, gpu_group) in enumerate(zip(worker_urls, worker_ports, worker_gpu_groups, strict=True)):
                master_port, scheduler_port = worker_internal_ports[i]
                worker_env = dict(env)
                worker_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
                print(
                    f"[no-router] Launching worker {i} at {url} "
                    f"(master={master_port}, scheduler={scheduler_port}) "
                    f"with CUDA_VISIBLE_DEVICES={worker_env['CUDA_VISIBLE_DEVICES']}",
                    flush=True,
                )
                worker_cmd = [
                    *sglang_cli_cmd,
                    "serve",
                    "--model-path",
                    args.model,
                    "--num-gpus",
                    str(args.num_gpus_per_worker),
                    "--host",
                    args.worker_host,
                    "--port",
                    str(port),
                    "--master-port",
                    str(master_port),
                    "--scheduler-port",
                    str(scheduler_port),
                ]
                if args.worker_extra_args:
                    worker_cmd += shlex.split(args.worker_extra_args)

                log_path = output_dir / f"worker_g{total_gpus}_i{i}_p{port}.log"
                log_f = open(log_path, "w", encoding="utf-8")
                worker_log_files.append(log_f)
                workers.append(
                    subprocess.Popen(
                        worker_cmd,
                        env=worker_env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )

            print(f"[no-router] Waiting for {worker_count} worker(s) to become healthy...", flush=True)
            for i, url in enumerate(worker_urls):
                _wait_for_health(url, args.wait_timeout, f"worker {url}", proc=workers[i])

            for num_prompts, max_concurrency in parsed_configs:
                print(
                    f"[no-router] Running config: total_gpus={total_gpus}, "
                    f"num_prompts={num_prompts}, max_concurrency={max_concurrency}",
                    flush=True,
                )

                prompt_splits, conc_splits = _build_shards(worker_count, num_prompts, max_concurrency)

                active_shards = [
                    (i, url, prompts_i, conc_i)
                    for i, (url, prompts_i, conc_i) in enumerate(zip(worker_urls, prompt_splits, conc_splits, strict=True))
                    if prompts_i > 0 and conc_i > 0
                ]
                if not active_shards:
                    raise RuntimeError(
                        "No active shards were created. "
                        f"worker_count={worker_count}, num_prompts={num_prompts}, max_concurrency={max_concurrency}"
                    )

                # If request-rate is finite, split it across shard clients so the total offered
                # rate stays comparable to the router case (single client).
                # Allocate proportionally to each shard's max_concurrency share.
                if args.request_rate != float("inf"):
                    total_conc = sum(conc_i for _, _, _, conc_i in active_shards)
                    if total_conc <= 0:
                        per_shard_request_rates = {i: float("inf") for i, *_ in active_shards}
                    else:
                        per_shard_request_rates = {
                            i: float(args.request_rate) * float(conc_i) / float(total_conc)
                            for i, _, _, conc_i in active_shards
                        }
                else:
                    per_shard_request_rates = {i: float("inf") for i, *_ in active_shards}

                bench_jobs: list[dict[str, object]] = []
                for i, url, prompts_i, conc_i in active_shards:
                    out_file = output_dir / f"bench_g{total_gpus}_p{num_prompts}_c{max_concurrency}_shard{i}.json"
                    log_file = output_dir / f"bench_g{total_gpus}_p{num_prompts}_c{max_concurrency}_shard{i}.log"
                    shard_request_rate = per_shard_request_rates.get(i, float(args.request_rate))
                    cmd = [
                        sys.executable,
                        "-m",
                        "sglang.multimodal_gen.benchmarks.bench_serving",
                        "--base-url",
                        url,
                        "--model",
                        args.model,
                        "--dataset",
                        args.dataset,
                        "--num-prompts",
                        str(prompts_i),
                        "--max-concurrency",
                        str(conc_i),
                        "--request-rate",
                        str(shard_request_rate),
                        "--log-level",
                        args.log_level,
                        "--output-file",
                        str(out_file),
                    ]
                    if args.dataset_path:
                        cmd += ["--dataset-path", args.dataset_path]
                    if args.task:
                        cmd += ["--task", args.task]
                    if args.width:
                        cmd += ["--width", str(args.width)]
                    if args.height:
                        cmd += ["--height", str(args.height)]
                    if args.num_frames:
                        cmd += ["--num-frames", str(args.num_frames)]
                    if args.fps:
                        cmd += ["--fps", str(args.fps)]
                    if args.disable_tqdm:
                        cmd.append("--disable-tqdm")

                    log_f = open(log_file, "w", encoding="utf-8")
                    worker_log_files.append(log_f)
                    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
                    bench_jobs.append(
                        {
                            "proc": proc,
                            "out_file": out_file,
                            "log_file": log_file,
                            "worker_idx": i,
                            "prompts": prompts_i,
                            "concurrency": conc_i,
                            "start_time": time.time(),
                        }
                    )

                total_shards = len(bench_jobs)
                pending = list(bench_jobs)
                wait_start = time.time()
                last_status_print = 0.0
                while pending:
                    now = time.time()
                    finished: list[dict[str, object]] = []

                    for job in pending:
                        proc = job["proc"]
                        assert isinstance(proc, subprocess.Popen)
                        rc = proc.poll()
                        if rc is None:
                            if args.bench_shard_timeout > 0:
                                start_time = float(job["start_time"])
                                if now - start_time > args.bench_shard_timeout:
                                    _signal_group(proc, signal.SIGTERM)
                                    raise TimeoutError(
                                        "A bench_serving shard exceeded --bench-shard-timeout "
                                        f"({args.bench_shard_timeout}s). "
                                        f"shard={job['worker_idx']} log={job['log_file']}"
                                    )
                            continue
                        if rc != 0:
                            raise RuntimeError(
                                f"A bench_serving shard exited with code {rc}. "
                                f"shard={job['worker_idx']} log={job['log_file']}"
                            )
                        finished.append(job)

                    if finished:
                        for job in finished:
                            pending.remove(job)
                        print(
                            f"[no-router] Completed shard(s): +{len(finished)} "
                            f"(done {total_shards - len(pending)}/{total_shards})",
                            flush=True,
                        )

                    if pending and now - last_status_print >= args.bench_status_interval:
                        progress_done = 0
                        progress_total = 0
                        parsed_count = 0
                        for job in pending:
                            log_path = job["log_file"]
                            assert isinstance(log_path, Path)
                            parsed = _read_log_progress(log_path)
                            if parsed:
                                done, total = parsed
                                progress_done += done
                                progress_total += total
                                parsed_count += 1

                        elapsed = now - wait_start
                        if progress_total > 0:
                            pct = 100.0 * progress_done / progress_total
                            print(
                                "[no-router] Bench shards still running: "
                                f"{len(pending)}/{total_shards} pending, "
                                f"progress~{progress_done}/{progress_total} ({pct:.1f}%), "
                                f"elapsed={elapsed:.0f}s",
                                flush=True,
                            )
                        else:
                            print(
                                "[no-router] Bench shards still running: "
                                f"{len(pending)}/{total_shards} pending, "
                                f"elapsed={elapsed:.0f}s (progress unavailable for {parsed_count}/{len(pending)} shard logs)",
                                flush=True,
                            )
                        last_status_print = now

                    if pending:
                        time.sleep(1)

                shard_metrics: list[dict] = []
                for job in bench_jobs:
                    out_file = job["out_file"]
                    assert isinstance(out_file, Path)
                    with out_file.open(encoding="utf-8") as f:
                        shard_metrics.append(json.load(f))

                completed = sum(int(m.get("completed_requests", 0)) for m in shard_metrics)
                failed = sum(int(m.get("failed_requests", 0)) for m in shard_metrics)
                durations = [float(m.get("duration", 0.0)) for m in shard_metrics]
                duration_max_s = max(durations) if durations else 0.0
                # NOTE: shards run in parallel; global throughput should be computed against wall-clock time.
                throughput_qps = completed / duration_max_s if duration_max_s > 0 else 0.0
                # Keep the sum for debugging, but don't use it as "global throughput".
                throughput_sum = sum(float(m.get("throughput_qps", 0.0)) for m in shard_metrics)
                peak_max = max((float(m.get("peak_memory_mb_max", 0.0)) for m in shard_metrics), default=0.0)
                peak_mean = mean([float(m.get("peak_memory_mb_mean", 0.0)) for m in shard_metrics]) if shard_metrics else 0.0
                peak_medians = [float(m.get("peak_memory_mb_median", 0.0)) for m in shard_metrics]
                peak_median = median(peak_medians) if peak_medians else 0.0
                latency_mean = _weighted_latency_mean(shard_metrics)

                aggregate = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "total_gpus": total_gpus,
                    "workers": worker_count,
                    "num_gpus_per_worker": args.num_gpus_per_worker,
                    "num_prompts": num_prompts,
                    "max_concurrency": max_concurrency,
                    "completed_requests": completed,
                    "failed_requests": failed,
                    "duration_s": duration_max_s,
                    "throughput_qps_sum": throughput_sum,
                    "duration_max_s": duration_max_s,
                    "throughput_qps": throughput_qps,
                    "latency_mean_weighted_s": latency_mean,
                    # Align key name with bench_serving outputs for easier comparison.
                    "latency_mean": latency_mean,
                    "peak_memory_mb_max": peak_max,
                    "peak_memory_mb_mean": peak_mean,
                    "peak_memory_mb_median": peak_median,
                    "worker_urls": worker_urls,
                    "active_shards": len(active_shards),
                    "request_rate_total": float(args.request_rate),
                }

                agg_file = output_dir / f"bench_g{total_gpus}_p{num_prompts}_c{max_concurrency}_aggregate.json"
                with open(agg_file, "w", encoding="utf-8") as f:
                    json.dump(aggregate, f, indent=2)
                print(f"[no-router] Wrote aggregate metrics: {agg_file}", flush=True)
        finally:
            _terminate_all(workers)
            for log_f in worker_log_files:
                try:
                    log_f.close()
                except Exception:
                    pass

    print("[no-router] Completed all requested runs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
