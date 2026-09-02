#!/usr/bin/env python3
"""Parallel, resumable, stage-gated Track-R input-component matrix launcher.

The single-process driver `run_input_component_ablation.py` executes jobs one at
a time.  This launcher runs the same preregistered Track-R search commands but
keeps several training subprocesses running across the available GPUs
(`--jobs-per-gpu` jobs may share one GPU, since a PhaseFormer run uses only a
fraction of a GPU), and enforces the preregistered execution order in hard
stages:

    Stage 1 (priority pass): horizon=192, seed=2021   -> 8 x 3 x 10 = 240 runs
    Stage 2:                 horizon in {96,336,720}, seed=2021
    Stage 3:                 seeds {2022,2023}, all four horizons

A stage never starts until the previous stage has fully completed.  A stage must
complete with zero terminal failures before the next begins; if any job fails
after `--max-attempts` retries the launcher records it and aborts (the matrix
must not advance incomplete).  All jobs are validation-only Track-R runs (no
test loader is constructed).  Completed runs are tracked in `control/done.tsv`;
a job whose run dir already contains `metrics.csv` is skipped by the child's own
`--resume`, so restarts are idempotent.

Only GPUs whose free memory is at least `--min-free-mb` are used; the pool is
re-read every `--refresh-sec` seconds (and --gpus-file is re-read too), so a GPU
that becomes free later is picked up without restarting the launcher.

Usage (formal):
    <python> scripts/run_input_components_parallel.py \
        --output-dir /home/niuyiming/PhaseFormer/research_runs/input_components_h134_scratch \
        --control-dir /home/niuyiming/PhaseFormer/research_runs/input_components_h134_control \
        --gpus 2,3 --jobs-per-gpu 4 --max-stage 3
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input_component_ablation import (
    CONDITIONS,
    DATASETS,
    MODELS,
    PRIORITY_HORIZON,
    PRIORITY_SEED,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stage_of(horizon: int, seed: int) -> int:
    if horizon == PRIORITY_HORIZON and seed == PRIORITY_SEED:
        return 1
    if seed == PRIORITY_SEED:
        return 2
    return 3


def job_key(stage, job):
    return (
        f"s{stage}_{job['dataset']}_{job['model']}_h{job['horizon']}_"
        f"seed{job['seed']}_{job['hypothesis']}-{job['variant']}"
    )


def build_queue(args):
    """Return list of (stage, job) in strict execution order."""
    jobs = []
    for dataset in args.datasets:
        for model in args.models:
            for horizon in sorted(args.horizons):
                for seed in sorted(args.seeds):
                    st = stage_of(horizon, seed)
                    if st > args.max_stage:
                        continue
                    for hypothesis, variant in CONDITIONS:
                        jobs.append(
                            (
                                st,
                                {
                                    "dataset": dataset,
                                    "model": model,
                                    "horizon": horizon,
                                    "seed": seed,
                                    "hypothesis": hypothesis,
                                    "variant": variant,
                                },
                            )
                        )
    jobs.sort(key=lambda item: item[0])
    return jobs


def build_command(args, job):
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "search_phaseformer.py"),
        "--dataset", job["dataset"],
        "--horizon", str(job["horizon"]),
        "--stage", "input_components",
        "--mechanism", job["model"],
        "--input-hypothesis", job["hypothesis"],
        "--input-variant", job["variant"],
        "--seed", str(job["seed"]),
        "--max-epochs", str(args.max_epochs),
        "--percent", str(args.percent),
        "--num-workers", str(args.num_workers),
        "--output-dir", str(args.output_dir),
        "--resume",
        "--require-cuda",
    ]


def free_memory_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=20,
        )
    except Exception:
        return {}
    result = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                result[int(parts[0])] = int(parts[1])
            except ValueError:
                continue
    return result


def load_lines(path):
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def atomic_append(path, line):
    with open(path, "a", newline="") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def atomic_write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def read_gpus_file(path):
    if path is None or not path.exists():
        return set()
    return {int(ln.strip()) for ln in path.read_text().splitlines() if ln.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--horizons", default="96,192,336,720")
    p.add_argument("--seeds", default="2021,2022,2023")
    p.add_argument("--gpus", default="2,3",
                   help="candidate physical GPU indices (comma list)")
    p.add_argument("--gpus-file", type=Path,
                   help="optional file listing candidate GPU indices (re-read); union of both")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="number of concurrent training jobs to place on one GPU")
    p.add_argument("--max-stage", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--min-free-mb", type=int, default=5000)
    p.add_argument("--refresh-sec", type=float, default=20.0)
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--percent", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "research_runs" / "input_components_h134_scratch")
    p.add_argument("--control-dir", type=Path,
                   default=REPO_ROOT / "research_runs" / "input_components_h134_control")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    def csvset(values, cast=str):
        return {cast(v.strip()) for v in values.split(",") if v.strip()}

    args.datasets = tuple(sorted(csvset(args.datasets)))
    args.models = tuple(sorted(csvset(args.models)))
    args.horizons = tuple(sorted(int(h) for h in csvset(args.horizons, int)))
    args.seeds = tuple(sorted(int(s) for s in csvset(args.seeds, int)))
    for d in set(args.datasets) - set(DATASETS):
        p.error(f"unknown dataset {d}")
    for m in set(args.models) - set(MODELS):
        p.error(f"unknown model {m}")

    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.control_dir = Path(args.control_dir).expanduser().resolve()
    args.control_dir.mkdir(parents=True, exist_ok=True)
    (args.control_dir / "jobs").mkdir(exist_ok=True)

    queue = build_queue(args)
    total = len(queue)
    counts = collections.Counter(stage for stage, _ in queue)
    print(f"[launcher] queue: stage counts {dict(counts)} (total {total})", flush=True)
    if args.dry_run:
        for stage in sorted(counts):
            first = next(j for s, j in queue if s == stage)
            print(f"  stage {stage}: {counts[stage]} jobs | first "
                  f"{first['dataset']}/{first['model']}/h{first['horizon']}/"
                  f"s{first['seed']}/{first['hypothesis']}-{first['variant']}", flush=True)
        print("[launcher] dry-run only; nothing launched", flush=True)
        return

    candidates = set(int(g) for g in args.gpus.split(",") if g.strip())
    done_path = args.control_dir / "done.tsv"
    fail_path = args.control_dir / "failures.tsv"
    done = load_lines(done_path)
    attempts = {}
    status_path = args.control_dir / "supervisor.json"

    # Slots: each active GPU contributes --jobs-per-gpu slots.
    slots = {}  # slot id -> {"gpu": int, "proc": Popen|None, "key": str|None}
    active_gpus = set()

    def running_slots():
        return {sid for sid, s in slots.items() if s["proc"] is not None}

    def dump(message, stage=None):
        info = {
            "message": message,
            "stage": stage,
            "done_total": len(done),
            "running": [slots[sid]["key"] for sid in sorted(slots)
                        if slots[sid]["proc"] is not None],
            "active_gpus": sorted(active_gpus),
            "slots": len(slots),
            "last_update": utc_now(),
        }
        atomic_write_json(status_path, info)
        print(f"[{utc_now()}] {message}", flush=True)

    def refresh_pool():
        free = free_memory_mb()
        now_candidates = candidates | read_gpus_file(args.gpus_file)
        for gpu in sorted(now_candidates):
            if gpu in active_gpus:
                continue
            mem = free.get(gpu)
            usable = mem is None or mem >= args.min_free_mb
            if not usable:
                print(f"[launcher] gpu {gpu} free mem only {mem} MiB (< {args.min_free_mb}); waiting",
                      flush=True)
                continue
            if mem is None:
                print(f"[launcher] nvidia-smi unavailable; assume gpu {gpu} usable", flush=True)
            for _ in range(args.jobs_per_gpu):
                sid = len(slots)
                slots[sid] = {"gpu": gpu, "proc": None, "key": None}
            active_gpus.add(gpu)
            print(f"[launcher] gpu {gpu} active with {args.jobs_per_gpu} slots", flush=True)

    dump("launcher start")
    terminal_failure = None
    try:
        for stage in range(1, args.max_stage + 1):
            stage_jobs = [item for item in queue if item[0] == stage]
            if not stage_jobs:
                continue
            pending = collections.deque(
                (s, j) for s, j in stage_jobs if job_key(s, j) not in done
            )
            print(f"[launcher] stage {stage}: {len(stage_jobs)} jobs, "
                  f"{len(stage_jobs) - len(pending)} already done, {len(pending)} pending",
                  flush=True)
            dump(f"stage {stage}: {len(stage_jobs)} jobs start", stage=stage)

            while pending or running_slots():
                # 1) reap finished jobs
                for sid in list(slots):
                    s = slots[sid]
                    if s["proc"] is None:
                        continue
                    rc = s["proc"].poll()
                    if rc is None:
                        continue
                    key = s["key"]
                    gpu = s["gpu"]
                    s["proc"] = None
                    s["key"] = None
                    if rc == 0:
                        atomic_append(done_path, key)
                        done.add(key)
                        print(f"[launcher] DONE   gpu{gpu}/slot{sid} {key}", flush=True)
                    else:
                        attempts[key] = attempts.get(key, 0) + 1
                        if attempts[key] < args.max_attempts:
                            print(f"[launcher] RETRY  gpu{gpu} {key} "
                                  f"(attempt {attempts[key]}/{args.max_attempts}) rc={rc}",
                                  flush=True)
                            pending.appendleft(
                                (stage, next(j for s2, j in stage_jobs
                                             if job_key(s2, j) == key))
                            )
                        else:
                            atomic_append(fail_path,
                                          f"{key}\t{utc_now()}\trc={rc}\tmax_attempts")
                            terminal_failure = key
                            print(f"[launcher] TERMINAL FAILURE {key} rc={rc}", flush=True)
                # 2) abort on terminal failure
                if terminal_failure:
                    break
                # 3) top up GPU pool and dispatch
                refresh_pool()
                idle = [sid for sid in sorted(slots) if slots[sid]["proc"] is None]
                while idle and pending:
                    stage_i, job = pending.popleft()
                    key = job_key(stage_i, job)
                    if key in done:
                        continue
                    sid = idle.pop(0)
                    gpu = slots[sid]["gpu"]
                    cmd = build_command(args, job)
                    logf = open(args.control_dir / "jobs" / f"{key}.log", "a")
                    logf.write(f"\n# {utc_now()} launch gpu{gpu} slot{sid}\n"
                               f"# {shlex.join(cmd)}\n")
                    logf.flush()
                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                            stdout=logf, stderr=subprocess.STDOUT)
                    slots[sid] = {"gpu": gpu, "proc": proc, "key": key}
                    attempts.setdefault(key, 0)
                    print(f"[launcher] LAUNCH gpu{gpu}/slot{sid} {key}", flush=True)
                # 4) wait a little
                if running_slots():
                    time.sleep(args.poll_sec)
                else:
                    time.sleep(args.refresh_sec)
            if terminal_failure:
                break
            dump(f"stage {stage}: completed ({len(stage_jobs)} runs)", stage=stage)
    except KeyboardInterrupt:
        dump("interrupted; terminating children")
        for sid in list(slots):
            if slots[sid]["proc"] is not None:
                try:
                    slots[sid]["proc"].terminate()
                except Exception:
                    pass
        raise

    if terminal_failure:
        dump(f"ABORT: terminal failure on {terminal_failure}; see failures.tsv")
        sys.exit(3)
    dump(f"ALL STAGES {list(range(1, args.max_stage + 1))} COMPLETE "
         f"({len(done)}/{total} recorded done)")
    print("[launcher] finished", flush=True)


if __name__ == "__main__":
    main()
