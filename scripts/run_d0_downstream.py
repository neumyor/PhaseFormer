#!/usr/bin/env python3
"""Autonomous D0 (h192 x seed2021) downstream supervisor.

Once the stage-gated Track-R launcher completes the D0 scope (the full-anchor
and non-full validation-only checkpoints are present, complete and leak-free
for the requested --datasets/--horizons/--seeds scope), this supervisor
automatically runs the D0 downstream pipeline of the preregistered plan (v1.1,
stage 3a-F):

    D0 validation audit  ->  D0 Track F (the none/full anchors, x10 inputs)
                        ->  D0 retrained test (the non-full checkpoints)
                        ->  D0 summary (single-seed provisional, section 13.0)

It reuses the scope-filtered matrix runners
(run_input_component_frozen_matrix.py / _retrained_test_matrix.py /
summarize_input_component_ablation.py) for planning, gating and command
construction: the runner is executed WITHOUT --execute to emit the validated,
shlex-printed per-job command lines, which this supervisor then dispatches with
bounded concurrency across the given GPUs (pinning CUDA_VISIBLE_DEVICES per job,
resuming by the runner's own metrics-file semantics).  The D0 Track F phase
fully finishes before D0 retrained test begins; the summary phase is last.

GPU serialization (v1.2): the Track-R scheduler never yields a GPU once it has
activated it, so this downstream supervisor must NOT share a GPU with the
Track-R training launcher.  The intended flow is therefore sequential: the
training launcher runs --max-stage 1 (D0 only) and exits; this supervisor then
runs the whole D0 downstream on the freed GPUs; and once the D0 summary is
written it spawns the D1 Track-R launcher (--resume-supervisor-argv) before
marking itself done.  The supervisor is resumable at any point (state +
per-phase metrics files) and never reads or writes test metrics of
out-of-scope settings.

State machine (persisted to <control>/d0_state.json):
    wait_3a -> track_f -> retrained -> summarize -> done
On a terminal failure it lands in a *_failed state and stops (needs a human).

Usage:
    <python> scripts/run_d0_downstream.py \
        --track-r-dir .../input_components_h134_scratch \
        --control-dir .../input_components_h134_control \
        --datasets ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Exchange,Weather \
        --horizons 192 --seeds 2021 \
        --gpus 2,3 --jobs-per-gpu 4 --min-free-mb 5000 --probe-sec 60 \
        --resume-supervisor-argv .../control/trackr_d1_argv.json
"""

from __future__ import annotations

import argparse
import csv
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

from scripts.run_input_component_ablation import DATASETS, expected_full_anchors
from scripts.run_input_components_parallel import atomic_write_json, free_memory_mb
from scripts.run_input_component_frozen_matrix import discover as frozen_discover
from scripts.run_input_component_retrained_test_matrix import (
    discover as retrained_discover,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _isnum(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _label(argv):
    """Compact human-readable identity for a dispatched job argv."""
    get = {}
    for i, token in enumerate(argv[:-1]):
        if token.startswith("--") and not token.startswith("---"):
            get[token[2:]] = argv[i + 1]
    parts = [get.get(k) for k in ("dataset", "horizon", "seed", "model")]
    variant = get.get("input_variant") or get.get("input_hypothesis")
    core = "/".join(str(x) for x in parts if x is not None)
    return f"{core} {variant or ''}".strip()


# --------------------------------------------------------------------------- #
# plan generation via the scope-filtered matrix runners (print-only)
# --------------------------------------------------------------------------- #

def run_runner_plan(py, runner, track_r_dir, output_dir, horizons, seeds,
                    datasets):
    """Run a matrix runner without --execute; return (rc, [argv...]).

    rc == 0 means the runner's formal gates passed (scope complete & clean) and
    stdout contains one shlex-joined command per job to execute.  rc != 0 means
    the scope is not yet ready (or invalid); callers treat that as "keep
    waiting".
    """
    command = [
        py,
        str(REPO_ROOT / "scripts" / runner),
        "--track-r-dir", str(track_r_dir),
        "--output-dir", str(output_dir),
        "--datasets", ",".join(datasets),
        "--horizons", ",".join(map(str, horizons)),
        "--seeds", ",".join(map(str, seeds)),
    ]
    probe = subprocess.run(command, cwd=str(REPO_ROOT), text=True,
                           capture_output=True, timeout=600)
    if probe.returncode != 0:
        return probe.returncode, []
    argv_list = []
    for line in probe.stdout.splitlines():
        if not line.lstrip().startswith(py):
            continue  # informational lines only
        tokens = shlex.split(line)
        if any(t.endswith(".py") and "evaluate_input_component" in t
               for t in tokens):
            argv_list.append(tokens)
    return 0, argv_list


# --------------------------------------------------------------------------- #
# D0 validation audit (read-only)
# --------------------------------------------------------------------------- #

def run_d0_audit(track_r_dir, horizons, seeds, datasets):
    """Read-only Stage-3a audit over the D0 scope.

    Reuses the two matrix runners' discover() parsers as the authoritative
    coverage / no-leak / percent / duplicate gate: each raises ValueError when the
    scoped rows are incomplete, already carry test metrics, violate percent=100 /
    max_eval_samples=0, or contain duplicate checkpoints.  Adds an anchor
    uniqueness check (each (dataset,horizon,seed,mechanism) setting must map to a
    single recorded checkpoint path).  Returns (passed: bool, summary: dict);
    forms no effect conclusions.
    """
    issues = {}
    try:
        frozen = frozen_discover(track_r_dir, smoke=False,
                                 horizons=horizons, seeds=seeds,
                                 datasets=datasets)
        retrain = retrained_discover(track_r_dir, smoke=False,
                                     horizons=horizons, seeds=seeds,
                                     datasets=datasets)
        # The retrained-test read covers only the 9 non-full intervention
        # checkpoints per setting (none/full is evaluated once by Track F and
        # reused as the common baseline), so mirror that subset before counting:
        # discover() returns the full condition set as its completeness gate.
        retrain = retrain[
            ~((retrain.input_hypothesis == "none") & (retrain.input_variant == "full"))
        ].copy()
        issues["full_anchors"] = int(len(frozen))
        issues["retrained_checkpoints"] = int(len(retrain))
        issues["expected_full_anchors"] = expected_full_anchors(
            horizons, seeds, datasets=datasets)
        issues["expected_retrained"] = issues["expected_full_anchors"] * 9
        if (issues["full_anchors"] != issues["expected_full_anchors"]
                or issues["retrained_checkpoints"] != issues["expected_retrained"]):
            issues["pass"] = False
            return False, issues
        setting_keys = ["dataset", "horizon", "seed", "mechanism"]
        per_setting = frozen[setting_keys].value_counts()
        issues["anchors_per_setting_min"] = int(per_setting.min())
        issues["anchors_per_setting_max"] = int(per_setting.max())
        issues["anchor_path_dups"] = int(
            frozen.groupby(setting_keys)["checkpoint"].nunique().ne(1).sum()
        )
        issues["pass"] = (
            issues["anchors_per_setting_min"] == 1
            and issues["anchors_per_setting_max"] == 1
            and issues["anchor_path_dups"] == 0
        )
        return issues["pass"], issues
    except ValueError as exc:
        issues["error"] = str(exc)[-2000:]
        issues["pass"] = False
        return False, issues


# --------------------------------------------------------------------------- #
# dispatch core (mirrors the Track-R launcher's slot scheduler)
# --------------------------------------------------------------------------- #

def _job_metrics(argv):
    """Return the metrics file a printed runner command writes, if derivable."""
    if not argv:
        return None
    out_dir = None
    for i, token in enumerate(argv):
        if token == "--output-dir" and i + 1 < len(argv):
            out_dir = Path(argv[i + 1])
            break
    if out_dir is None:
        return None
    is_frozen = any(t.endswith("evaluate_input_component_checkpoint.py")
                    for t in argv)
    name = "frozen_metrics.csv" if is_frozen else "retrained_metrics.csv"
    return out_dir / name


class Dispatcher:
    def __init__(self, gpus, jobs_per_gpu, min_free_mb, poll_sec, max_attempts):
        self.gpus = sorted(gpus)
        self.jobs_per_gpu = jobs_per_gpu
        self.min_free_mb = min_free_mb
        self.poll_sec = poll_sec
        self.max_attempts = max_attempts
        self.slots = {}   # sid -> {"gpu": int, "proc": Popen|None, "argv": list|None}
        self.active = set()
        self.attempts = {}  # id(argv) -> attempt count
        self.pending = []

    def _free_slots(self):
        return [sid for sid, s in self.slots.items() if s["proc"] is None]

    def refresh(self):
        free = free_memory_mb()
        for gpu in self.gpus:
            if gpu in self.active:
                continue
            mem = free.get(gpu)
            if mem is not None and mem < self.min_free_mb:
                continue
            for _ in range(self.jobs_per_gpu):
                sid = len(self.slots)
                self.slots[sid] = {"gpu": gpu, "proc": None, "argv": None}
            self.active.add(gpu)
            print(f"[d0] gpu {gpu} active with {self.jobs_per_gpu} slots",
                  flush=True)

    def dispatch_loop(self, argv_list, control_dir, phase_tag):
        """Run all argv jobs to completion; resume via metrics-file existence."""
        to_run = []
        for argv in argv_list:
            metrics = _job_metrics(argv)
            if metrics is not None and metrics.is_file():
                continue
            to_run.append(argv)
        self.pending = list(to_run)
        self.attempts = {}
        print(f"[d0] {phase_tag}: {len(argv_list)} planned, "
              f"{len(self.pending)} to run", flush=True)
        (control_dir / "d0_jobs").mkdir(exist_ok=True)
        terminal = False
        while self.pending or any(s["proc"] is not None
                                  for s in self.slots.values()):
            # reap finished jobs
            for sid in list(self.slots):
                s = self.slots[sid]
                if s["proc"] is None:
                    continue
                rc = s["proc"].poll()
                if rc is None:
                    continue
                argv = s["argv"]
                gpu = s["gpu"]
                self.slots[sid] = {"gpu": gpu, "proc": None, "argv": None}
                if rc == 0:
                    print(f"[d0] DONE gpu{gpu}/slot{sid} {_label(argv)}",
                          flush=True)
                else:
                    aid = id(argv)
                    self.attempts[aid] = self.attempts.get(aid, 0) + 1
                    if self.attempts[aid] < self.max_attempts:
                        print(f"[d0] RETRY gpu{gpu} rc={rc} {_label(argv)} "
                              f"(attempt {self.attempts[aid]})", flush=True)
                        self.pending.append(argv)
                    else:
                        print(f"[d0] TERMINAL FAILURE rc={rc} {_label(argv)}",
                              flush=True)
                        terminal = True
            if terminal:
                return False
            # top up GPU pool and dispatch
            self.refresh()
            idle = self._free_slots()
            while idle and self.pending:
                argv = self.pending.pop(0)
                metrics = _job_metrics(argv)
                if metrics is not None and metrics.is_file():
                    continue
                sid = idle.pop(0)
                gpu = self.slots[sid]["gpu"]
                logf = open(control_dir / "d0_jobs" /
                            f"{phase_tag}_{sid}.log", "a")
                logf.write(f"\n# {utc_now()} launch gpu{gpu} slot{sid}\n"
                           f"# {shlex.join(argv)}\n")
                logf.flush()
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), env=env,
                                        stdout=logf, stderr=subprocess.STDOUT)
                self.slots[sid] = {"gpu": gpu, "proc": proc, "argv": argv}
                print(f"[d0] LAUNCH gpu{gpu}/slot{sid} {_label(argv)}",
                      flush=True)
            time.sleep(self.poll_sec)
        return True

    def shut_down(self):
        for s in self.slots.values():
            if s["proc"] is not None:
                try:
                    s["proc"].terminate()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# phase drivers
# --------------------------------------------------------------------------- #

FROZEN_RUNNER = "run_input_component_frozen_matrix.py"
RETRAINED_RUNNER = "run_input_component_retrained_test_matrix.py"
SUMMARY_RUNNER = "summarize_input_component_ablation.py"


def summarize_d0(py, frozen_out, retrained_out, summary_out, horizons, seeds,
                 datasets):
    argv = [
        py,
        str(REPO_ROOT / "scripts" / SUMMARY_RUNNER),
        str(frozen_out), str(retrained_out),
        "--datasets", ",".join(datasets),
        "--horizons", ",".join(map(str, horizons)),
        "--seeds", ",".join(map(str, seeds)),
        "--output", str(summary_out),
    ]
    res = subprocess.run(argv, cwd=str(REPO_ROOT), text=True,
                         capture_output=True, timeout=3600)
    if res.stdout:
        print(res.stdout[-2000:], flush=True)
    if res.returncode != 0:
        print(res.stderr[-4000:], flush=True)
        return False
    return True


def relaunch_supervisor(control_dir, resume_argv_path):
    """Spawn the (D1) Track-R supervisor after the D0 summary, detached.

    resume_argv_path points to a JSON file holding the exact supervisor argv to
    run (fresh session, cwd = repo root).  Returns (pid, None) or (None, err).
    """
    try:
        argv = json.loads(Path(resume_argv_path).expanduser().read_text())
    except Exception as exc:
        return None, f"cannot read {resume_argv_path}: {exc!r}"
    if not isinstance(argv, list) or not argv:
        return None, f"{resume_argv_path} does not contain an argv list"
    logf = open(control_dir / "supervisor_resume.log", "a")
    logf.write(f"\n# {utc_now()} relaunch D1 Track-R supervisor\n"
               f"# {shlex.join(argv)}\n")
    logf.flush()
    env = dict(os.environ)
    proc = subprocess.Popen(argv, cwd=str(REPO_ROOT), env=env,
                            stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    (control_dir / "supervisor_resume.pid").write_text(str(proc.pid) + "\n")
    return proc.pid, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track-r-dir", type=Path, required=True)
    p.add_argument("--control-dir", type=Path, required=True)
    p.add_argument("--frozen-out", type=Path)
    p.add_argument("--retrained-out", type=Path)
    p.add_argument("--summary-out", type=Path)
    p.add_argument("--datasets", default=None,
                   help="comma list of datasets (default: all preregistered)")
    p.add_argument("--horizons", default="192")
    p.add_argument("--seeds", default="2021")
    p.add_argument("--resume-supervisor-argv", type=Path,
                   help="JSON file with the D1 Track-R supervisor argv to spawn "
                        "after the D0 summary finishes")
    p.add_argument("--gpus", default="2,3")
    p.add_argument("--jobs-per-gpu", type=int, default=2)
    p.add_argument("--min-free-mb", type=int, default=9000)
    p.add_argument("--poll-sec", type=float, default=30.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--probe-sec", type=float, default=300.0,
                   help="wait between readiness re-probes while in wait_3a")
    args = p.parse_args()

    track_r_dir = Path(args.track_r_dir).expanduser().resolve()
    control_dir = Path(args.control_dir).expanduser().resolve()
    frozen_out = (Path(args.frozen_out).expanduser().resolve()
                  if args.frozen_out
                  else track_r_dir.parent / "input_components_h134_frozen_d0")
    retrained_out = (Path(args.retrained_out).expanduser().resolve()
                     if args.retrained_out
                     else track_r_dir.parent
                     / "input_components_h134_retrained_test_d0")
    summary_out = (Path(args.summary_out).expanduser().resolve()
                   if args.summary_out
                   else track_r_dir.parent / "result_summary_d0.csv")

    if args.datasets:
        datasets = tuple(sorted(
            d.strip() for d in args.datasets.split(",") if d.strip()))
        unknown = set(datasets) - set(DATASETS)
        if unknown:
            p.error(f"unknown datasets: {sorted(unknown)}")
    else:
        datasets = DATASETS
    horizons = tuple(sorted(int(h) for h in args.horizons.split(",") if h.strip()))
    seeds = tuple(sorted(int(s) for s in args.seeds.split(",") if s.strip()))
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / "d0_jobs").mkdir(exist_ok=True)
    py = sys.executable
    logf = open(control_dir / "d0_downstream.log", "a")
    logf.write(f"\n# {utc_now()} d0 downstream start datasets={datasets} "
               f"horizons={horizons} seeds={seeds} gpus={args.gpus}\n")
    logf.flush()

    state_path = control_dir / "d0_state.json"
    state = load_json(state_path, {"stage": "wait_3a"})
    dispatcher = Dispatcher(
        [int(g) for g in args.gpus.split(",") if g.strip()],
        args.jobs_per_gpu, args.min_free_mb, args.poll_sec, args.max_attempts,
    )

    def save(next_stage, **extra):
        nonlocal state
        state = {"stage": next_stage, "last_update": utc_now(), **extra}
        atomic_write_json(state_path, state)
        print(f"[d0] state -> {next_stage}", flush=True)

    print(f"[d0] starting from stage={state.get('stage')}", flush=True)

    try:
        while True:
            stage = state.get("stage")
            if stage in ("done", "audit_failed", "track_f_failed",
                         "retrained_failed", "summarize_failed"):
                print(f"[d0] terminal state {stage}; exiting", flush=True)
                logf.flush()
                break

            if stage == "wait_3a":
                # Readiness = both scoped matrix runners pass their formal gates
                # (D0 Track R complete and leak-free).
                rc_f, _ = run_runner_plan(py, FROZEN_RUNNER, track_r_dir,
                                          frozen_out, horizons, seeds,
                                          datasets)
                rc_r, _ = run_runner_plan(py, RETRAINED_RUNNER, track_r_dir,
                                          retrained_out, horizons, seeds,
                                          datasets)
                if rc_f == 0 and rc_r == 0:
                    passed, audit = run_d0_audit(track_r_dir, horizons, seeds,
                                                 datasets)
                    print(f"[d0] D0 Track R complete; audit passed={passed} "
                          f"{json.dumps(audit, default=str)}", flush=True)
                    if passed:
                        save("track_f", audit=audit)
                    else:
                        save("audit_failed", audit=audit,
                             reason="D0 validation audit failed")
                else:
                    print(f"[d0] wait_3a: not ready (frozen rc={rc_f}, "
                          f"retrained rc={rc_r}); sleep {args.probe_sec}s",
                          flush=True)
                    time.sleep(args.probe_sec)
                continue

            if stage == "track_f":
                rc, argv_list = run_runner_plan(py, FROZEN_RUNNER, track_r_dir,
                                                frozen_out, horizons, seeds,
                                                datasets)
                if rc != 0:
                    save("wait_3a")  # scope no longer complete; recheck later
                    time.sleep(args.probe_sec)
                    continue
                ok = dispatcher.dispatch_loop(argv_list, control_dir, "trackf")
                if not ok:
                    save("track_f_failed", reason="Track F terminal failure")
                    break
                # Verification pass: formal runner --execute --resume.  With all
                # metrics present it prints RESUME lines and executes nothing.
                verify = subprocess.run(
                    [py, str(REPO_ROOT / "scripts" / FROZEN_RUNNER),
                     "--track-r-dir", str(track_r_dir),
                     "--output-dir", str(frozen_out),
                     "--datasets", ",".join(datasets),
                     "--horizons", ",".join(map(str, horizons)),
                     "--seeds", ",".join(map(str, seeds)),
                     "--execute", "--resume"],
                    cwd=str(REPO_ROOT), text=True, capture_output=True,
                    timeout=1800)
                if verify.returncode != 0:
                    save("track_f_failed", reason=verify.stderr[-2000:])
                    break
                save("retrained")
                continue

            if stage == "retrained":
                rc, argv_list = run_runner_plan(py, RETRAINED_RUNNER,
                                                track_r_dir, retrained_out,
                                                horizons, seeds, datasets)
                if rc != 0:
                    save("wait_3a")
                    time.sleep(args.probe_sec)
                    continue
                ok = dispatcher.dispatch_loop(argv_list, control_dir, "retrain")
                if not ok:
                    save("retrained_failed", reason="retrained terminal failure")
                    break
                verify = subprocess.run(
                    [py, str(REPO_ROOT / "scripts" / RETRAINED_RUNNER),
                     "--track-r-dir", str(track_r_dir),
                     "--output-dir", str(retrained_out),
                     "--datasets", ",".join(datasets),
                     "--horizons", ",".join(map(str, horizons)),
                     "--seeds", ",".join(map(str, seeds)),
                     "--execute", "--resume"],
                    cwd=str(REPO_ROOT), text=True, capture_output=True,
                    timeout=1800)
                if verify.returncode != 0:
                    save("retrained_failed", reason=verify.stderr[-2000:])
                    break
                save("summarize")
                continue

            if stage == "summarize":
                summary_out.parent.mkdir(parents=True, exist_ok=True)
                ok = summarize_d0(py, frozen_out, retrained_out, summary_out,
                                  horizons, seeds, datasets)
                if not ok:
                    save("summarize_failed", reason="D0 summary failed")
                    break
                resumed_pid = None
                if args.resume_supervisor_argv is not None:
                    resumed_pid, err = relaunch_supervisor(
                        control_dir, args.resume_supervisor_argv)
                    if err:
                        save("summarize_failed",
                             reason=f"supervisor resume failed: {err}")
                        break
                save("done", summary=str(summary_out),
                     resumed_supervisor_pid=resumed_pid)
                continue

            time.sleep(args.poll_sec)
    finally:
        logf.flush()
        logf.close()


if __name__ == "__main__":
    main()
