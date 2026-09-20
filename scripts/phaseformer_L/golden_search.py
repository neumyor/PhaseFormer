#!/usr/bin/env python3
"""G1: test-set-driven golden-search over the 8 failing §4.2 settings.

Pre-registered plan: docs/PhaseFormer_L_golden_search_plan.md (2026-09-20).
User decisions (2026-09-20, recorded verbatim in the plan §0):

* target = beat Golden on both metrics; phase_only reference is also reported;
* reference baselines = the existing E14 phase_only / l_main runs (their test
  metrics have already been read once by E14 stage B -- disclosed);
* selection criterion = TEST metrics; the reported winner is the best of the
  3 seeds (per-seed best, not 3-seed mean);
* a win reached by shrinking the gate to (near) zero counts, but must be
  labelled "gate-shrunk" in every table (the corrector is then effectively
  off, so the win belongs to the phase backbone, not the level channel);
* searchable axes: weak_period_residual_gate_init, learning_rate, and
  pooled_lowrank rank (plus the shared dense head).

This is explicitly a test-set-selection search: every number it produces is
conditional on the 8 test settings and must never be described as a blind or
unbiased generalization estimate.

Stages
------
plan     build the full grid, verify reuse/resume state, print the budget;
smoke    run 2 cheap cells end-to-end (1 epoch) before the full launch;
search   run the whole stage-1 grid (seed 2021) on the requested GPUs;
confirm  run seeds 2022/2023 for the per-setting stage-1 winners;
select   read test metrics, pick winners, write the results table.

Test metrics are read through search_phaseformer.py --evaluate-test exactly
once per run, after its best-val checkpoint is restored (the runner's own
protocol; same as E14 stage B).  Because selection is BY test, the test read
is not gated behind selection -- that is the point of this experiment and the
reason every result carries the test-set-selection label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "scripts" / "search_phaseformer.py"
PY = "/home/yyk/yyk03/miniconda3/envs/time/bin/python"

# ---------------------------------------------------------------- protocol
# The 8 settings that failed to beat Golden on both metrics in §4.2
# (main-24; Traffic appendix excluded).  Gap columns = l_main vs Golden (%).
SETTINGS = (
    ("ETTh1", 96, 2.664, 4.039),
    ("ETTh1", 192, 3.160, 4.101),
    ("ETTh1", 336, 3.032, 3.341),
    ("ETTm1", 96, 4.398, 2.480),
    ("ETTm1", 192, 4.754, 2.249),
    ("ETTm1", 336, 3.032, 1.544),
    ("ETTm1", 720, 1.143, 0.880),
    ("Electricity", 96, 0.143, 0.774),
)

SEED_STAGE1 = 2021
SEEDS_CONFIRM = (2022, 2023)
LOOKBACK = 720
PERIOD = 24
MAX_EPOCHS = 30
LOSS = "huber"
# Round-2 loss axis.  All are implemented by DefaultModule._get_criterion;
# the runner exposes them so the MSE--MAE trade-off can be searched without
# touching model code.  MAE loss is the natural lever for the settings whose
# MAE gap is the larger one (ETTh1-96/192, ETTm1-96/192).
LOSSES = ("huber", "mae")
# Round-2 lr grid (plan §2b): round 1 used {1e-4, 3e-4, 1e-3} and its first 48
# cells showed 1e-4 is the worst rung on every head, so the sweet spot may sit
# at or above 1e-3.  Round 2 therefore moves the grid up; the two grids share
# 3e-4 and 1e-3, and the ids collide harmlessly because the runs are identical.
LRS_ROUND2 = (3e-4, 1e-3, 3e-3)
PERCENT = 100

# Search grid (frozen in the plan §2).
GATE_INITS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
LRS = (1e-4, 3e-4, 1e-3)
# head spec: "shared" dense head, or pooled_lowrank with rank = H // div.
HEAD_DIVS = (None, 32, 16, 8, 4)

# Q5: reference baselines are the EXISTING E14 runs (their test metrics were
# already read once by E14 stage B, disclosed).  Three-seed means, frozen here
# so the final table can report "vs Golden" and "vs phase_only/l_main" (Q1)
# without re-reading anything.  Source: research_runs/phaseformer_L_e14_main_v1/
# main_table.csv.
E14_REFERENCE = {
    ("ETTh1", 96): dict(phase_only=(0.361402, 0.386687), l_main=(0.368563, 0.397429)),
    ("ETTh1", 192): dict(phase_only=(0.404664, 0.410919), l_main=(0.409547, 0.420570)),
    ("ETTh1", 336): dict(phase_only=(0.441922, 0.434654), l_main=(0.437887, 0.438168)),
    ("ETTm1", 96): dict(phase_only=(0.302441, 0.351168), l_main=(0.305886, 0.352530)),
    ("ETTm1", 192): dict(phase_only=(0.330420, 0.363285), l_main=(0.338355, 0.369119)),
    ("ETTm1", 336): dict(phase_only=(0.359302, 0.381157), l_main=(0.368855, 0.386882)),
    ("ETTm1", 720): dict(phase_only=(0.415068, 0.412775), l_main=(0.416710, 0.413607)),
    ("Electricity", 96): dict(phase_only=(0.130440, 0.222771), l_main=(0.129184, 0.222710)),
}

GOLDEN = {  # from docs/PhaseFormer_gold_standard.md
    ("ETTh1", 96): (0.359, 0.382), ("ETTh1", 192): (0.397, 0.404),
    ("ETTh1", 336): (0.425, 0.424), ("ETTh1", 720): (0.431, 0.450),
    ("ETTh1", 336): (0.425, 0.424),
    ("ETTm1", 96): (0.293, 0.344), ("ETTm1", 192): (0.323, 0.361),
    ("ETTm1", 336): (0.358, 0.381), ("ETTm1", 720): (0.412, 0.410),
    ("Electricity", 96): (0.129, 0.221),
    ("ETTh2", 96): (0.275, 0.338), ("ETTh2", 192): (0.341, 0.376),
    ("ETTh2", 336): (0.369, 0.405), ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256), ("ETTm2", 192): (0.219, 0.293),
    ("ETTm2", 336): (0.269, 0.326), ("ETTm2", 720): (0.351, 0.379),
    ("Weather", 96): (0.148, 0.195), ("Weather", 192): (0.193, 0.237),
    ("Weather", 336): (0.242, 0.278), ("Weather", 720): (0.309, 0.332),
    ("Electricity", 192): (0.148, 0.238), ("Electricity", 336): (0.165, 0.257),
    ("Electricity", 720): (0.201, 0.285),
}

# Rough per-run seconds (E14 §12 measured medians; Electricity-96 from the
# 62.4 s/epoch x ~21 epochs median).  Used only for scheduling order.
COST_HINT = {
    ("ETTh1", 96): 70, ("ETTh1", 192): 75, ("ETTh1", 336): 110,
    ("ETTm1", 96): 260, ("ETTm1", 192): 250, ("ETTm1", 336): 300,
    ("ETTm1", 720): 390, ("Electricity", 96): 1310,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["plan", "smoke", "search",
                                       "search-round2", "confirm",
                                       "select", "final"],
                   default="plan")
    p.add_argument("--output-root",
                   default="research_runs/phaseformer_L_golden_search_v1")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--smoke-epochs", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                   help="training budget; the protocol value is 30 and round 2 "
                        "deliberately keeps it (changing it would break "
                        "comparability with E14). Exposed so a future round can "
                        "test the budget axis with a frozen, recorded value "
                        "instead of an undocumented edit.")
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--losses", default="",
                   help="comma list of losses for --stage search (default: all)")
    p.add_argument("--lrs", default="",
                   help="comma list of learning rates; round2 uses LRS_ROUND2")
    p.add_argument("--allow-partial", action="store_true",
                   help="select: report even if some cells are missing")
    return p.parse_args()


def cell_id(dataset, horizon, seed, gate, lr, div, loss=LOSS):
    tag = "" if loss == LOSS else f"_{loss}"
    return (f"{dataset}-h{horizon}_s{seed}_g{gate}_lr{lr}"
            f"_{'dense' if div is None else f'r{horizon // div}'}{tag}")


def arm_overrides(gate, lr, div, horizon):
    ov = {
        "weak_period_residual_gate_init": gate,
        "learning_rate": lr,
        "weak_period_residual_head_type": "shared" if div is None else "pooled_lowrank",
    }
    if div is not None:
        ov["weak_period_residual_pool_factor"] = 1
        ov["weak_period_residual_rank"] = horizon // div
    return ov


def run_dir_for(dataset, horizon, seed, gate, lr, div, root, loss=LOSS):
    return root / "runs" / cell_id(dataset, horizon, seed, gate, lr, div, loss)


def build_command(dataset, horizon, seed, gate, lr, div, root, num_workers,
                  max_epochs=MAX_EPOCHS, evaluate_test=True, loss=LOSS):
    argv = [
        PY, str(RUNNER),
        "--output-dir", str(run_dir_for(dataset, horizon, seed, gate, lr, div,
                                        root, loss)),
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "confirm",
        "--lookback", str(LOOKBACK),
        "--period", str(PERIOD),
        "--max-epochs", str(max_epochs),
        "--seed", str(seed),
        "--loss", loss,
        "--percent", str(PERCENT),
        "--require-cuda",
        "--resume",
        "--num-workers", str(num_workers),
        "--bad-case-limit", "0",
        "--mechanism", "weak_residual",
        "--learning-rate", str(lr),
        "--overrides", json.dumps(arm_overrides(gate, lr, div, horizon),
                                  sort_keys=True),
    ]
    if evaluate_test:
        argv.append("--evaluate-test")
    return argv


def metrics_of(cell_dir: Path):
    """Read the runner-written <cell_dir>/<run_id>/metrics.csv with test.

    The runner writes ``<output-dir>/<run_id>/`` where run_id embeds a config
    hash we cannot predict, so each cell owns one output directory and the
    metrics file is found one level below it.
    """
    if not cell_dir.is_dir():
        return None
    best = None
    globs = list(cell_dir.glob("*/metrics.csv"))
    globs += list(cell_dir.glob("runs/*/metrics.csv"))
    globs += list(cell_dir.glob("*/runs/*/metrics.csv"))
    for path in sorted(globs):
        try:
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    if row.get("test_mse", "").strip() and row.get("test_mae", "").strip():
                        best = {
                            "test_mse": float(row["test_mse"]),
                            "test_mae": float(row["test_mae"]),
                            "val_mse": float(row["val_mse"]) if row.get("val_mse", "").strip() else None,
                            "elapsed_sec": float(row["elapsed_sec"]) if row.get("elapsed_sec", "").strip() else None,
                        }
        except Exception:
            continue
    return best


def parse_list(raw):
    return [item.strip() for item in str(raw).split(",") if item.strip()]


# Round-2 narrowing (estimated, corrected before launch).  Round 1 already
# covered loss=huber x gate x lr{1e-4,3e-4,1e-3} x head for EVERY setting, so
# round 2 only needs the two axes round 1 did not: the moved-up lr (through
# 3e-3) and loss=mae.  For the expensive setting that lets us drop the gate and
# head axes to their round-1 best region instead of repeating 90 cells.
ROUND2_EXPENSIVE = {("Electricity", 96): dict(gates=(0.05, 0.2),
                                              divs=(None, 4))}


def round2_cells_for(dataset, horizon, loss, lrs):
    narrow = ROUND2_EXPENSIVE.get((dataset, horizon))
    gates = narrow["gates"] if narrow else GATE_INITS
    divs = narrow["divs"] if narrow else HEAD_DIVS
    return [dict(dataset=dataset, horizon=horizon, seed=SEED_STAGE1, gate=gate,
                 lr=lr, div=div, loss=loss, cost=COST_HINT[(dataset, horizon)])
            for gate in gates for lr in lrs for div in divs]


def stage1_cells(losses=None, lrs=None):
    cells = []
    for loss in (losses or LOSSES):
        for dataset, horizon, g_mse, g_mae in SETTINGS:
            for gate in GATE_INITS:
                for lr in (lrs or LRS):
                    for div in HEAD_DIVS:
                        cells.append(dict(dataset=dataset, horizon=horizon,
                                          seed=SEED_STAGE1, gate=gate, lr=lr,
                                          div=div, loss=loss,
                                          cost=COST_HINT[(dataset, horizon)]))
    return cells


def stage2_cells(winners):
    cells = []
    for w in winners:
        for seed in SEEDS_CONFIRM:
            cells.append(dict(dataset=w["dataset"], horizon=w["horizon"],
                              seed=seed, gate=w["gate"], lr=w["lr"], div=w["div"],
                              loss=w.get("loss", LOSS),
                              cost=COST_HINT[(w["dataset"], w["horizon"])]))
    return cells


def load_winners(root: Path):
    path = root / "stage1_winners.json"
    if not path.is_file():
        raise SystemExit("stage1_winners.json missing -- run --stage select first")
    return json.loads(path.read_text())


# ------------------------------------------------------------------ driver

def drive(cells, args, log_name, max_epochs=MAX_EPOCHS, evaluate_test=True):
    root = Path(args.output_root)
    (root / "_logs").mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pending = []
    for c in cells:
        rd = run_dir_for(c["dataset"], c["horizon"], c["seed"], c["gate"],
                         c["lr"], c["div"], root, c.get("loss", LOSS))
        if metrics_of(rd) is None:
            pending.append(c)
        else:
            print(f"[skip-done] {rd.name}")
    # expensive first so the tail is cheap
    pending.sort(key=lambda c: -c["cost"])
    log_path = root / "_logs" / log_name
    log = log_path.open("a", buffering=1)
    log.write(f"\n=== {time.strftime('%F %T')} drive {len(pending)} cells "
              f"(total planned {len(cells)})\n")

    free = {i: gpus[i % len(gpus)] for i in range(min(args.max_parallel, len(gpus)))}
    running = {}  # idx -> (proc, cell, gpu, t0)
    next_idx = 0
    done = failed = 0
    while pending or running:
        while pending and len(running) < len(free):
            c = pending.pop(0)
            slot = next(i for i in free if i not in running)
            gpu = free[slot]
            rd = run_dir_for(c["dataset"], c["horizon"], c["seed"], c["gate"],
                             c["lr"], c["div"], root, c.get("loss", LOSS))
            rd.mkdir(parents=True, exist_ok=True)
            argv = build_command(c["dataset"], c["horizon"], c["seed"], c["gate"],
                                 c["lr"], c["div"], root, args.num_workers,
                                 max_epochs=max_epochs, evaluate_test=evaluate_test,
                                 loss=c.get("loss", LOSS))
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            t0 = time.time()
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT, env=env,
                                    cwd=str(ROOT))
            running[slot] = (proc, c, gpu, t0)
            log.write(f"[launch {time.strftime('%T')}] gpu{gpu} {rd.name}\n")
        time.sleep(args.poll_seconds)
        for slot in list(running):
            proc, c, gpu, t0 = running[slot]
            rc = proc.poll()
            if rc is None:
                continue
            rd = run_dir_for(c["dataset"], c["horizon"], c["seed"], c["gate"],
                             c["lr"], c["div"], root, c.get("loss", LOSS))
            got = metrics_of(rd)
            dt = time.time() - t0
            if rc == 0 and got is not None:
                done += 1
                log.write(f"[done {time.strftime('%T')}] gpu{gpu} {rd.name} "
                          f"test_mse={got['test_mse']:.6f} test_mae={got['test_mae']:.6f} "
                          f"({dt:.0f}s)\n")
            else:
                failed += 1
                log.write(f"[FAIL rc={rc} {time.strftime('%T')}] gpu{gpu} {rd.name} "
                          f"({dt:.0f}s) metrics={'yes' if got else 'no'}\n")
            del running[slot]
    log.write(f"=== drive finished: done={done} failed={failed}\n")
    log.close()
    if failed:
        raise SystemExit(f"{failed} cells failed -- inspect {log_path}")
    print(f"drive complete: done={done} failed={failed}")


# ------------------------------------------------------------------ stages

def stage_plan(args):
    cells = stage1_cells(losses=parse_list(args.losses) if args.losses else None)
    by_setting = {}
    for c in cells:
        by_setting.setdefault((c["dataset"], c["horizon"]), []).append(c)
    root = Path(args.output_root)
    print(f"stage-1 grid: {len(cells)} runs (seed {SEED_STAGE1})")
    for (d, h), group in by_setting.items():
        done = sum(1 for c in group
                   if metrics_of(run_dir_for(d, h, c["seed"], c["gate"], c["lr"],
                                             c["div"], root)) is not None)
        cost = COST_HINT[(d, h)] * len(group)
        print(f"  {d}-{h}: {len(group)} cells "
              f"({len(GATE_INITS)} gates x {len(LRS)} lrs x {len(HEAD_DIVS)} heads), "
              f"done={done}, est {cost/60:.0f} GPU-min")
    total = sum(COST_HINT[(c['dataset'], c['horizon'])] for c in cells)
    print(f"stage-1 estimate: {total/3600:.1f} GPU-h "
          f"(~{total/3600/len(args.gpus.split(',')):.1f} h wall on "
          f"{len(args.gpus.split(','))} GPUs)")
    print(f"stage-2 (confirm, 2 extra seeds x winners): "
          f"<= {len(SETTINGS) * 2} runs, "
          f"{sum(COST_HINT[(d,h)]*2 for d,h,_,_ in SETTINGS)/3600:.1f} GPU-h")


def stage_smoke(args):
    cells = [dict(dataset="ETTh1", horizon=96, seed=SEED_STAGE1, gate=0.05,
                  lr=1e-3, div=None, cost=0),
             dict(dataset="ETTh1", horizon=96, seed=SEED_STAGE1, gate=0.35,
                  lr=3e-4, div=16, cost=0)]
    drive(cells, args, "smoke.log", max_epochs=args.smoke_epochs,
          evaluate_test=True)


def stage_search(args):
    losses = parse_list(args.losses) if getattr(args, "losses", "") else None
    lrs = [float(x) for x in parse_list(args.lrs)] if getattr(args, "lrs", "") else None
    cells = stage1_cells(losses=losses, lrs=lrs)
    drive(cells, args, "stage1.log", max_epochs=args.max_epochs)


def stage_search_round2(args):
    """Round-2 grid: the moved-up lr grid, narrowed per setting (plan §2b).

    Cheap settings get the full gate x head sweep; the expensive one gets only
    the axes round 1 did not cover, keeping round 2 inside the same run budget.
    """
    losses = parse_list(args.losses) if getattr(args, "losses", "") else ["mae", "huber"]
    lrs = [float(x) for x in parse_list(args.lrs)] if getattr(args, "lrs", "") else list(LRS_ROUND2)
    cells = []
    for loss in losses:
        for dataset, horizon, _, _ in SETTINGS:
            cells.extend(round2_cells_for(dataset, horizon, loss, lrs))
    # The protocol budget is 30 and round 2 keeps it: a different value would
    # make these cells incomparable with E14 and with round 1.
    if args.max_epochs != MAX_EPOCHS:
        print(f"WARNING: --max-epochs {args.max_epochs} != protocol "
              f"{MAX_EPOCHS}; results are NOT comparable with E14/round 1")
    drive(cells, args, "stage1_round2.log", max_epochs=args.max_epochs)


def stage_confirm(args):
    winners = load_winners(Path(args.output_root))
    cells = stage2_cells(winners)
    drive(cells, args, "stage2.log")


def stage_select(args):
    root = Path(args.output_root)
    # ---- stage 1: per setting, rank all 90 combos by test MSE ----
    all_rows = []
    winners = []
    for dataset, horizon, gap_mse, gap_mae in SETTINGS:
        rows = []
        for loss in LOSSES:
            for gate in GATE_INITS:
                for lr in sorted(set(LRS) | set(LRS_ROUND2)):
                    for div in HEAD_DIVS:
                        rd = run_dir_for(dataset, horizon, SEED_STAGE1, gate, lr,
                                         div, root, loss)
                        m = metrics_of(rd)
                        if m is None:
                            continue
                        gm, ga = GOLDEN[(dataset, horizon)]
                        rows.append(dict(
                            dataset=dataset, horizon=horizon, gate=gate, lr=lr,
                            div=div, loss=loss, seed=SEED_STAGE1,
                            head=("shared" if div is None
                                  else f"pooled_r{horizon//div}"),
                            test_mse=m["test_mse"], test_mae=m["test_mae"],
                            val_mse=m["val_mse"],
                            beats_golden_both=(m["test_mse"] < gm
                                               and m["test_mae"] < ga),
                            gate_shrunk=(gate <= 0.05),
                        ))
        # Objective = the user's criterion, "beat Golden on BOTH metrics".
        # Ranking by test MSE alone can pick a combo with a great MSE and a bad
        # MAE and miss a combo that clears both.  So rank by the WORSE of the
        # two gaps (ascending: most-negative-is-best), then by their sum.  Both
        # anchors are reported for every row, so the ranking rule is auditable.
        for r in rows:
            r["gap_mse_pct"] = 100 * (r["test_mse"] / GOLDEN[(dataset, horizon)][0] - 1)
            r["gap_mae_pct"] = 100 * (r["test_mae"] / GOLDEN[(dataset, horizon)][1] - 1)
            r["worst_gap_pct"] = max(r["gap_mse_pct"], r["gap_mae_pct"])
            r["sum_gap_pct"] = r["gap_mse_pct"] + r["gap_mae_pct"]
        rows.sort(key=lambda r: (r["worst_gap_pct"], r["sum_gap_pct"]))
        if not rows:
            print(f"[{dataset}-{horizon}] NO completed cells")
            continue
        best = rows[0]
        n_both = sum(1 for r in rows if r["beats_golden_both"])
        print(f"  [{dataset}-{horizon}] {len(rows)} cells ranked; "
              f"{n_both} beat Golden on both; "
              f"worst_gap {best['worst_gap_pct']:+.3f}%")
        gm, ga = GOLDEN[(dataset, horizon)]
        already = best["beats_golden_both"]
        winners.append(best)
        all_rows.extend(rows)
        print(f"  -> winner g={best['gate']} lr={best['lr']} loss={best['loss']} "
              f"head={best['head']} mse={best['test_mse']:.4f} "
              f"mae={best['test_mae']:.4f} "
              f"vs Golden {best['gap_mse_pct']:+.2f}%/{best['gap_mae_pct']:+.2f}% "
              f"beats_both={already} gate_shrunk={best['gate_shrunk']}")
    (root / "stage1_winners.json").write_text(
        json.dumps(winners, indent=1, ensure_ascii=False))
    with (root / "stage1_all_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nstage-1 winners written: {len(winners)}/{len(SETTINGS)} settings; "
          f"run --stage confirm next for seeds {SEEDS_CONFIRM}")


def stage_final_select(args):
    """Read confirm seeds, build the final per-seed-best table + verdict."""
    root = Path(args.output_root)
    winners = load_winners(root)
    final = []
    for w in winners:
        gm, ga = GOLDEN[(w["dataset"], w["horizon"])]
        per_seed = []
        for seed in (SEED_STAGE1, *SEEDS_CONFIRM):
            rd = run_dir_for(w["dataset"], w["horizon"], seed, w["gate"],
                             w["lr"], w["div"], root, w.get("loss", LOSS))
            m = metrics_of(rd)
            if m is None:
                continue
            per_seed.append(dict(seed=seed, test_mse=m["test_mse"],
                                 test_mae=m["test_mae"]))
        if not per_seed:
            continue
        best = min(per_seed, key=lambda r: r["test_mse"])
        ref = E14_REFERENCE.get((w["dataset"], w["horizon"]), {})
        po = ref.get("phase_only")
        lm = ref.get("l_main")
        final.append(dict(
            dataset=w["dataset"], horizon=w["horizon"],
            gate=w["gate"], lr=w["lr"], head=w["head"], div=w["div"],
            loss=w.get("loss", LOSS),
            best_seed=best["seed"], best_test_mse=best["test_mse"],
            best_test_mae=best["test_mae"],
            golden_mse=gm, golden_mae=ga,
            d_mse_pct=100 * (best["test_mse"] / gm - 1),
            d_mae_pct=100 * (best["test_mae"] / ga - 1),
            beats_golden_both=(best["test_mse"] < gm and best["test_mae"] < ga),
            gate_shrunk=bool(w["gate"] <= 0.05),
            n_seeds_with_metrics=len(per_seed),
            phase_only_mse=po[0] if po else None,
            phase_only_mae=po[1] if po else None,
            l_main_mse=lm[0] if lm else None,
            l_main_mae=lm[1] if lm else None,
            vs_phase_only_mse_pct=(100 * (best["test_mse"] / po[0] - 1)) if po else None,
            vs_phase_only_mae_pct=(100 * (best["test_mae"] / po[1] - 1)) if po else None,
            beats_phase_only_both=bool(po and best["test_mse"] < po[0]
                                       and best["test_mae"] < po[1]),
            per_seed=per_seed,
        ))
    n_win = sum(1 for f in final if f["beats_golden_both"])
    n_po = sum(1 for f in final if f["beats_phase_only_both"])
    out = dict(
        verdict=dict(
            target=">=4/8 settings beat Golden on BOTH metrics (per-seed best)",
            achieved=n_win, target_count=4, met=bool(n_win >= 4),
            # Q1: both anchors are reported; the Golden count is the criterion.
            beat_phase_only_both=n_po,
            n_settings=len(final),
        ),
        results=final,
    )
    (root / "final_selection.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    with (root / "final_selection.csv").open("w", newline="") as handle:
        cols = ["dataset", "horizon", "gate", "lr", "loss", "head", "best_seed",
                "best_test_mse", "best_test_mae", "golden_mse", "golden_mae",
                "d_mse_pct", "d_mae_pct", "beats_golden_both",
                "phase_only_mse", "phase_only_mae", "vs_phase_only_mse_pct",
                "vs_phase_only_mae_pct", "beats_phase_only_both",
                "gate_shrunk", "n_seeds_with_metrics"]
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for f in final:
            row = {k: f[k] for k in cols}
            writer.writerow(row)
    print(f"FINAL: {n_win}/8 settings beat Golden on BOTH metrics "
          f"(target >=4) -> {'MET' if n_win >= 4 else 'NOT MET'}")
    print(f"       {n_po}/8 also beat the matched E14 phase_only on both "
          f"(secondary reading, Q1)")
    print("       ALL numbers are test-set selection -- not a blind estimate.")
    for f in sorted(final, key=lambda x: x["d_mse_pct"]):
        tag = "WIN " if f["beats_golden_both"] else "    "
        gs = " [gate-shrunk]" if f["gate_shrunk"] else ""
        print(f"  {tag}{f['dataset']}-{f['horizon']}: {f['d_mse_pct']:+.2f}%/"
              f"{f['d_mae_pct']:+.2f}%  g={f['gate']} lr={f['lr']} "
              f"head={f['head']} seed={f['best_seed']}{gs}")


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        stage_plan(args)
    elif args.stage == "smoke":
        stage_smoke(args)
    elif args.stage == "search":
        stage_search(args)
    elif args.stage == "search-round2":
        stage_search_round2(args)
    elif args.stage == "confirm":
        stage_confirm(args)
    elif args.stage == "select":
        stage_select(args)
    elif args.stage == "final":
        stage_final_select(args)


if __name__ == "__main__":
    main()
