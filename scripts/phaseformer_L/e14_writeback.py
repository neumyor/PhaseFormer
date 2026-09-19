#!/usr/bin/env python3
"""E14 stage B+ / §4.2 write-back: audit the matrix and build the main table.

Consumes E14's stage-A manifest plus the single-test-read ``results.csv`` and
emits

* ``main_table.csv``  -- one row per setting with every §4.2 column,
* ``main_table.md``   -- the same rows formatted for the minipaper,
* ``variant_table.csv`` -- the per-arm macro averages behind the variant rows,
* ``claims.json``     -- the frozen §4.0 verdicts A/B/C/D, reported under both
  readings where the wording is ambiguous,
* ``audit.json``      -- the stage-5 row/column/reconcile audit.

Reads nothing from a split itself: every number comes from files another stage
wrote, so this stays a pure aggregation and cannot leak test information.

Claim definitions (frozen 2026-09-18, minipaper §4.0):

* **A** -- PhaseFormer-L shows no double-metric regression beyond 1.0% against
  the matched ``phase_only`` rerun on the 24 main settings.  "双指标回退超过
  1.0%" is ambiguous between "both metrics" and "either metric", so both are
  reported; the strict *either-metric* reading is primary because A is a
  no-degradation claim and the permissive reading could hide a real regression.
* **B** -- on the datasets the §4.7 diagnostic marks ``s=1`` (ETTh2, ETTm2,
  Weather), at least 3/4 of those settings beat ``phase_only`` on both metrics.
  Every diagnostic miss is listed.
* **C** -- count of settings whose three-seed mean plus sample std is below the
  Golden value on both metrics (strict existing standard), with no minimum.
* **D** -- ``|dMSE|`` and ``|dMAE|`` macro-averaged over q=1/8 versus the dense
  PhaseFormer-L are at most 0.5%.

Usage::

    python scripts/phaseformer_L/e14_writeback.py \\
        --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \\
        --results  research_runs/phaseformer_L_e14_main_v1/results.csv \\
        --stats    research_runs/phaseformer_L_e19_predictive_v1/level_statistics.csv \\
        --golden   docs/PhaseFormer_gold_standard.md \\
        --output-root research_runs/phaseformer_L_e14_main_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAIN_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
TRAFFIC = "Traffic"
SEEDS = (2021, 2022, 2023)

PHASE_ONLY = "phase_only"
L_MAIN = "l_main"
ARM_ORDER = ("phase_only", "l_main", "l_q1_4", "l_q1_8", "l_rcrf", "a1")
ARM_LABEL = {
    "phase_only": "`phase_only`",
    "l_main": "PhaseFormer-L",
    "l_q1_4": "L-q1/4",
    "l_q1_8": "L-q1/8",
    "l_rcrf": "L-rcrf",
    "a1": "A1",
}

# Frozen §4.0 thresholds.
REGRESSION_BOUND_PCT = 1.0
CLAIM_B_FRACTION = 0.75
CLAIM_D_BOUND_PCT = 0.5

# Diagnostic column (minipaper §3.4.3 / §4.7), frozen at the midpoint of the
# separable interval of the sign-known datasets.  Not part of the model (D-5).
NU_STAT = "tau_hat_steps"
NU_STAR = 57.35
S1_DATASETS = ("ETTh2", "ETTm2", "Weather")

# External citation, NOT a local result.  FITS (ICLR 2024 Spotlight) MSE at
# lookback 720, transcribed from the official repository's final results table
# (https://github.com/VEWOXIC/FITS, README "Result Update"), fetched 2026-09-19,
# which states those numbers match the ICLR camera-ready version after the
# `drop_last` bug fix.  Needed by minipaper §4.2 must-answer (a), which asks
# whether ETTh2 reaches "the FITS numbers"; the repository's own Golden file
# contains no external model at all.  The source table reports MSE ONLY, so no
# FITS MAE is recorded here and none may be inferred.  Provenance and caveats:
# docs/PhaseFormer_L_external_refs.md.
FITS_MSE = {
    ("ETTh1", 96): 0.372, ("ETTh1", 192): 0.404,
    ("ETTh1", 336): 0.427, ("ETTh1", 720): 0.424,
    ("ETTh2", 96): 0.271, ("ETTh2", 192): 0.331,
    ("ETTh2", 336): 0.354, ("ETTh2", 720): 0.377,
    ("ETTm1", 96): 0.303, ("ETTm1", 192): 0.337,
    ("ETTm1", 336): 0.366, ("ETTm1", 720): 0.415,
    ("ETTm2", 96): 0.162, ("ETTm2", 192): 0.216,
    ("ETTm2", 336): 0.268, ("ETTm2", 720): 0.348,
    ("Weather", 96): 0.143, ("Weather", 192): 0.186,
    ("Weather", 336): 0.236, ("Weather", 720): 0.307,
    ("Electricity", 96): 0.134, ("Electricity", 192): 0.149,
    ("Electricity", 336): 0.165, ("Electricity", 720): 0.203,
    ("Traffic", 96): 0.385, ("Traffic", 192): 0.397,
    ("Traffic", 336): 0.410, ("Traffic", 720): 0.448,
}
FITS_NOTE = (
    "FITS MSE is an external citation (github.com/VEWOXIC/FITS, fetched "
    "2026-09-19), MSE-only by construction; see "
    "docs/PhaseFormer_L_external_refs.md. Used for must-answer (a); it never "
    "enters claims A-D."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--stats", default="")
    parser.add_argument("--params", default="",
                        help="parameter_table.csv from e14_params.py; default: "
                             "<output-root>/parameter_table.csv")
    parser.add_argument("--golden", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_golden(path: Path) -> dict:
    """Parse the Golden table; values are 3-decimal MSE/MAE."""
    golden = {}
    # The dataset token must admit digits: only Weather/Electricity/Traffic are
    # pure letters, while ETTh1/ETTh2/ETTm1/ETTm2 all carry one.  A letters-only
    # class silently matched 12 of the 28 rows.
    row_re = re.compile(
        r"^\|\s*([A-Za-z0-9]+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row_re.match(line.strip())
        if not match:
            continue
        dataset, horizon, mse, mae = match.groups()
        if dataset == "Dataset":
            continue
        golden[(dataset, int(horizon))] = (float(mse), float(mae))
    return golden


def read_results(path: Path) -> dict:
    """(arm, dataset, horizon) -> {'mse': [...], 'mae': [...], 'gate': [...]}."""
    index: dict = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            arm = str(row.get("arm", "")).strip()
            dataset = str(row.get("dataset", "")).strip()
            try:
                horizon = int(row["horizon"])
            except (KeyError, TypeError, ValueError):
                continue
            entry = index.setdefault(
                (arm, dataset, horizon), {"mse": [], "mae": [], "gate": [],
                                          "seeds": [], "status": []})
            for column, bucket in (("test_mse", "mse"), ("test_mae", "mae")):
                raw = str(row.get(column, "")).strip()
                if raw:
                    try:
                        entry[bucket].append(float(raw))
                    except ValueError:
                        pass
            gate = str(row.get("gate_value", "")).strip()
            if gate:
                try:
                    entry["gate"].append(float(gate))
                except ValueError:
                    pass
            if row.get("seed"):
                entry["seeds"].append(int(row["seed"]))
            if row.get("status"):
                entry["status"].append(str(row["status"]))
    return index


def read_stats(path: Path) -> dict:
    if not path or not Path(path).exists():
        return {}
    out = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (row["dataset"], int(row["horizon"]))
            except (KeyError, TypeError, ValueError):
                continue
            out[key] = row
    return out


def read_parameters(path: Path) -> dict:
    """(arm, horizon) -> parameter accounting, from ``e14_params.py``.

    The corrector's size legitimately depends on the horizon (a rank-``H/4``
    head at H=720 is 7.5x the size of one at H=96), but it must NOT depend on
    the seed, so constancy is checked across seeds within a (arm, horizon)
    rather than across the whole arm.
    """
    if not path or not Path(path).exists():
        return {}
    cells: dict = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            arm = str(row.get("arm", "")).strip()
            try:
                horizon = int(row["horizon"])
                total = int(row["total_params"])
                residual = int(row["residual_params"])
                share = float(row["residual_share"])
            except (KeyError, TypeError, ValueError):
                continue
            gate = str(row.get("gate_value_from_checkpoint", "")).strip()
            entry = cells.setdefault((arm, horizon), {"totals": set(),
                                                      "residuals": set(),
                                                      "shares": set(), "n": 0,
                                                      "gates": set()})
            if gate:
                try:
                    entry["gates"].add(round(float(gate), 6))
                except ValueError:
                    pass
            entry["totals"].add(total)
            entry["residuals"].add(residual)
            entry["shares"].add(round(share, 6))
            entry["n"] += 1
    out = {}
    for key, entry in cells.items():
        if not entry["totals"]:
            continue
        out[key] = {
            "n_seeds": entry["n"],
            "total_params": sorted(entry["totals"])[0],
            "residual_params": sorted(entry["residuals"])[0],
            "residual_share": sorted(entry["shares"])[0],
            "gate_from_checkpoint": (sorted(entry["gates"])[0]
                                     if entry["gates"] else None),
            "constant_across_seeds": len(entry["totals"]) == 1
            and len(entry["residuals"]) == 1,
        }
    return out


def mean_std(values):
    if not values:
        return None, None, 0
    if len(values) == 1:
        return float(values[0]), 0.0, 1
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)), int(arr.size)


def pct_change(new, baseline):
    if new is None or baseline in (None, 0):
        return None
    return 100.0 * (new - baseline) / baseline


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    results = read_results(Path(args.results))
    golden = read_golden(Path(args.golden))
    stats = read_stats(args.stats)
    params_path = args.params or str(Path(args.output_root) / "parameter_table.csv")
    params = read_parameters(Path(params_path))

    # Which settings each arm is declared on, straight from the manifest.
    declared: dict = {}
    for cell in manifest.get("cells", []):
        declared.setdefault(str(cell["arm"]), set()).add(
            (str(cell["dataset"]), int(cell["horizon"])))

    settings = sorted(
        {(d, h) for (d, h) in golden},
        key=lambda k: (MAIN_DATASETS.index(k[0]) if k[0] in MAIN_DATASETS else 99,
                       k[1]),
    )

    rows = []
    incomplete = []
    for dataset, horizon in settings:
        entry = {"dataset": dataset, "horizon": horizon,
                 "setting": f"{dataset}-{horizon}",
                 "is_traffic_appendix": dataset == TRAFFIC}
        gold = golden[(dataset, horizon)]
        entry["golden_mse"], entry["golden_mae"] = gold
        # The FITS delta can only be computed once the arm loop below has
        # populated l_main_mse, so it is filled in after that loop, not here.
        entry["fits_mse"] = FITS_MSE.get((dataset, horizon))

        for arm in ARM_ORDER:
            bucket = results.get((arm, dataset, horizon))
            if not bucket or not bucket["mse"] or not bucket["mae"]:
                entry[f"{arm}_mse"] = None
                entry[f"{arm}_mae"] = None
                entry[f"{arm}_std_mse"] = None
                entry[f"{arm}_std_mae"] = None
                entry[f"{arm}_n"] = 0
                entry[f"{arm}_gate_mean"] = None
                continue
            mse, mse_sd, n = mean_std(bucket["mse"])
            mae, mae_sd, _ = mean_std(bucket["mae"])
            entry[f"{arm}_mse"] = round(mse, 6)
            entry[f"{arm}_mae"] = round(mae, 6)
            entry[f"{arm}_std_mse"] = round(mse_sd, 6)
            entry[f"{arm}_std_mae"] = round(mae_sd, 6)
            entry[f"{arm}_n"] = n
            entry[f"{arm}_gate_mean"] = (
                round(float(np.mean(bucket["gate"])), 6) if bucket["gate"] else None)

        # §4.2's gate column and §4.7's rho versus g.  results.csv is preferred
        # (it comes from the single test read), but the E3-lineage metrics.csv
        # records no gate at all, so for those reused cells the value is read off
        # the checkpoint by e14_params.py instead.  The source is recorded so the
        # table note can say which cells came from where.
        for g_arm in (L_MAIN, "l_q1_4", "l_q1_8"):
            from_results = entry.get(f"{g_arm}_gate_mean")
            from_ckpt = (params.get((g_arm, horizon)) or {}).get(
                "gate_from_checkpoint")
            entry[f"{g_arm}_gate_mean"] = (
                from_results if from_results is not None else from_ckpt)
            entry[f"{g_arm}_gate_source"] = (
                "results_csv" if from_results is not None
                else ("checkpoint" if from_ckpt is not None else None))

        # Required comparisons.
        entry["phaseformer_l_vs_fits_pct"] = (
            round(pct_change(entry.get("l_main_mse"), FITS_MSE[(dataset, horizon)]), 4)
            if (dataset, horizon) in FITS_MSE else None)
        po_mse, po_mae = entry.get("phase_only_mse"), entry.get("phase_only_mae")
        if po_mse is None or entry.get("l_main_mse") is None:
            incomplete.append(entry["setting"] + " (missing phase_only or l_main)")
        entry["delta_mse_pct"] = pct_change(entry.get("l_main_mse"), po_mse)
        entry["delta_mae_pct"] = pct_change(entry.get("l_main_mae"), po_mae)
        entry["delta_mse_pct"] = (None if entry["delta_mse_pct"] is None
                                  else round(entry["delta_mse_pct"], 4))
        entry["delta_mae_pct"] = (None if entry["delta_mae_pct"] is None
                                  else round(entry["delta_mae_pct"], 4))

        better = (entry.get("l_main_mse") is not None and po_mse is not None
                  and entry["l_main_mse"] < po_mse
                  and entry["l_main_mae"] < po_mae)
        entry["double_metric_better"] = bool(better)

        # Two Golden comparisons are reported, because the two governing
        # documents define them differently:
        #  * "双指标提升" per PhaseFormer_gold_standard.md §4 = both metrics'
        #    THREE-SEED MEAN below the Golden value;
        #  * "稳定超过" per minipaper §4.0 (claim C) = mean PLUS sample std still
        #    below the Golden value, which also answers the standard's caution
        #    that a 3-decimal Golden must not turn rounding into a gain.
        for arm in (L_MAIN, PHASE_ONLY):
            if entry.get(f"{arm}_mse") is None:
                entry[f"{arm}_stable_beyond_golden"] = False
                entry[f"{arm}_double_metric_improvement"] = False
                continue
            entry[f"{arm}_double_metric_improvement"] = bool(
                entry[f"{arm}_mse"] < gold[0] and entry[f"{arm}_mae"] < gold[1])
            entry[f"{arm}_stable_beyond_golden"] = bool(
                entry[f"{arm}_mse"] + (entry[f"{arm}_std_mse"] or 0.0) < gold[0]
                and entry[f"{arm}_mae"] + (entry[f"{arm}_std_mae"] or 0.0) < gold[1])

        # Diagnostic column.
        stat = stats.get((dataset, horizon))
        if stat:
            try:
                tau = float(stat[NU_STAT])
                entry["tau_hat_steps"] = round(tau, 4)
                entry["diagnostic_s"] = int(tau > NU_STAR)
            except (KeyError, TypeError, ValueError):
                entry["tau_hat_steps"] = None
                entry["diagnostic_s"] = None
        else:
            entry["tau_hat_steps"] = None
            entry["diagnostic_s"] = None
        if entry["diagnostic_s"] is not None and entry["delta_mse_pct"] is not None:
            helps = entry["delta_mse_pct"] < 0.0
            entry["diagnostic_hit"] = bool(entry["diagnostic_s"]) == helps
        else:
            entry["diagnostic_hit"] = None
        rows.append(entry)

    # ---- claim A -----------------------------------------------------------
    main_rows = [r for r in rows if not r["is_traffic_appendix"]]
    def regressions(subset):
        strict, both = [], []
        for row in subset:
            dm, da = row["delta_mse_pct"], row["delta_mae_pct"]
            if dm is None or da is None:
                continue
            if dm > REGRESSION_BOUND_PCT or da > REGRESSION_BOUND_PCT:
                strict.append(row["setting"])
            if dm > REGRESSION_BOUND_PCT and da > REGRESSION_BOUND_PCT:
                both.append(row["setting"])
        return strict, both

    strict_24, both_24 = regressions(main_rows)
    strict_all, both_all = regressions(rows)
    claim_a = {
        "bound_pct": REGRESSION_BOUND_PCT,
        "primary_reading": "either_metric",
        "violations_either_metric_main24": strict_24,
        "verdict_either_metric": not strict_24,
        "violations_both_metrics_main24": both_24,
        "verdict_both_metrics": not both_24,
        "violations_either_metric_all28": strict_all,
    }

    # ---- claim B -----------------------------------------------------------
    s1_rows = [r for r in rows if r["dataset"] in S1_DATASETS]
    wins = [r["setting"] for r in s1_rows if r["double_metric_better"]]
    need = int(np.ceil(CLAIM_B_FRACTION * len(s1_rows))) if s1_rows else 0
    misses = [r["setting"] for r in rows
              if r["diagnostic_hit"] is False]
    claim_b = {
        "s1_datasets": list(S1_DATASETS),
        "s1_settings": len(s1_rows),
        "wins": len(wins),
        "required": need,
        "winning_settings": wins,
        "verdict": len(wins) >= need if s1_rows else None,
        "diagnostic_misses": misses,
    }

    # ---- must-answer (a): ETTh2 versus FITS --------------------------------
    etth2 = []
    for row in rows:
        if row["dataset"] != "ETTh2" or row["fits_mse"] is None:
            continue
        etth2.append({
            "setting": row["setting"],
            "golden_mse": row["golden_mse"],
            "fits_mse": row["fits_mse"],
            "phase_only_mse": row.get("phase_only_mse"),
            "phaseformer_l_mse": row.get("l_main_mse"),
            "phaseformer_l_vs_phase_only_pct": row["delta_mse_pct"],
            "phaseformer_l_vs_golden_pct": (
                None if row.get("l_main_mse") is None else
                round(pct_change(row["l_main_mse"], row["golden_mse"]), 4)),
            "phaseformer_l_vs_fits_pct": row.get("phaseformer_l_vs_fits_pct"),
            "reaches_fits_mse": (
                None if row.get("l_main_mse") is None
                else bool(row["l_main_mse"] <= row["fits_mse"])),
        })
    must_answer_a = {
        "question": "minipaper 4.2 (a): does ETTh2 close the gap to phase_only and "
                    "Golden, and does it reach the cited FITS numbers?",
        "fits_source": "github.com/VEWOXIC/FITS README 'Result Update', fetched 2026-09-19",
        "fits_reports_mae": False,
        "settings": etth2,
        "reaches_fits_count": sum(1 for e in etth2 if e["reaches_fits_mse"]),
        "note": "MSE only: the FITS table reports no MAE, so must-answer (a) is "
                "answered on MSE and the MAE side is left uncompared rather than "
                "filled from an unrelated source.",
    }

    # ---- claim C -----------------------------------------------------------
    claim_c = {
        "minimum_required": None,
        "definition": "three-seed mean + sample std strictly below Golden on BOTH metrics",
        "stable_beyond_golden_l_main": sum(
            1 for r in rows if r["l_main_stable_beyond_golden"]),
        "stable_beyond_golden_phase_only": sum(
            1 for r in rows if r["phase_only_stable_beyond_golden"]),
        "settings": [r["setting"] for r in rows if r["l_main_stable_beyond_golden"]],
        # The gold standard's own, weaker rule, reported alongside so the table
        # note can state both counts instead of conflating them.
        "double_metric_improvement_definition": "three-seed mean below Golden on BOTH metrics "
                                                "(PhaseFormer_gold_standard.md §4)",
        "double_metric_improvement_l_main": sum(
            1 for r in rows if r["l_main_double_metric_improvement"]),
        "double_metric_improvement_phase_only": sum(
            1 for r in rows if r["phase_only_double_metric_improvement"]),
        "double_metric_improvement_settings": [
            r["setting"] for r in rows if r["l_main_double_metric_improvement"]],
    }

    # ---- claim D -----------------------------------------------------------
    pairs = []
    for row in rows:
        direct, low = row.get("l_main_mse"), row.get("l_q1_8_mse")
        if direct is None or low is None:
            continue
        pairs.append((abs(pct_change(low, direct)),
                      abs(pct_change(row["l_q1_8_mae"], row["l_main_mae"]))))
    claim_d = {
        "bound_pct": CLAIM_D_BOUND_PCT,
        "n_settings": len(pairs),
        "macro_abs_delta_mse_pct": round(float(np.mean([p[0] for p in pairs])), 4)
        if pairs else None,
        "macro_abs_delta_mae_pct": round(float(np.mean([p[1] for p in pairs])), 4)
        if pairs else None,
    }
    claim_d["verdict"] = (
        claim_d["macro_abs_delta_mse_pct"] is not None
        and claim_d["macro_abs_delta_mse_pct"] <= CLAIM_D_BOUND_PCT
        and claim_d["macro_abs_delta_mae_pct"] <= CLAIM_D_BOUND_PCT
    )

    # ---- variant macro averages -------------------------------------------
    variant_rows = []
    for arm in ARM_ORDER:
        subset = [r for r in rows if r.get(f"{arm}_mse") is not None]
        if not subset:
            continue
        entry = {"arm": arm, "label": ARM_LABEL[arm],
                 "n_settings": len(subset),
                 "mean_mse": round(float(np.mean([r[f"{arm}_mse"] for r in subset])), 6),
                 "mean_mae": round(float(np.mean([r[f"{arm}_mae"] for r in subset])), 6)}
        deltas_mse = [pct_change(r[f"{arm}_mse"], r["phase_only_mse"]) for r in subset
                      if r.get("phase_only_mse") is not None]
        deltas_mae = [pct_change(r[f"{arm}_mae"], r["phase_only_mae"]) for r in subset
                      if r.get("phase_only_mae") is not None]
        entry["macro_delta_mse_pct"] = (round(float(np.mean(deltas_mse)), 4)
                                        if deltas_mse else None)
        entry["macro_delta_mae_pct"] = (round(float(np.mean(deltas_mae)), 4)
                                        if deltas_mae else None)
        main_only = [pct_change(r[f"{arm}_mse"], r["phase_only_mse"])
                     for r in subset
                     if not r["is_traffic_appendix"]
                     and r.get("phase_only_mse") is not None]
        entry["macro_delta_mse_pct_main24"] = (
            round(float(np.mean(main_only)), 4) if main_only else None)
        # §4.2 parameter columns: total (backbone + corrector + gate) and the
        # corrector alone.  Reported only when the arm resolves for every
        # horizon; otherwise the list of missing horizons is recorded instead.
        param_rows = [params.get((arm, r["horizon"])) for r in subset]
        present = [p for p in param_rows if p]
        if present and len(present) == len(subset):
            entry["total_params_per_horizon"] = json.dumps(
                {str(r["horizon"]): params[(arm, r["horizon"])]["total_params"]
                 for r in subset}, sort_keys=True)
            entry["residual_params_per_horizon"] = json.dumps(
                {str(r["horizon"]): params[(arm, r["horizon"])]["residual_params"]
                 for r in subset}, sort_keys=True)
            entry["residual_share_max"] = max(
                params[(arm, r["horizon"])]["residual_share"] for r in subset)
            entry["params_constant_across_seeds"] = all(
                params[(arm, r["horizon"])]["constant_across_seeds"] for r in subset)
        else:
            entry["total_params_per_horizon"] = None
            entry["residual_params_per_horizon"] = None
            entry["residual_share_max"] = None
            entry["params_constant_across_seeds"] = None

        stable_field = {
            L_MAIN: "l_main_stable_beyond_golden",
            PHASE_ONLY: "phase_only_stable_beyond_golden",
        }.get(arm)
        entry["stable_beyond_golden"] = (
            sum(1 for r in subset if r.get(stable_field)) if stable_field else None
        )
        variant_rows.append(entry)

    # ---- markdown ---------------------------------------------------------
    def fmt(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"

    lines = []
    for row in rows:
        po = (f"{fmt(row.get('phase_only_mse'))}/{fmt(row.get('phase_only_mae'))}"
              if row.get("phase_only_mse") is not None else "—")
        lm = (f"{fmt(row.get('l_main_mse'))}/{fmt(row.get('l_main_mae'))}"
              if row.get("l_main_mse") is not None else "—")
        delta = "—"
        if row["delta_mse_pct"] is not None:
            delta = f"{row['delta_mse_pct']:+.2f}%/{row['delta_mae_pct']:+.2f}%"
        gate = fmt(row.get("l_main_gate_mean"), 3)
        diag = "—" if row["diagnostic_s"] is None else str(row["diagnostic_s"])
        stable = "✓" if row["l_main_stable_beyond_golden"] else "✗"
        note = "探索性附录，不进入判定" if row["is_traffic_appendix"] else ""
        lines.append("| %s | %d | %s/%s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["dataset"], row["horizon"],
            f"{row['golden_mse']:.3f}", f"{row['golden_mae']:.3f}",
            po, lm, delta, gate, diag, stable, note))

    out_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() \
        else Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fields = ["dataset", "horizon", "setting", "is_traffic_appendix",
              "golden_mse", "golden_mae", "fits_mse", "phaseformer_l_vs_fits_pct"]
    for arm in ARM_ORDER:
        fields += [f"{arm}_mse", f"{arm}_mae", f"{arm}_std_mse", f"{arm}_std_mae",
                   f"{arm}_n", f"{arm}_gate_mean"]
    for arm in (L_MAIN, "l_q1_4", "l_q1_8"):
        fields.append(f"{arm}_gate_source")
    fields += ["delta_mse_pct", "delta_mae_pct", "double_metric_better",
               "l_main_stable_beyond_golden", "phase_only_stable_beyond_golden",
               "l_main_double_metric_improvement",
               "phase_only_double_metric_improvement",
               "tau_hat_steps", "diagnostic_s", "diagnostic_hit"]
    with (out_root / "main_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_root / "variant_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variant_rows[0].keys()))
        writer.writeheader()
        writer.writerows(variant_rows)
    (out_root / "claims.json").write_text(
        json.dumps({"A": claim_a, "B": claim_b, "C": claim_c, "D": claim_d,
                    "must_answer_a": must_answer_a,
                    "fits_note": FITS_NOTE},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_root / "audit.json").write_text(
        json.dumps({
            "settings_expected": len(settings),
            "settings_with_rows": len(rows),
            "settings_incomplete": incomplete,
            "arms_declared": {arm: sorted(f"{d}-{h}" for d, h in cells)
                              for arm, cells in declared.items()},
            "gate_means_by_setting": {
                r["setting"]: r.get("l_main_gate_mean") for r in rows},
            "disclosures": [
                "§4.2 mixes three gate priors: 0.2 for new l_main/l_q1_4/l_q1_8 "
                "cells, 0.5 held by the rcrf_nlinear_plain and gold_combo_* "
                "presets for l_rcrf/a1, and the Stage-0 frozen value on the 81 "
                "reused cells.",
                "The seven test-selected settings are not a blind sample; their "
                "reused cells come from experiments that participated in "
                "test-set selection.",
                "The diagnostic s column is not part of the model (D-5); it "
                "predicts which datasets need the level channel and any miss is "
                "reported.",
                "Parameter counts are read from each checkpoint's parameter "
                "shapes by e14_params.py and cross-checked against "
                "metrics.csv:parameter_count; FLOPs are not reported because the "
                "original Table 4 convention is not reproduced in this repository.",
            ],
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "event": "finished",
        "settings": len(rows),
        "incomplete": incomplete,
        "claim_A_either_metric": claim_a["verdict_either_metric"],
        "claim_A_violations": strict_24,
        "claim_B": claim_b["verdict"],
        "claim_B_wins": f"{claim_b['wins']}/{claim_b['s1_settings']}",
        "claim_C_stable_beyond_golden": claim_c["stable_beyond_golden_l_main"],
        "claim_C_double_metric_improvement": claim_c["double_metric_improvement_l_main"],
        "claim_D": claim_d["verdict"],
        "diagnostic_misses": claim_b["diagnostic_misses"],
        "must_answer_a_reaches_fits": must_answer_a["reaches_fits_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
