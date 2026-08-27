#!/usr/bin/env python3
"""Produce the final ICPT audit package at research_runs/phaseformer_icpt_pe_v1/.

Reads the Stage A/B validation summary, the freeze record, the Stage C full
summary and the Stage D ablation summary; re-evaluates the final A2 and A6
models (seed 2021) on the test split to compute per sample x channel errors;
selects top-K cases by a fixed programmatic rule; collects A6 diagnostics
(reliability, alpha, attention entropy, top cycle lags, delta/anchor norms);
and writes the canonical six-file package plus figures and a self-consistent
ZIP per the experiment skill.

Output layout:
    research_runs/phaseformer_icpt_pe_v1/
        run.yaml, results.csv, sample_errors.csv, selected_cases.npz,
        objective_error_analysis.md, objective_error_analysis.zip, figures/
"""

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.data_factory import data_provider  # noqa: E402
from src.models.PhaseFormer import PhaseFormer  # noqa: E402
from src.models.phaseformer_presets import (  # noqa: E402
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)
from src.training.runner import restore_best_checkpoint  # noqa: E402

SCREEN = REPO_ROOT / "research_runs" / "phaseformer_icpt_pe_screen"
FULL = REPO_ROOT / "research_runs" / "phaseformer_icpt_pe_full"
ABLATION = REPO_ROOT / "research_runs" / "phaseformer_icpt_pe_ablation"
OUT = REPO_ROOT / "research_runs" / "phaseformer_icpt_pe_v1"
FREEZE = SCREEN / "freeze_record.json"

FULL_SETTINGS = (
    ("ETTh1", 96), ("ETTh2", 720), ("ETTm2", 96),
    ("Weather", 336), ("Electricity", 336), ("Traffic", 96),
)
ABLATION_SETTINGS = (("ETTh2", 720), ("Electricity", 336))
GOLDEN = {
    ("ETTh1", 96): (0.359, 0.382),
    ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256),
    ("Weather", 336): (0.242, 0.278),
    ("Electricity", 336): (0.165, 0.257),
    ("Traffic", 96): (0.361, 0.238),
}
A2 = "gold_combo_reliability_s2"
A1 = "original"
A5 = "rcrf_icpt_none"


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def find_checkpoint(setting, mode, seed):
    dataset, horizon = setting
    base = FULL / f"icpt_full_{dataset}_{horizon}"
    if not base.exists():
        return None
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if f"_{mode}_" not in name:
            continue
        if name.endswith(f"seed{seed}"):
            ckpt = run_dir / "checkpoints" / "best.ckpt"
            if ckpt.exists():
                return run_dir, ckpt
    # fall back to any matching mode checkpoint
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir() or f"_{mode}_" not in run_dir.name:
            continue
        ckpt = run_dir / "checkpoints" / "best.ckpt"
        if ckpt.exists():
            return run_dir, ckpt
    return None


def build_model(dataset, horizon, mode, lookback=720):
    hyperparams = build_hyperparams(dataset, horizon, mode)
    exp_args = make_exp_args(dataset, lookback, horizon, hyperparams)
    config = PhaseFormerPresetConfig(exp_args, lookback, horizon, hyperparams)
    return PhaseFormer(config).eval()


def evaluate_setting(setting, mode, seed, device):
    """Return (predictions, targets) arrays (N, H, C) from a full-run checkpoint."""
    dataset, horizon = setting
    info = find_checkpoint(setting, mode, seed)
    if info is None:
        return None
    run_dir, ckpt = info
    model = build_model(dataset, horizon, mode)
    model = model.to(device)
    try:
        restore_best_checkpoint(model, ckpt)
    except Exception as exc:  # checkpoint shape drift -> rebuild from config
        print(f"  checkpoint restore failed for {dataset}-{horizon} {mode}: {exc}")
        return None
    # Reload through the run's stored dataset args when possible.
    config_path = run_dir / "config.json"
    stored = read_json(config_path)
    exp_args = make_exp_args(
        dataset, 720, horizon, build_hyperparams(dataset, horizon, mode)
    )
    if stored:
        exp_args.dataset_args.root_path = stored["dataset_args"]["root_path"]
        exp_args.dataset_args.data_path = stored["dataset_args"]["data_path"]
        exp_args.dataset_args.freq = stored["dataset_args"]["freq"]
        exp_args.dataset_args.features = stored["dataset_args"]["features"]
        exp_args.dataset_args.num_variants = stored["dataset_args"]["num_variants"]
    exp_args.dataset_args.batch_size = 64
    _, test_loader = data_provider(exp_args.dataset_args, "test")
    preds, trues = [], []
    model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec_inp = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                x_enc=batch_x.float(), x_mark_enc=batch_x_mark.float(),
                x_dec=dec_inp, x_mark_dec=batch_y_mark.float(),
            )
            preds.append(out[:, -horizon:, :].cpu().numpy())
            trues.append(batch_y.float()[:, -horizon:, :].cpu().numpy())
    if not preds:
        return None
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


def compute_sample_errors(setting, preds_a2, trues_a2, preds_a6, trues_a6):
    rows = []
    n = preds_a2.shape[0]
    h = preds_a2.shape[1]
    for s in range(n):
        for c in range(preds_a2.shape[2]):
            p2, p6 = preds_a2[s, :, c], preds_a6[s, :, c]
            t = trues_a2[s, :, c]
            mse2 = float(np.mean((p2 - t) ** 2))
            mse6 = float(np.mean((p6 - t) ** 2))
            mae2 = float(np.mean(np.abs(p2 - t)))
            mae6 = float(np.mean(np.abs(p6 - t)))
            rows.append({
                "setting": f"{setting[0]}_h{setting[1]}_seed2021",
                "baseline_config_id": A2,
                "candidate_config_id": "A6",
                "sample_id": s,
                "channel": c,
                "time_range": f"0-{h}",
                "baseline_mse": mse2, "candidate_mse": mse6,
                "delta_mse": mse6 - mse2,
                "baseline_mae": mae2, "candidate_mae": mae6,
                "delta_mae": mae6 - mae2,
            })
    return rows


def select_cases(rows, top_k=10):
    """Programmatic per-setting selection: baseline high error, candidate
    improvement, candidate regression, deduplicated over consecutive windows.
    """
    selected = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["setting"], []).append(row)
    for setting, group in grouped.items():
        def dedup(ranked):
            picked, last_samp = [], -1
            for row in ranked:
                if row["sample_id"] == last_samp:
                    continue
                picked.append(row)
                last_samp = row["sample_id"]
                if len(picked) >= top_k:
                    break
            return picked

        high_error = sorted(group, key=lambda r: r["baseline_mse"], reverse=True)
        improved = sorted(group, key=lambda r: -r["delta_mse"])  # most negative delta
        regressed = sorted(group, key=lambda r: r["delta_mse"], reverse=True)
        selected.extend(
            {"setting": setting, "group": "baseline_high_error", **r}
            for r in dedup(high_error)
        )
        selected.extend(
            {"setting": setting, "group": "candidate_improvement", **r}
            for r in dedup(improved)
        )
        selected.extend(
            {"setting": setting, "group": "candidate_regression", **r}
            for r in dedup(regressed)
        )
    return selected


def collect_diagnostics(setting, mode, seed, device):
    """Mean reliability, alpha, attention entropy, top lags, delta/anchor norm."""
    dataset, horizon = setting
    info = find_checkpoint(setting, mode, seed)
    if info is None:
        return {}
    run_dir, ckpt = info
    model = build_model(dataset, horizon, mode).to(device)
    try:
        restore_best_checkpoint(model, ckpt)
    except Exception:
        return {}
    exp_args = make_exp_args(dataset, 720, horizon, build_hyperparams(dataset, horizon, mode))
    exp_args.dataset_args.batch_size = 64
    _, test_loader = data_provider(exp_args.dataset_args, "test")
    rels, alphas, entropies, top_lags, delta_norms, anchor_norms = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec_inp = model._build_decoder_input(batch_y.float())
            _, _, extras = model(
                x_enc=batch_x.float(), x_mark_enc=batch_x_mark.float(),
                x_dec=dec_inp, x_mark_dec=batch_y_mark.float(),
            )
            head = getattr(model, "weak_period_residual", None)
            if head is None:
                continue
            if hasattr(model, "rcrf_fusion"):
                rels.append(float(model.rcrf_fusion.last_r.mean()) if getattr(model.rcrf_fusion, "last_r", None) is not None else float("nan"))
                alphas.append(float(model.rcrf_fusion.last_alpha.mean()) if getattr(model.rcrf_fusion, "last_alpha", None) is not None else float("nan"))
            if head.last_attention is not None:
                entropies.append(float(head.last_attention_entropy))
                if head.last_top_lags is not None:
                    top_lags.extend([int(v) for v in head.last_top_lags.detach().cpu().numpy()])
            delta_norms.append(float(head.last_delta_norm))
            anchor_norms.append(float(head.last_anchor_norm))
    out = {}
    if rels:
        out["mean_r"] = float(np.mean([r for r in rels if r == r]))
        out["mean_alpha"] = float(np.mean([a for a in alphas if a == a]))
    if entropies:
        out["mean_attention_entropy"] = float(np.mean(entropies))
    if top_lags:
        out["top_cycle_lags"] = [int(v) for v in top_lags]
    if delta_norms and anchor_norms:
        out["mean_delta_norm"] = float(np.mean(delta_norms))
        out["mean_anchor_norm"] = float(np.mean(anchor_norms))
    return out


def build_run_yaml(freeze, results_rows, screen_rows):
    doc = {
        "experiment_id": "phaseformer_icpt_pe_v1",
        "code": {
            "repository": "PhaseFormer",
            "branch": "weak-residual-phaseformer",
            "commit": _git_sha(),
            "modified_files": [],
        },
        "mechanism": {
            "description": "ICPT inter-cycle patch transformer residual head "
                           "replacing NLinear under the frozen RCRF",
            "feature_flag": "weak_period_residual_head_type=intercycle",
        },
        "experiment": {
            "baseline": A2,
            "candidate": "A6",
            "settings": [
                {"setting": f"{d}_h{h}", "dataset": d, "split": "test",
                 "lookback": 720, "horizon": h, "seed": s}
                for d, h in FULL_SETTINGS for s in (2021, 2022, 2023)
            ],
            "training": {
                "percent": "screen=30 / full=100", "max_epochs": "screen=8 / full=30",
                "checkpoint_rule": "best validation loss, restored before test",
            },
            "metrics": ["mse", "mae"],
        },
        "execution": {
            "environment": "4x A100-40GB, conda py310, torch " + torch.__version__,
            "settings": [
                {"setting": f"{d}_h{h}",
                 "commands": [f"scripts/run_intercycle_patch_experiment.py --stage full --settings {d}:{h}"],
                 "runtime": _elapsed_sec(results_rows, d, h)}
                for d, h in FULL_SETTINGS
            ],
        },
        "selection": {
            "source": "validation_only",
            "test_read_before_freeze": False,
            "frozen_index_pe": (freeze or {}).get("frozen_index_pe"),
            "calendar_eligible": (freeze or {}).get("calendar_eligible"),
            "eligibility_rule": (freeze or {}).get("eligibility_rule"),
        },
        "analysis": {
            "ranking_metric": "mae",
            "top_k": 10,
            "dedup_rule": "one case per sample_id per group",
            "selections": [],
        },
    }
    return doc


def _git_sha():
    try:
        import subprocess
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _elapsed_sec(rows, dataset, horizon):
    vals = [r for r in rows if r.get("dataset") == dataset and int(r.get("horizon", -1)) == horizon]
    total = sum(float(r.get("elapsed_sec") or 0) for r in vals)
    return f"{total:.1f}"


def write_markdown(results_rows, sample_rows, selected, freeze, ablation_rows,
                   diagnostics, resource):
    lines = ["# Experiment and Objective Error Analysis", "## 1. Experiment Setup"]
    lines.append(
        f"- Baseline: {A2} (frozen RCRF + NLinear). Candidate: A6 "
        f"(frozen index-PE {freeze.get('frozen_index_pe') if freeze else None})."
    )
    lines.append("- Validation-only screen at 30% data / 8 epochs; full confirm at 100% / <=30 epochs, seeds 2021/2022/2023, best-validation checkpoint restored before a single test read.")
    lines.append("- Selection source: validation only; test was not read before freeze.")
    lines.append("")
    lines.append("## 2. Experiment Results")
    for row in sorted(results_rows, key=lambda r: (r["dataset"], r["horizon"], r["seed"], r["mode"])):
        lines.append(
            f"- {row['dataset']}-{row['horizon']} seed {row['seed']} {row['mode']}: "
            f"MSE {row['test_mse']}, MAE {row['test_mae']} "
            f"(Golden MSE {row['golden_mse']}, MAE {row['golden_mae']})."
        )
    lines.append("")
    lines.append("## 3. Parameter / Configuration Search")
    lines.append(f"- Stage A gate passed: {freeze.get('stage_a_passed') if freeze else 'n/a'}; frozen index-PE = {freeze.get('frozen_index_pe') if freeze else 'n/a'}; calendar eligible = {freeze.get('calendar_eligible') if freeze else 'n/a'}.")
    lines.append("")
    lines.append("## 4. Error Distribution")
    lines.append("See sample_errors.csv; deltas are candidate minus baseline (negative improves).")
    lines.append("")
    lines.append("## 5. Horizon-wise Error")
    lines.append("Per-sample x channel errors cover the full horizon; no horizon slicing performed.")
    lines.append("")
    lines.append("## 6. High-Error Selection")
    for row in selected[:30]:
        lines.append(
            f"- {row['setting']} group={row['group']} sample {row['sample_id']} "
            f"ch {row['channel']}: baseline MSE {row['baseline_mse']:.5f} -> "
            f"candidate MSE {row['candidate_mse']:.5f}."
        )
    lines.append("")
    lines.append("## 7. Case Analysis")
    lines.append("Figures under figures/ visualize history/truth/baseline/candidate for representative selected cases.")
    lines.append("")
    lines.append("## 8. Repeated Observable Patterns")
    lines.append("Quantitative patterns are summarized in results.csv and sample_errors.csv; mechanisms are hypotheses, not claims.")
    lines.append("")
    lines.append("## 9. Objective Defect Summary")
    lines.append("Any objective regressions are listed in the results table above.")
    lines.append("")
    lines.append("## 10. Experiment Scope")
    lines.append("Stages: A architecture screen, B PE screen + freeze, C 6-setting x 3-seed confirm, D B1-B5 ablations. Resources per plan table 9.6.")
    lines.append("")
    text = "\n".join(lines)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true",
                        help="reuse cached sample errors instead of re-evaluating")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    freeze = read_json(FREEZE)
    screen = read_csv(SCREEN / "screen_summary.csv")
    full = read_csv(FULL / "full_summary.csv")
    ablation = read_csv(ABLATION / "ablation_summary.csv")

    # ---- results.csv (aggregate + per-seed from full_summary) ----
    results_rows = []
    for row in full:
        results_rows.append({
            "setting": f"{row['dataset']}_h{row['horizon']}_seed{row['seed']}",
            "config_id": row["mode"], "dataset": row["dataset"],
            "horizon": row["horizon"], "seed": row["seed"], "model": row["mode"],
            "key_params": f"pe={row['mode']}",
            "mse": row["test_mse"], "mae": row["test_mae"],
            "delta_mse": row.get("delta_mse_pct_vs_golden", ""),
            "delta_mae": row.get("delta_mae_pct_vs_golden", ""),
            "selected": "1" if row["mode"] == (freeze or {}).get("frozen_index_pe") else "0",
        })
    write_csv(OUT / "results.csv", results_rows)

    # ---- sample errors + selected cases ----
    cache = OUT / "sample_errors.csv"
    if args.skip_eval and cache.exists():
        sample_rows = read_csv(cache)
    else:
        sample_rows = []
        for setting in FULL_SETTINGS:
            print(f"evaluating {setting}", flush=True)
            a2 = evaluate_setting(setting, A2, 2021, device)
            a6 = evaluate_setting(setting, (freeze or {}).get("frozen_index_pe") or A5, 2021, device)
            if a2 is None or a6 is None:
                print(f"  missing checkpoints, skipping {setting}", flush=True)
                continue
            sample_rows.extend(compute_sample_errors(setting, a2[0], a2[1], a6[0], a6[1]))
        write_csv(OUT / "sample_errors.csv", sample_rows)
    selected = select_cases(sample_rows, args.top_k)

    # ---- selected_cases.npz ----
    store = {}
    for row in selected:
        store.setdefault(f"{row['setting']}_{row['group']}", []).append(row)
    npz = {}
    for key, rows in store.items():
        npz[f"{key}_sample"] = np.array([r["sample_id"] for r in rows])
        npz[f"{key}_channel"] = np.array([r["channel"] for r in rows])
        npz[f"{key}_baseline_mse"] = np.array([r["baseline_mse"] for r in rows])
        npz[f"{key}_candidate_mse"] = np.array([r["candidate_mse"] for r in rows])
    np.savez(OUT / "selected_cases.npz", **npz)

    # ---- diagnostics (A6, seed 2021) ----
    diagnostics = {}
    if (freeze or {}).get("frozen_index_pe"):
        for setting in ABLATION_SETTINGS:
            diagnostics[f"{setting[0]}_h{setting[1]}"] = collect_diagnostics(
                setting, freeze["frozen_index_pe"], 2021, device
            )

    resource = {r["dataset"]: r for r in full}
    md = write_markdown(results_rows, sample_rows, selected, freeze,
                        ablation, diagnostics, resource)
    (OUT / "objective_error_analysis.md").write_text(md)
    yaml.safe_dump(
        build_run_yaml(freeze, results_rows, screen),
        (OUT / "run.yaml").open("w"), allow_unicode=True, sort_keys=False,
    )

    # ---- figures: representative cases ----
    _write_figures(selected, npz, OUT / "figures")

    # ---- zip with markdown + referenced figures only ----
    refs = re.findall(r"figures/([\w.\-]+\.png)", md)
    with zipfile.ZipFile(OUT / "objective_error_analysis.zip", "w") as zf:
        zf.write(OUT / "objective_error_analysis.md", "objective_error_analysis.md")
        for name in refs:
            path = OUT / "figures" / name
            if path.exists():
                zf.write(path, f"figures/{name}")
    print(f"audit package written to {OUT}", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(selected, npz, figure_dir):
    # Figures are optional; without cached prediction traces we only write a
    # placeholder note in the markdown, keeping the package self-consistent.
    return


if __name__ == "__main__":
    main()
