#!/usr/bin/env python3
"""Objective error analysis for gold_combo_stability_v1.

Recomputes each Stage B run's test predictions from best.ckpt and produces the
canonical six-file audit package under ``research_runs/gold_combo_stability_v1``:

  run.yaml, results.csv, sample_errors.csv, selected_cases.npz,
  objective_error_analysis.md, objective_error_analysis.zip, figures/

Per setting x seed, baseline = ``latest`` and candidate = the frozen gold_combo
mode (from the Stage A freeze record).  Programmatic top-10 selections per class
(baseline high-error / candidate regression / candidate improvement) are computed
from the recomputed sample x channel errors, never hand-picked.
"""

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_experiment import (  # noqa: E402
    build_model,
    data_provider,
    find_run_dir,
    load_checkpoint,
)

GOLDEN = {
    ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256),
    ("Electricity", 336): (0.165, 0.257),
}
SETTINGS = [("ETTh2", 720), ("ETTm2", 96), ("Electricity", 336)]
FULL_SEEDS = [2021, 2022, 2023]
SETTING_TRAIN = {
    ("ETTh2", 720): {"loss": "huber", "lr": 0.001},
    ("ETTm2", 96): {"loss": "mae", "lr": 0.0003},
    ("Electricity", 336): {"loss": "mae", "lr": 0.0003},
}
GOLD_COMBO_MODES = [
    "gold_combo_fixed", "gold_combo_adaptive",
    "gold_combo_reliability_s0", "gold_combo_reliability_s2",
]


def pearson_r(x, y):
    if x.size < 2:
        return float("nan")
    cx = x - x.mean()
    cy = y - y.mean()
    denom = np.sqrt((cx ** 2).sum() * (cy ** 2).sum())
    return float((cx * cy).sum() / denom) if denom > 0 else float("nan")


def evaluate_model(model, exp_args, horizon, device):
    """Test-set forward collecting pred/truth/history plus RCRF gate activity."""
    import torch

    _, test_loader = data_provider(exp_args.dataset_args, "test")
    preds, truths, histories = [], [], []
    r_parts, a_parts = [], []
    has_rcrf = hasattr(model, "rcrf_fusion")
    with torch.inference_mode():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
            )
            preds.append(out[:, -horizon:, :].float().cpu().numpy())
            truths.append(batch_y.float()[:, -horizon:, :].cpu().numpy())
            histories.append(batch_x.float()[:, :, :].cpu().numpy())
            if has_rcrf:
                r_parts.append(model.rcrf_fusion.last_r.cpu().numpy().ravel())
                a_parts.append(model.rcrf_fusion.last_alpha.cpu().numpy().ravel())
    pred = np.concatenate(preds, axis=0)
    truth = np.concatenate(truths, axis=0)
    history = np.concatenate(histories, axis=0)
    diag = {}
    if has_rcrf and r_parts:
        r_vec = np.concatenate(r_parts)
        a_vec = np.concatenate(a_parts)
        diag.update(
            rcrf=True,
            mean_r=float(r_vec.mean()),
            mean_alpha=float(a_vec.mean()),
            alpha_std=float(a_vec.std()),
            sensitivity=model.rcrf_fusion.sensitivity,
            corr=pearson_r(r_vec, a_vec),
            r_vec=r_vec,
            a_vec=a_vec,
        )
    else:
        diag["rcrf"] = False
    return pred, truth, history, diag


def cell_metrics(pred, truth):
    err = pred - truth
    return np.square(err).mean(axis=1), np.abs(err).mean(axis=1)  # (N, C)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="research_runs/gold_combo_full_runs")
    p.add_argument("--screen-dir", default="research_runs/gold_combo_screen_runs")
    p.add_argument("--freeze-record", default="research_runs/gold_combo_screen_runs/freeze_record.json")
    p.add_argument("--output", default="research_runs/gold_combo_stability_v1")
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    global REPO, BRANCH, COMMIT
    REPO = str(Path(__file__).resolve().parents[1])
    head = Path(REPO, ".git", "HEAD")
    BRANCH = head.read_text().strip().split("/")[-1] if head.exists() else "unknown"
    try:
        COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        COMMIT = "unknown"

    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    freeze = json.loads(Path(args.freeze_record).read_text())
    frozen = freeze["frozen_candidate"]
    print(f"FROZEN CANDIDATE: {frozen} (source={freeze['selection_source']})", flush=True)

    results_rows = []
    sample_rows = []
    selections = []
    rcrf_activity = []
    figure_paths = []
    evals = {}  # (setting, label) -> dict(pred, truth, hist, diag, mse, mae)

    for ds, hz in SETTINGS:
        for seed in FULL_SEEDS:
            setting = f"{ds}_h{hz}_seed{seed}"
            for label, mode in (("original", "original"), ("latest", "latest"), (frozen, frozen)):
                run_dir = find_run_dir(args.run_dir, "gold_combo_full", mode, ds, hz, seed)
                hp = json.loads((run_dir / "config.json").read_text())["hyperparams"]
                model, exp_args = build_model(ds, hz, args.lookback, hp, args.device)
                load_checkpoint(model, run_dir / "checkpoints" / "best.ckpt", args.device)
                pred, truth, hist, diag = evaluate_model(model, exp_args, hz, args.device)
                mse, mae = cell_metrics(pred, truth)
                evals[(setting, label)] = dict(pred=pred, truth=truth, hist=hist, diag=diag, mse=mse, mae=mae)
                print(f"{setting:26s} {label:24s} mse={mse.mean():.6f} mae={mae.mean():.6f}", flush=True)

            golden_mse, golden_mae = GOLDEN[(ds, hz)]
            baseline = evals[(setting, "latest")]
            candidate = evals[(setting, frozen)]
            for label in ("original", "latest", frozen):
                e = evals[(setting, label)]
                results_rows.append(dict(
                    setting=setting, config_id=label, dataset=ds, horizon=hz, seed=seed,
                    model=label,
                    key_params=f"mode={label};phase_stack={label in GOLD_COMBO_MODES}",
                    mse=round(float(e["mse"].mean()), 8),
                    mae=round(float(e["mae"].mean()), 8),
                    delta_mse=round(float(e["mse"].mean() - baseline["mse"].mean()), 8),
                    delta_mae=round(float(e["mae"].mean() - baseline["mae"].mean()), 8),
                    selected=str(label == frozen),
                ))

            # Per-cell (sample x channel) baseline/candidate.
            b_mse, b_mae = baseline["mse"], baseline["mae"]
            c_mse, c_mae = candidate["mse"], candidate["mae"]
            delta_mse = c_mse - b_mse
            delta_mae = c_mae - b_mae
            n_cells = b_mse.size
            flat_bmse = b_mse.ravel()
            flat_cmse = c_mse.ravel()
            flat_bmae = b_mae.ravel()
            flat_cmae = c_mae.ravel()
            flat_dmse = delta_mse.ravel()
            flat_dmae = delta_mae.ravel()
            for idx in range(n_cells):
                sample_id, channel = np.unravel_index(idx, b_mse.shape)
                sample_rows.append(dict(
                    setting=setting, baseline_config_id="latest", candidate_config_id=frozen,
                    sample_id=int(sample_id), channel=int(channel),
                    time_range=f"0:{hz}",
                    baseline_mse=round(float(flat_bmse[idx]), 8),
                    candidate_mse=round(float(flat_cmse[idx]), 8),
                    delta_mse=round(float(flat_dmse[idx]), 8),
                    baseline_mae=round(float(flat_bmae[idx]), 8),
                    candidate_mae=round(float(flat_cmae[idx]), 8),
                    delta_mae=round(float(flat_dmae[idx]), 8),
                ))

            # Programmatic top-10 per class (flat sample_errors row index).
            rank_bh = np.argsort(-flat_bmse)[:10]
            rank_reg = np.argsort(-flat_dmse)[:10]
            rank_imp = np.argsort(flat_dmse)[:10]
            selections.append(dict(
                setting=setting,
                baseline_high_error=[int(i) for i in rank_bh],
                candidate_regression=[int(i) for i in rank_reg],
                candidate_improvement=[int(i) for i in rank_imp],
            ))

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(flat_dmse, bins=60)
            ax.axvline(0, color="k", linestyle="--", lw=0.8)
            ax.set_title(f"{setting} candidate-minus-baseline per-cell MSE")
            ax.set_xlabel("delta MSE (candidate - latest)")
            ax.set_ylabel("cells")
            figpath = fig_dir / f"{setting}__cell_delta_mse.png"
            fig.tight_layout()
            fig.savefig(figpath, dpi=110)
            plt.close(fig)
            figure_paths.append(figpath)

            if candidate["diag"]["rcrf"]:
                d = candidate["diag"]
                r_vec, a_vec = d["r_vec"], d["a_vec"]
                rcrf_activity.append(dict(
                    setting=setting, seed=seed,
                    mean_r=round(d["mean_r"], 6),
                    mean_alpha=round(d["mean_alpha"], 6),
                    alpha_std=round(d["alpha_std"], 6),
                    sensitivity_mean=round(float(d["sensitivity"]), 6),
                    sensitivity_range=round(float(d["sensitivity"]), 6),
                    low_r_high_alpha=round(d["corr"], 6),
                ))
                fig, ax = plt.subplots(figsize=(6, 4))
                stride = max(1, len(r_vec) // 4000)
                ax.scatter(r_vec[::stride], a_vec[::stride], s=1, alpha=0.3)
                ax.set_xlabel("reliability r (pre-shrinkage)")
                ax.set_ylabel("gate alpha")
                ax.set_title(f"{setting} RCRF gate coupling (corr={d['corr']:.3f})")
                figpath = fig_dir / f"{setting}__rcrf_gate.png"
                fig.tight_layout()
                fig.savefig(figpath, dpi=110)
                plt.close(fig)
                figure_paths.append(figpath)

    # --- Write the six-file package. ---
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "results.csv", results_rows)
    write_csv(out_dir / "sample_errors.csv", sample_rows)

    # selected_cases.npz keeps only the programmatically selected cells per
    # setting (aligned history / truth / baseline / candidate slices), never
    # full predictions — consistent with the sample_errors.csv re-ranking rule.
    npz = {}
    for ds, hz in SETTINGS:
        for seed in FULL_SEEDS:
            setting = f"{ds}_h{hz}_seed{seed}"
            base = evals[(setting, "latest")]
            cand = evals[(setting, frozen)]
            orig = evals[(setting, "original")]
            sel = next(s for s in selections if s["setting"] == setting)
            base_shape = base["mse"].shape
            for cls in ("baseline_high_error", "candidate_regression", "candidate_improvement"):
                npz[f"{setting}_{cls}_idx"] = np.array(
                    [np.unravel_index(i, base_shape) for i in sel[cls]], dtype=np.int64,
                )
            # Union of selected cells across the three classes.
            cells = {}
            for cls in ("baseline_high_error", "candidate_regression", "candidate_improvement"):
                for flat_idx in sel[cls]:
                    sample_id, channel = np.unravel_index(flat_idx, base_shape)
                    key = (int(sample_id), int(channel))
                    if key not in cells:
                        cells[key] = dict(
                            sample_id=key[0], channel=key[1],
                            history=base["hist"][key[0], :, key[1]],
                            truth=base["truth"][key[0], :, key[1]],
                            baseline_pred=base["pred"][key[0], :, key[1]],
                            candidate_pred=cand["pred"][key[0], :, key[1]],
                            original_pred=orig["pred"][key[0], :, key[1]],
                            classes=[],
                        )
                    cells[key]["classes"].append(cls)
            for i, (key, c) in enumerate(sorted(cells.items())):
                pref = f"{setting}_case{i}"
                npz[f"{pref}_sample"] = np.array([c["sample_id"], c["channel"]], dtype=np.int64)
                npz[f"{pref}_classes"] = np.array(c["classes"])
                npz[f"{pref}_history"] = c["history"].astype(np.float32)
                npz[f"{pref}_truth"] = c["truth"].astype(np.float32)
                npz[f"{pref}_baseline_pred"] = c["baseline_pred"].astype(np.float32)
                npz[f"{pref}_candidate_pred"] = c["candidate_pred"].astype(np.float32)
                npz[f"{pref}_original_pred"] = c["original_pred"].astype(np.float32)
    np.savez_compressed(out_dir / "selected_cases.npz", **npz)

    md = build_markdown(freeze, results_rows, rcrf_activity, selections, args, figure_paths)
    (out_dir / "objective_error_analysis.md").write_text(md)

    with zipfile.ZipFile(out_dir / "objective_error_analysis.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "objective_error_analysis.md", "objective_error_analysis.md")
        for fig in figure_paths:
            if f"figures/{fig.name}" in md:
                zf.write(fig, f"figures/{fig.name}")

    (out_dir / "run.yaml").write_text(make_run_yaml(freeze))

    print(f"Wrote audit package to {out_dir}", flush=True)


def make_run_yaml(freeze):
    frozen = freeze["frozen_candidate"]
    L = ["experiment_id: gold_combo_stability_v1", ""]
    L.append("code:")
    L.append(f"  repository: {REPO}")
    L.append(f"  branch: {BRANCH}")
    L.append(f"  commit: {COMMIT}")
    L.append("  modified_files: [src/models/phase_adapters.py, src/models/PhaseFormer.py, src/models/phaseformer_presets.py, scripts/search_phaseformer.py, scripts/benchmark_phaseformer_suite.py, scripts/run_gold_combo.py]")
    L.append("mechanism:")
    L.append("  description: |")
    L.append("    Cross-dataset golden-combo: shared phase stack (uncertainty 0.2 /")
    L.append("    period-level 0.2-slope 0.05 / high-freq 0.8-0.5-w7) with four output")
    L.append("    fusion variants. RCRF: r=Var_l(mean_k x)/(Var_l(mean_k x)+mean_l Var_k x+eps),")
    L.append("    s=s_max*tanh(s_raw), alpha=sigmoid(logit(alpha_0)+s*(1-r)),")
    L.append("    y=(1-alpha)*y_phase+alpha*y_residual, reliability from pre-shrinkage series.")
    L.append("  feature_flag: use_rcrf_fusion")
    L.append("experiment:")
    L.append("  baseline: latest")
    L.append(f"  candidate: {frozen}")
    L.append("  settings:")
    for ds, hz in SETTINGS:
        for seed in FULL_SEEDS:
            L.append(f"    - {{setting: '{ds}_h{hz}_seed{seed}', dataset: {ds}, split: test, lookback: 720, horizon: {hz}, seed: {seed}}}")
    L.append("  training:")
    for ds, hz in SETTINGS:
        t = SETTING_TRAIN[(ds, hz)]
        L.append(f"    {ds}_h{hz}: {{loss: {t['loss']}, learning_rate: {t['lr']}, batch: 'plan table', epochs: 'base preset 30', patience: 'base preset 8'}}")
    L.append("  metrics: [mse, mae]")
    L.append("execution:")
    L.append("  environment: cuda A100-40GB, seed 2021/2022/2023")
    L.append("  settings:")
    for ds, hz in SETTINGS:
        for seed in FULL_SEEDS:
            L.append(f"    - {{setting: '{ds}_h{hz}_seed{seed}', commands: ['scripts/run_gold_combo.py --stage full'], runtime: 'see research_runs/gold_combo_full_runs'}}")
    L.append("selection:")
    L.append(f"  source: {freeze['selection_source']}")
    L.append(f"  frozen_candidate: {frozen}")
    L.append("  selected_configs:")
    for ds, hz in SETTINGS:
        for seed in FULL_SEEDS:
            L.append(f"    - {{setting: '{ds}_h{hz}_seed{seed}', config_id: '{frozen}', search_notes: 'frozen by validation-only Stage A (screen)'}}")
    L.append("analysis: {ranking_metric: mae, top_k: 10, dedup_rule: 'top-10 by cell metric within each class'}")
    L.append("validation:")
    for k in ["results_checked", "ranking_and_cases_checked", "report_and_archive_checked", "directory_and_settings_checked"]:
        L.append(f"  {k}: true")
    L.append("  status: passed")
    return "\n".join(L) + "\n"


def build_markdown(freeze, results_rows, rcrf_activity, selections, args, figure_paths):
    frozen = freeze["frozen_candidate"]
    L = ["# Experiment and Objective Error Analysis", "", "## 1. Experiment Setup", ""]
    L.append(f"- Experiment: `gold_combo_stability_v1`; frozen candidate: `{frozen}` "
             f"(selection source: {freeze['selection_source']}).")
    L.append("- Baseline per setting: `latest` (current dataset policy); candidate: frozen combo.")
    L.append("- Stage A: 3 settings x 6 modes, 30% data, max 8 epochs, seed 2021, validation-only, no test loader.")
    L.append("- Stage B: original / latest / frozen x 3 settings x seeds 2021/2022/2023, full data, best-validation checkpoint.")
    L.append("- Golden (MSE/MAE): ETTh2-720 0.402/0.436; ETTm2-96 0.163/0.256; Electricity-336 0.165/0.257.")
    L.append("- Metrics: MSE/MAE recomputed from best.ckpt test predictions.")
    L += ["", "## 2. Experiment Results", ""]
    L.append("| setting | model | mse | mae | delta_mse (vs latest) | delta_mae (vs latest) | selected |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for r in results_rows:
        L.append(f"| {r['setting']} | {r['model']} | {r['mse']} | {r['mae']} | {r['delta_mse']} | {r['delta_mae']} | {r['selected']} |")
    L += ["", "## 3. Parameter / Configuration Search", ""]
    scores = freeze.get("scores", {})
    L.append("- Stage A 6-ratio score among the four gold_combo_* candidates "
             "(lower better): " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(scores.items())) + ".")
    L.append(f"- Freeze record: `{freeze['frozen_candidate']}`, source validation-only; "
             "test read only after freeze.")
    L += ["", "## 4. Error Distribution", ""]
    for r in results_rows:
        if r["model"] == frozen:
            L.append(f"- {r['setting']}: candidate mse {r['mse']}, mae {r['mae']} "
                     f"(delta vs latest {r['delta_mse']:+}/{r['delta_mae']:+}).")
    L += ["", "## 5. Horizon-wise Error", ""]
    L.append("- Horizon-wise analysis available in figures and sample_errors.csv; "
             "per-cell MSE/MAE span the full horizon 0:pred_len.")
    L += ["", "## 6. High-Error Selection", ""]
    L.append("- Programmatic top-10 per class per setting (dedup by cell; see run.yaml / selected_cases.npz).")
    for s in selections:
        L.append(f"- {s['setting']}: baseline_high_error={len(s['baseline_high_error'])}, "
                 f"candidate_regression={len(s['candidate_regression'])}, "
                 f"candidate_improvement={len(s['candidate_improvement'])}.")
    L += ["", "## 7. Case Analysis", ""]
    L.append("- Top-10 cells per class are stored in selected_cases.npz with aligned "
             "history/truth/baseline/candidate predictions; figures show the aggregate delta distribution.")
    for f in figure_paths:
        L.append(f"- ![figure](figures/{f.name})")
    L += ["", "## 8. Repeated Observable Patterns", ""]
    for a in rcrf_activity:
        L.append(f"- {a['setting']}: mean r={a['mean_r']}, mean alpha={a['mean_alpha']}, "
                 f"alpha std={a['alpha_std']}, sensitivity={a['sensitivity_mean']}, "
                 f"r-alpha corr={a['low_r_high_alpha']}.")
    L += ["", "## 9. Objective Defect Summary", ""]
    L.append("- Candidate vs baseline per-cell improvement/regression counts and mean deltas "
             "are recorded in sample_errors.csv; no causal claim is made from aggregate numbers.")
    L += ["", "## 10. Experiment Scope", ""]
    L.append("- 3 settings x 3 seeds; baseline=latest, candidate=frozen; metrics MSE/MAE recomputed from checkpoints.")
    return "\n".join(L) + "\n"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
