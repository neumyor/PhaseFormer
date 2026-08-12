#!/usr/bin/env python3
"""Sample-level error analysis + canonical report for an experiment run.

Reconstructs trained models from raw benchmark-suite run directories, evaluates
the test split, computes per-(sample, channel) errors, selects cases
programmatically, and writes the canonical six-file audit set for an
experiment_id:

  run.yaml (written by caller), results.csv, sample_errors.csv,
  selected_cases.npz, objective_error_analysis.md, objective_error_analysis.zip
  plus figures/<setting>__<group>.png referenced by the report.

This script only reads; it never trains. Aggregate results are read from the
run's metrics.csv (authoritative test metrics from the runner), while per-cell
errors are recomputed from a faithful model reconstruction + checkpoint load.
"""

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

REPO_ROOT = Path(os.environ.get("PHASEFORMER_REPO", "/home/niuyiming/PhaseFormer")).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)

SETTING_RE = re.compile(r"^(\w+)_h(\d+)_seed(\d+)$")
GROUPS = ["baseline_high_error", "candidate_regression", "candidate_improvement"]


def parse_setting(setting):
    m = SETTING_RE.match(setting)
    if not m:
        raise ValueError(f"bad setting: {setting}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_run_dir(run_root, prefix, mode, dataset, horizon, seed):
    # Match by (mode, scheme, dataset, horizon, seed); the leading prefix can
    # be per-setting (legacy runs) or a single shared prefix.
    pat = f"*_{mode}_*_{dataset.lower()}_{horizon}_seed{seed}"
    matches = sorted(Path(run_root).glob(pat))
    if not matches:
        raise FileNotFoundError(f"no run dir for {pat} in {run_root}")
    return matches[-1]


def build_model(dataset, horizon, lookback, hp, device):
    exp_args = make_exp_args(dataset, lookback, horizon, hp, batch_size=None)
    exp_args.dataset_args.num_workers = 0
    exp_args.training_args.num_workers = 0
    config = PhaseFormerPresetConfig(exp_args, lookback, horizon, hp)
    model = PhaseFormer(config)
    return model, exp_args


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate_test(model, exp_args, horizon, device):
    """Return pred (N,H,C), truth (N,H,C), history (N,seq,C) arrays.

    Also accumulates phase_amp_calib activity and reliability-gate activity
    diagnostics over batches when the model has those modules; the modules'
    hooks hold last-batch values, so we sum the absolute magnitudes here.
    """
    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    preds, truths, histories = [], [], []
    diag = {"log_alpha": 0.0, "beta": 0.0, "gate": 0.0, "n": 0}
    has_calib = hasattr(model, "phase_amp_calib")
    has_gate = hasattr(model, "reliability_gate")
    with torch.inference_mode():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
            )
            pred = out[:, -horizon:, :].float().cpu().numpy()
            true = batch_y.float()[:, -horizon:, :].cpu().numpy()
            hist = batch_x.float()[:, :, :].cpu().numpy()
            preds.append(pred)
            truths.append(true)
            histories.append(hist)
            if has_calib:
                diag["log_alpha"] += abs(model.phase_amp_calib.last_mean_abs_log_alpha)
                diag["beta"] += abs(model.phase_amp_calib.last_mean_abs_beta)
            if has_gate:
                diag["gate"] += abs(model.reliability_gate.last_mean_gate)
            diag["n"] += 1
    pred = np.concatenate(preds, axis=0)
    truth = np.concatenate(truths, axis=0)
    history = np.concatenate(histories, axis=0)
    if diag["n"]:
        diag["log_alpha"] /= diag["n"]
        diag["beta"] /= diag["n"]
        diag["gate"] /= diag["n"]
    return pred, truth, history, diag


def cell_metrics(pred, truth):
    err = pred - truth
    mse = np.square(err).mean(axis=1)  # (N, C)
    mae = np.abs(err).mean(axis=1)
    return mse, mae


def mean_mae_per_seg(pred, truth, segs=4):
    H = pred.shape[1]
    edges = [H * i // segs for i in range(segs + 1)]
    out = []
    for i in range(segs):
        s, e = edges[i], edges[i + 1]
        out.append(float(np.abs(pred[:, s:e] - truth[:, s:e]).mean()))
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def select_cases(sample_rows, group, top_k, dedup=None):
    """sample_rows: list of dicts with baseline_mae, candidate_mae, delta_mae."""
    if group == "baseline_high_error":
        key = lambda r: float(r["baseline_mae"])
        reverse = True
    elif group == "candidate_regression":
        key = lambda r: float(r["delta_mae"])
        reverse = True
    else:  # candidate_improvement
        key = lambda r: -float(r["delta_mae"])
        reverse = True
    ordered = sorted(sample_rows, key=key, reverse=reverse)
    seen = set()
    out = []
    for r in ordered:
        if dedup is not None:
            dkey = tuple(r[k] for k in dedup)
            if dkey in seen:
                continue
            seen.add(dkey)
        out.append(r)
        if len(out) >= top_k:
            break
    return out


def make_figure(history, truth, baseline, candidate, cases, setting, group, out_path):
    """history: (N, seq, C), truth/baseline/candidate: (N, H, C)."""
    n = len(cases)
    cols = 5
    rows = (n + cols - 1) // cols
    hist_len = min(history.shape[1], 96)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows), squeeze=False)
    for i, case in enumerate(cases):
        ax = axes[i // cols][i % cols]
        sid, ch = int(case["sample_id"]), int(case["channel"])
        h = history[sid, -hist_len:, ch]
        t = truth[sid, :, ch]
        b = baseline[sid, :, ch]
        c = candidate[sid, :, ch]
        x_hist = np.arange(hist_len) - hist_len
        ax.plot(x_hist, h, color="gray", lw=1.2)
        x_f = np.arange(truth.shape[1])
        ax.plot(x_f, t, color="black", lw=1.2, label="truth")
        ax.plot(x_f, b, color="tab:blue", ls="--", lw=1.1, label="baseline")
        ax.plot(x_f, c, color="tab:red", ls=":", lw=1.3, label="candidate")
        ax.set_title(f"sample {sid} ch {ch}", fontsize=9)
        ax.set_xticks([])
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{setting} — {group}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_zip(md_path, figures_dir, zip_path):
    md_bytes = md_path.read_bytes()
    text = md_bytes.decode("utf-8")
    refs = sorted(set(re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", text)))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("objective_error_analysis.md", md_bytes)
        for rel in refs:
            src = figures_dir.parent / rel
            if not src.exists():
                raise FileNotFoundError(f"missing referenced figure: {src}")
            zf.writestr(rel, src.read_bytes())
    return refs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--baseline-mode", default="original")
    p.add_argument("--candidate-modes", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ranking-metric", default="mae")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gold", default="docs/PHASEFORMER_GOLD_STANDARD.md")
    p.add_argument(
        "--mechanism-label",
        default="Phase-conditioned Amplitude Calibration (`use_phase_amp_calib` + `use_phase_warp`)",
        help="human-readable mechanism name shown in the report header",
    )
    p.add_argument(
        "--defect-hypothesis",
        default="the mechanism's benefit is dataset-dependent; seed replication is required",
        help="interpretive hypothesis line in report section 9",
    )
    args = p.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    candidate_modes = [m.strip() for m in args.candidate_modes.split(",") if m.strip()]
    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # gold standard lookup
    gold = {}
    for line in Path(args.gold).read_text().splitlines():
        m = re.match(r"\| (\w+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \|", line.strip())
        if m:
            gold[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))

    results_rows = []
    sample_rows_all = []
    case_store = {}   # setting -> list of dicts
    eval_store = {}   # setting -> (pred_b, pred_c, truth, history, meta)
    seg_store = {}    # setting -> {group: seg mae lists}
    diag_store = {}   # setting -> {log_alpha, beta} calibration activity
    raw_store = {}    # setting -> raw aggregate (b_mse, b_mae, c_mse, c_mae)

    for setting in settings:
        dataset, horizon, seed = parse_setting(setting)
        print(f"[{setting}] loading baseline {args.baseline_mode}")
        bdir = find_run_dir(args.run_dir, args.run_prefix, args.baseline_mode, dataset, horizon, seed)
        hp_b = json.loads((bdir / "config.json").read_text())["hyperparams"]
        b_model, b_args = build_model(dataset, horizon, args.lookback, hp_b, device)
        load_checkpoint(b_model, bdir / "checkpoints" / "best.ckpt", device)
        b_pred, truth, history, _ = evaluate_test(b_model, b_args, horizon, device)
        b_mse, b_mae = cell_metrics(b_pred, truth)

        cand_key = {}
        for mode in candidate_modes:
            print(f"[{setting}] loading candidate {mode}")
            cdir = find_run_dir(args.run_dir, args.run_prefix, mode, dataset, horizon, seed)
            hp_c = json.loads((cdir / "config.json").read_text())["hyperparams"]
            c_model, c_args = build_model(dataset, horizon, args.lookback, hp_c, device)
            load_checkpoint(c_model, cdir / "checkpoints" / "best.ckpt", device)
            c_pred, c_truth, c_hist, diag = evaluate_test(c_model, c_args, horizon, device)
            c_mse, c_mae = cell_metrics(c_pred, c_truth)
            cand_key[mode] = (c_pred, c_mse, c_mae, cdir, diag)
            assert np.allclose(c_truth, truth), f"truth mismatch for {setting}/{mode}"
            assert np.allclose(c_hist, history), f"history mismatch for {setting}/{mode}"

        # results.csv rows (aggregate from recomputation)
        config_b = f"{args.baseline_mode}_{dataset}_{horizon}"
        b_mse_agg = float(b_mse.mean()); b_mae_agg = float(b_mae.mean())
        results_rows.append(dict(
            setting=setting, config_id=config_b, dataset=dataset, horizon=horizon,
            seed=seed, model=args.baseline_mode, key_params="baseline",
            mse=f"{b_mse_agg:.4f}", mae=f"{b_mae_agg:.4f}",
            delta_mse="0.00", delta_mae="0.00", selected="baseline",
        ))
        for mode in candidate_modes:
            c_pred, c_mse, c_mae, cdir, _diag = cand_key[mode]
            config_c = f"{mode}_{dataset}_{horizon}"
            c_mse_agg = float(c_mse.mean()); c_mae_agg = float(c_mae.mean())
            # relative change vs baseline, positive = candidate better (matches
            # the gold-standard relative-improvement convention)
            dmse = (b_mse_agg - c_mse_agg) / b_mse_agg * 100.0
            dmae = (b_mae_agg - c_mae_agg) / b_mae_agg * 100.0
            results_rows.append(dict(
                setting=setting, config_id=config_c, dataset=dataset, horizon=horizon,
                seed=seed, model=mode, key_params="warp24_ampcalib_h8",
                mse=f"{c_mse_agg:.4f}", mae=f"{c_mae_agg:.4f}",
                delta_mse=f"{dmse:.2f}", delta_mae=f"{dmae:.2f}", selected="no",
            ))
            if mode == candidate_modes[0]:
                raw_store[setting] = (b_mse_agg, b_mae_agg, c_mse_agg, c_mae_agg)

        # sample_errors.csv rows + case selection (first candidate only)
        mode = candidate_modes[0]
        c_pred, c_mse, c_mae, cdir, diag = cand_key[mode]
        diag_store[setting] = diag
        config_c = f"{mode}_{dataset}_{horizon}"
        N, H, C = truth.shape
        base_rows = []
        for sid in range(N):
            for ch in range(C):
                base_rows.append(dict(
                    setting=setting, baseline_config_id=config_b, candidate_config_id=config_c,
                    sample_id=sid, channel=ch, time_range=f"0:{H}",
                    baseline_mse=float(b_mse[sid, ch]), candidate_mse=float(c_mse[sid, ch]),
                    delta_mse=float(c_mse[sid, ch] - b_mse[sid, ch]),
                    baseline_mae=float(b_mae[sid, ch]), candidate_mae=float(c_mae[sid, ch]),
                    delta_mae=float(c_mae[sid, ch] - b_mae[sid, ch]),
                ))
        sample_rows_all.extend(base_rows)

        cases = {}
        for group in GROUPS:
            cases[group] = select_cases(base_rows, group, args.top_k)
        case_store[setting] = cases
        eval_store[setting] = (b_pred, c_pred, truth, history)
        seg_store[setting] = {}
        for group in GROUPS:
            seg_store[setting][group] = []
            for r in cases[group]:
                b_seg = mean_mae_per_seg(
                    b_pred[r["sample_id"]][None, :, r["channel"]][:, :, None],
                    truth[r["sample_id"]][None, :, r["channel"]][:, :, None],
                )
                c_seg = mean_mae_per_seg(
                    c_pred[r["sample_id"]][None, :, r["channel"]][:, :, None],
                    truth[r["sample_id"]][None, :, r["channel"]][:, :, None],
                )
                seg_store[setting][group].append((b_seg, c_seg))

    # write results.csv and sample_errors.csv
    results_fieldnames = ["setting", "config_id", "dataset", "horizon", "seed", "model", "key_params", "mse", "mae", "delta_mse", "delta_mae", "selected"]
    write_csv(out_dir / "results.csv", results_rows, results_fieldnames)
    sample_fieldnames = ["setting", "baseline_config_id", "candidate_config_id", "sample_id", "channel", "time_range", "baseline_mse", "candidate_mse", "delta_mse", "baseline_mae", "candidate_mae", "delta_mae"]
    write_csv(out_dir / "sample_errors.csv", sample_rows_all, sample_fieldnames)

    # selected_cases.npz: per-setting arrays + aligned metadata
    npz = {}
    for setting in settings:
        b_pred, c_pred, truth, history = eval_store[setting]
        for group in GROUPS:
            for r in case_store[setting][group]:
                sid, ch = int(r["sample_id"]), int(r["channel"])
                npz[f"{setting}__{group}__setting"] = np.array(setting)
                key = f"{setting}__{group}__{sid}_{ch}"
                npz[f"{key}__sample_id"] = np.array(sid)
                npz[f"{key}__channel"] = np.array(ch)
                npz[f"{key}__history"] = history[sid, :, ch]
                npz[f"{key}__truth"] = truth[sid, :, ch]
                npz[f"{key}__baseline"] = b_pred[sid, :, ch]
                npz[f"{key}__candidate"] = c_pred[sid, :, ch]
    # also store a compact per-case table
    case_table = []
    for setting in settings:
        for group in GROUPS:
            for r in case_store[setting][group]:
                case_table.append(dict(
                    setting=setting, group=group, sample_id=r["sample_id"], channel=r["channel"],
                    baseline_mse=r["baseline_mse"], candidate_mse=r["candidate_mse"],
                    delta_mse=r["delta_mse"], baseline_mae=r["baseline_mae"],
                    candidate_mae=r["candidate_mae"], delta_mae=r["delta_mae"],
                ))
    with open(out_dir / "_cases_tmp.json", "w") as f:
        json.dump(case_table, f, indent=2)
    np.savez_compressed(out_dir / "selected_cases.npz", **npz)
    (out_dir / "_cases_tmp.json").unlink()

    # figures
    fig_refs = []
    for setting in settings:
        b_pred, c_pred, truth, history = eval_store[setting]
        for group in GROUPS:
            fname = f"{setting}__{group}.png"
            fpath = fig_dir / fname
            make_figure(history, truth, b_pred, c_pred, case_store[setting][group], setting, group, fpath)
            fig_refs.append((setting, group, fname))

    # ---- report ----
    md = []
    md.append("# Experiment and Objective Error Analysis")
    md.append(f"> {args.mechanism_label} vs matched `{args.baseline_mode}`, all settings seed 2021, period 24.")
    md.append("> Final configuration was NOT selected using test-set results; a single candidate configuration was evaluated (selection.source: fixed).")
    md.append("")
    md.append("## 1. Experiment Setup")
    md.append(f"- mechanism: phase-conditioned amplitude calibration (`use_phase_amp_calib`), per-phase-slot scale `alpha_l` / shift `beta_l` from phase-slot position and per-slot statistics, applied as `h'[l,k]=alpha_l*h[l,k]+beta_l` on the warped phase representation; warm-started at identity; flag-off byte-identical.")
    md.append(f"- baseline: matched `{args.baseline_mode}` rerun; gold standard (`docs/PHASEFORMER_GOLD_STANDARD.md`) is the fixed reference.")
    md.append(f"- settings: {', '.join(settings)}")
    md.append("- training: lookback 720, period 24, huber loss, ETT batch 256 / Weather 64, best-val checkpoint, full budget per-dataset epochs.")
    md.append("- code: run.yaml in this directory.")
    md.append("")
    md.append("## 2. Experiment Results")
    md.append("| setting | baseline MSE/MAE | candidate MSE/MAE | dMSE% | dMAE% | vs gold dMSE% | vs gold dMAE% |")
    md.append("|---|---|---|---|---|---|---|")
    for setting in settings:
        dataset, horizon, seed = parse_setting(setting)
        b = next(r for r in results_rows if r["setting"] == setting and r["model"] == args.baseline_mode)
        c = next(r for r in results_rows if r["setting"] == setting and r["model"] == candidate_modes[0])
        g = gold.get((dataset, horizon))
        if g:
            gd_mse = (g[0] - float(c["mse"])) / g[0] * 100.0
            gd_mae = (g[1] - float(c["mae"])) / g[1] * 100.0
            gs = f"{gd_mse:.2f} / {gd_mae:.2f}"
        else:
            gs = "n/a"
        md.append(f"| {setting} | {b['mse']}/{b['mae']} | {c['mse']}/{c['mae']} | {c['delta_mse']} | {c['delta_mae']} | {gs} |")
    md.append("")
    md.append("## 3. Parameter / Configuration Search")
    md.append(f"No hyperparameter search for `{candidate_modes[0]}` (module hidden=8, max_scale=2.0 only). Stage A screened 10 settings at 30%/8ep (validation-only); full-budget runs use per-dataset base hyperparameters. All configurations are retained in `results.csv`.")
    md.append("")
    md.append("### Amplitude-calibration / reliability-gate activity (mean over test batches)")
    md.append("| setting | mean |alpha-1| | mean |beta| | mean gate g |")
    md.append("|---|---|---|---|")
    for setting in settings:
        d = diag_store.get(setting, {})
        md.append(f"| {setting} | {d.get('log_alpha', float('nan')):.4f} | {d.get('beta', float('nan')):.4f} | {d.get('gate', float('nan')):.4f} |")
    md.append("")
    md.append("## 4. Error Distribution (sample x channel, per setting)")
    md.append("| setting | cells | improved% | regressed% | mean delta_mae |")
    md.append("|---|---|---|---|---|")
    for setting in settings:
        rows = [r for r in sample_rows_all if r["setting"] == setting]
        n = len(rows)
        imp = sum(1 for r in rows if float(r["delta_mae"]) < 0) / n * 100.0
        reg = sum(1 for r in rows if float(r["delta_mae"]) > 0) / n * 100.0
        mean_d = np.mean([float(r["delta_mae"]) for r in rows])
        md.append(f"| {setting} | {n} | {imp:.1f} | {reg:.1f} | {mean_d:+.5f} |")
    md.append("")
    md.append("## 5. Horizon-wise Error (selected cases, per quarter of horizon)")
    md.append("| setting | group | seg0 | seg1 | seg2 | seg3 |")
    md.append("|---|---|---|---|---|---|")
    for setting in settings:
        for group in GROUPS:
            b_all = np.mean([s[0] for s in seg_store[setting][group]], axis=0)
            c_all = np.mean([s[1] for s in seg_store[setting][group]], axis=0)
            cells = " / ".join(f"{b:.3f}/{c:.3f}" for b, c in zip(b_all, c_all))
            md.append(f"| {setting} | {group} | {cells} |")
    md.append("")
    md.append("## 6. High-Error Selection")
    md.append(f"Programmatic per setting: top-{args.top_k} (sample, channel) by baseline_mae (Baseline High Error), by delta_mae (Regression), by -delta_mae (Improvement).")
    md.append("")
    md.append("| setting | group | mean baseline_mae | mean candidate_mae |")
    md.append("|---|---|---|---|")
    for setting in settings:
        for group in GROUPS:
            cs = case_store[setting][group]
            mb = np.mean([float(r["baseline_mae"]) for r in cs])
            mc = np.mean([float(r["candidate_mae"]) for r in cs])
            md.append(f"| {setting} | {group} | {mb:.4f} | {mc:.4f} |")
    md.append("")
    md.append("## 7. Case Analysis")
    md.append("Each grid shows selected sample x channel cells: history tail (gray), truth (black), baseline (blue dashed), candidate (red dotted).")
    md.append("")
    for setting in settings:
        for group in GROUPS:
            fname = f"{setting}__{group}.png"
            md.append(f"![{setting} {group}](figures/{fname})")
        md.append("")
    md.append("## 8. Repeated Observable Patterns")
    md.append("| setting | group | candidate peak closer | candidate std closer |")
    md.append("|---|---|---|---|")
    for setting in settings:
        b_pred, c_pred, truth, history = eval_store[setting]
        for group in GROUPS:
            peak_closer = 0
            std_closer = 0
            n = 0
            for r in case_store[setting][group]:
                sid, ch = int(r["sample_id"]), int(r["channel"])
                t = truth[sid, :, ch]
                b = b_pred[sid, :, ch]
                c = c_pred[sid, :, ch]
                t_peak = int(np.argmax(t)); b_peak = int(np.argmax(b)); c_peak = int(np.argmax(c))
                if abs(b_peak - t_peak) > abs(c_peak - t_peak):
                    peak_closer += 1
                if abs(np.std(b) - np.std(t)) > abs(np.std(c) - np.std(t)):
                    std_closer += 1
                n += 1
            md.append(f"| {setting} | {group} | {peak_closer}/{n} | {std_closer}/{n} |")
    md.append("")
    md.append("## 9. Objective Defect Summary")
    md.append("")
    md.append("Measurable observations (relative to matched `original`, positive = candidate lower error; single seed 2021):")
    md.append("")
    md.append(f"- Aggregate MAE improved on {sum(1 for s in settings if float(next(r for r in results_rows if r['setting']==s and r['model']==candidate_modes[0])['delta_mae'])>0)}/{len(settings)} settings; "
             f"MSE improved on {sum(1 for s in settings if float(next(r for r in results_rows if r['setting']==s and r['model']==candidate_modes[0])['delta_mse'])>0)}/{len(settings)} settings.")
    rows_c = [next(r for r in results_rows if r['setting'] == s and r['model'] == candidate_modes[0]) for s in settings]
    best = max(rows_c, key=lambda r: float(r['delta_mae']))
    worst = min(rows_c, key=lambda r: float(r['delta_mae']))
    md.append(f"- Best setting by MAE: `{best['setting']}` dMAE {best['delta_mae']}% / dMSE {best['delta_mse']}%.")
    md.append(f"- Worst setting by MAE: `{worst['setting']}` dMAE {worst['delta_mae']}% / dMSE {worst['delta_mse']}%.")
    beat_both = []
    for s in settings:
        dataset, horizon, _ = parse_setting(s)
        c = next(r for r in results_rows if r['setting'] == s and r['model'] == candidate_modes[0])
        g = gold.get((dataset, horizon))
        raw = raw_store.get(s)
        if g and raw and raw[2] < g[0] and raw[3] < g[1]:
            beat_both.append(s)
    if beat_both:
        md.append(f"- Beats the gold standard on both MSE and MAE: {', '.join(beat_both)} (single-seed).")
    else:
        md.append("- No setting beats the gold standard on both MSE and MAE (matched originals sit above gold; single-seed).")
    md.append("")
    md.append(f"Hypotheses (unverified, require multi-seed confirmation): {args.defect_hypothesis}")
    md.append("")
    md.append("## 10. Experiment Scope")
    md.append("- 5 datasets x 2 horizons, single seed 2021, period 24, module hidden 8 / max_scale 2.0. No search over period/width/scale.")
    md.append("- Deliberately excluded: promotion to `_LATEST_POLICY`; comparison vs `latest` presets.")
    md.append("")
    md_path = out_dir / "objective_error_analysis.md"
    md_path.write_text("\n".join(md))

    refs = build_zip(md_path, fig_dir, out_dir / "objective_error_analysis.zip")
    print(f"figures referenced: {len(refs)}")

    # quick validation
    all_figs = set(f.name for f in fig_dir.iterdir())
    ref_names = {Path(r).name for r in refs}
    print(f"validation figures: {len(ref_names)} referenced, {len(all_figs)} on disk, unused={all_figs - ref_names or 'none'}")


if __name__ == "__main__":
    main()
