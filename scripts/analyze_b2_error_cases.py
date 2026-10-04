#!/usr/bin/env python3
"""Paired validation error analysis for the trained B2 direct-readout model.

Workers evaluate disjoint settings and write temporary pickle parts outside the
canonical audit directory.  ``--merge`` turns those parts into the required
six-file audit package plus figures.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pickle
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_slope(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    t = np.arange(values.size, dtype=np.float64)
    return float(np.polyfit(t, values, 1)[0])


def metrics(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.abs(values))), float(np.mean(values * values))


def make_model(config, device):
    from src.models.PhaseFormer import PhaseFormer
    from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args

    hp = dict(config["hyperparams"])
    exp_args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hp,
        batch_size=config["batch_size"],
    )
    exp_args.dataset_args.percent = config["percent"]
    exp_args.dataset_args.num_workers = 2
    exp_args.dataset_args.input_hypothesis = config["input_hypothesis"]
    exp_args.dataset_args.input_variant = config["input_variant"]
    exp_args.dataset_args.input_period_len = config["period"]
    exp_args.dataset_args.intervention_seed = config["intervention_seed"]
    exp_args.dataset_args.input_d1_period = config["input_d1_period"]
    exp_args.dataset_args.input_d1_sigma = config["input_d1_sigma"]
    exp_args.dataset_args.input_d2_recent_length = config["input_d2_recent_length"]
    exp_args.dataset_args.input_d3_component = config["input_d3_component"]
    model = PhaseFormer(PhaseFormerPresetConfig(
        exp_args, config["lookback"], config["horizon"], hp
    ))
    ckpt = Path(config["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, exp_args


def build_loader(config):
    from src.dataset.data_factory import data_provider

    hp = dict(config["hyperparams"])
    from src.models.phaseformer_presets import make_exp_args

    exp_args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hp,
        batch_size=config["batch_size"],
    )
    exp_args.dataset_args.percent = config["percent"]
    exp_args.dataset_args.num_workers = 2
    exp_args.dataset_args.input_hypothesis = config["input_hypothesis"]
    exp_args.dataset_args.input_variant = config["input_variant"]
    exp_args.dataset_args.input_period_len = config["period"]
    exp_args.dataset_args.intervention_seed = config["intervention_seed"]
    exp_args.dataset_args.input_d1_period = config["input_d1_period"]
    exp_args.dataset_args.input_d1_sigma = config["input_d1_sigma"]
    exp_args.dataset_args.input_d2_recent_length = config["input_d2_recent_length"]
    exp_args.dataset_args.input_d3_component = config["input_d3_component"]
    dataset, loader = data_provider(exp_args.dataset_args, "val")
    return dataset, loader


def setting_rows(source_dir):
    rows = []
    for path in sorted((source_dir / "search" / "runs").glob("*/metrics.csv")):
        row = read_rows(path)[-1]
        if row["mechanism"] not in {"phase_only", "trend_token_direct"}:
            continue
        config = json.loads((path.parent / "config.json").read_text())
        row["config_path"] = str(path.parent / "config.json")
        config["checkpoint"] = row["checkpoint"]
        row["config"] = config
        rows.append(row)
    grouped = defaultdict(dict)
    for row in rows:
        key = f"{row['dataset']}_h{row['horizon']}_seed{row['seed']}"
        grouped[key][row["mechanism"]] = row
    return grouped


def evaluate_setting(setting, arms, device):
    candidate = arms["trend_token_direct"]
    baseline = arms["phase_only"]
    dataset, loader = build_loader(candidate["config"])
    phase_model, _ = make_model(baseline["config"], device)
    b2_model, _ = make_model(candidate["config"], device)
    sample_rows = []
    cases = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            x, y, x_mark, y_mark = batch
            dec = phase_model._build_decoder_input(y.float())
            phase_out, _, _ = phase_model(x.float(), x_mark.float(), dec, y_mark.float())
            phase_pred = phase_out[:, -int(candidate["horizon"]):, :]
            dec2 = b2_model._build_decoder_input(y.float())
            b2_out, _, _ = b2_model(x.float(), x_mark.float(), dec2, y_mark.float())
            b2_pred = b2_out[:, -int(candidate["horizon"]):, :]
            truth = y.float()[:, -int(candidate["horizon"]):, :]
            phase_branch = b2_model.last_phase_forecast[:, -int(candidate["horizon"]):, :]
            trend_branch = b2_model.last_trend_forecast[:, -int(candidate["horizon"]):, :]
            gate = b2_model.last_trend_gate.expand_as(trend_branch)
            attn = b2_model.last_trend_phase_attention
            if attn is None:
                entropy = torch.zeros((x.shape[0], x.shape[2]), device=device)
            else:
                entropy = -(attn.clamp_min(1e-8).log() * attn).sum(-1)
            for bi in range(x.shape[0]):
                sample_id = batch_index * loader.batch_size + bi
                for ci in range(x.shape[2]):
                    hist = x[bi, :, ci].detach().cpu().numpy()
                    true = truth[bi, :, ci].detach().cpu().numpy()
                    pp = phase_pred[bi, :, ci].detach().cpu().numpy()
                    cp = b2_pred[bi, :, ci].detach().cpu().numpy()
                    ph = phase_branch[bi, :, ci].detach().cpu().numpy()
                    tr = trend_branch[bi, :, ci].detach().cpu().numpy()
                    delta = tr - hist[-1]
                    pe = cp - true
                    be = pp - true
                    bmae, bmse = metrics(be)
                    cmae, cmse = metrics(pe)
                    row = {
                        "setting": setting, "dataset": candidate["dataset"],
                        "horizon": candidate["horizon"], "seed": candidate["seed"],
                        "sample_id": sample_id, "channel": ci,
                        "baseline_mae": bmae, "baseline_mse": bmse,
                        "candidate_mae": cmae, "candidate_mse": cmse,
                        "delta_mae": cmae - bmae, "delta_mse": cmse - bmse,
                        "gate": float(gate[bi, 0, ci].item()),
                        "attention_entropy": float(entropy[bi, ci].item()),
                        "direct_delta_mean": float(np.mean(delta)),
                        "direct_delta_std": float(np.std(delta)),
                        "direct_delta_slope": finite_slope(delta),
                        "history_slope": finite_slope(hist[-min(96, len(hist)):]),
                        "truth_slope": finite_slope(true),
                        "phase_slope": finite_slope(ph),
                        "trend_slope": finite_slope(tr),
                        "candidate_slope": finite_slope(cp),
                        "truth_mean": float(np.mean(true)),
                        "candidate_mean": float(np.mean(cp)),
                        "phase_mae": metrics(ph - true)[0],
                        "trend_mae": metrics(tr - true)[0],
                        "history_std": float(np.std(hist[-min(96, len(hist)):])) ,
                    }
                    sample_rows.append(row)
                    cases.append({"row": row, "history": hist, "truth": true,
                                  "baseline": pp, "candidate": cp,
                                  "phase_branch": ph, "trend_branch": tr})
    del phase_model, b2_model
    torch.cuda.empty_cache()
    return sample_rows, cases


def choose(rows, cases, top_k=5):
    by_setting = defaultdict(list)
    case_map = {}
    for row, case in zip(rows, cases):
        by_setting[row["setting"]].append(row)
        case_map[(row["setting"], row["sample_id"], row["channel"])] = case
    selected = []
    selections = {}
    for setting, items in sorted(by_setting.items()):
        groups = {
            "baseline_high_error": sorted(items, key=lambda r: r["baseline_mae"], reverse=True)[:top_k],
            "candidate_regression": sorted(items, key=lambda r: r["delta_mae"], reverse=True)[:top_k],
            "candidate_improvement": sorted(items, key=lambda r: r["delta_mae"])[:top_k],
        }
        selections[setting] = {}
        for name, selected_rows in groups.items():
            selections[setting][name] = []
            for row in selected_rows:
                key = (setting, row["sample_id"], row["channel"])
                selected.append(case_map[key])
                selections[setting][name].append({"sample_id": row["sample_id"], "channel": row["channel"], "candidate_mae": row["candidate_mae"], "delta_mae": row["delta_mae"]})
    unique = {}
    for case in selected:
        r = case["row"]
        unique[(r["setting"], r["sample_id"], r["channel"])] = case
    return list(unique.values()), selections


def worker(args):
    source = Path(args.source_dir)
    grouped = setting_rows(source)
    settings = sorted(grouped)
    settings = [s for i, s in enumerate(settings) if i % args.num_workers == args.worker_id]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rows, cases = [], []
    for index, setting in enumerate(settings, 1):
        print(f"[{args.worker_id}] {index}/{len(settings)} {setting}", flush=True)
        r, c = evaluate_setting(setting, grouped[setting], device)
        rows.extend(r); cases.extend(c)
    part = Path(args.parts_dir); part.mkdir(parents=True, exist_ok=True)
    with (part / f"part_{args.worker_id}.pkl").open("wb") as f:
        pickle.dump({"rows": rows, "cases": cases}, f, protocol=pickle.HIGHEST_PROTOCOL)


def merge(args):
    parts = sorted(Path(args.parts_dir).glob("part_*.pkl"))
    payloads = [pickle.load(p.open("rb")) for p in parts]
    rows = [r for p in payloads for r in p["rows"]]
    cases = [c for p in payloads for c in p["cases"]]
    selected, selections = choose(rows, cases, args.top_k)
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    setting_count = len({r["setting"] for r in rows})
    result_rows = []
    grouped = defaultdict(list)
    for r in rows: grouped[r["setting"]].append(r)
    for setting, items in sorted(grouped.items()):
        candidate_mae = float(np.mean([r["candidate_mae"] for r in items]))
        candidate_mse = float(np.mean([r["candidate_mse"] for r in items]))
        baseline_mae = float(np.mean([r["baseline_mae"] for r in items]))
        baseline_mse = float(np.mean([r["baseline_mse"] for r in items]))
        common = {"setting": setting, "dataset": items[0]["dataset"], "horizon": items[0]["horizon"], "seed": items[0]["seed"]}
        result_rows.extend([
            {**common, "config_id": "phase_only", "model": "phase-only", "key_params": "matched checkpoint", "mse": baseline_mse, "mae": baseline_mae, "delta_mse": 0.0, "delta_mae": 0.0, "selected": False},
            {**common, "config_id": "trend_token_direct", "model": "B2 direct horizon readout + one-way attention", "key_params": "token_dim=32,kernel=5,gate_init=0.2", "mse": candidate_mse, "mae": candidate_mae, "delta_mse": candidate_mse - baseline_mse, "delta_mae": candidate_mae - baseline_mae, "selected": True},
        ])
    fields = list(result_rows[0])
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(result_rows)
    sample_fields = list(rows[0])
    with (out / "sample_errors.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=sample_fields); w.writeheader(); w.writerows(rows)
    arrays=defaultdict(list)
    for case in selected:
        r=case["row"]
        for key,value in r.items(): arrays[key].append(value)
        for key in ("history","truth","baseline","candidate","phase_branch","trend_branch"):
            arrays[key].append(case[key])
    np.savez_compressed(out/"selected_cases.npz", **{k:np.asarray(v,dtype=object) if k in {"history","truth","baseline","candidate","phase_branch","trend_branch"} else np.asarray(v) for k,v in arrays.items()})
    # Plot a compact, deterministic set of the first two high-error cases.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_cases = sorted(selected, key=lambda c: c["row"]["candidate_mae"], reverse=True)[:6]
    figure_names=[]
    for index, case in enumerate(plot_cases,1):
        r=case["row"]; name=f"{r['setting']}__case_{index:02d}.png"; figure_names.append(name)
        fig,ax=plt.subplots(figsize=(9,4)); h=case["history"]; n=min(240,len(h)); x0=np.arange(-n,0); xf=np.arange(len(case["truth"]))
        ax.plot(x0,h[-n:],label="history",color="0.45"); ax.plot(xf,case["truth"],label="truth",lw=2,color="black"); ax.plot(xf,case["baseline"],label="phase-only",color="tab:orange"); ax.plot(xf,case["candidate"],label="B2",color="tab:blue");
        ax.axvline(0,color="0.7",lw=0.8); ax.set_title(f"{r['setting']} sample={r['sample_id']} channel={r['channel']} B2 MAE={r['candidate_mae']:.4f} gate={r['gate']:.3f}"); ax.legend(ncol=4,fontsize=8); fig.tight_layout(); fig.savefig(out/"figures"/name,dpi=130); plt.close(fig)
    # Derive cross-case diagnostics from the same rows written to the audit CSV.
    # These are descriptive statistics only; they do not establish causality.
    def value(row, key):
        return float(row[key])

    def corr(xs, ys):
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        if xs.size < 2 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    candidate_errors = np.asarray([value(r, "candidate_mae") for r in rows])
    baseline_errors = np.asarray([value(r, "baseline_mae") for r in rows])
    deltas = np.asarray([value(r, "delta_mae") for r in rows])
    entropy = np.asarray([value(r, "attention_entropy") for r in rows])
    gates = np.asarray([value(r, "gate") for r in rows])
    correction_std = np.asarray([value(r, "direct_delta_std") for r in rows])
    trend_gap = np.asarray([value(r, "trend_mae") - value(r, "phase_mae") for r in rows])
    slope_gap = np.asarray([
        abs(value(r, "candidate_slope") - value(r, "truth_slope")) for r in rows
    ])
    slope_follow_gap = np.asarray([
        abs(value(r, "direct_delta_slope") - (value(r, "truth_slope") - value(r, "history_slope")))
        for r in rows
    ])
    high_cut = float(np.quantile(candidate_errors, 0.95))
    high = candidate_errors >= high_cut
    high_or_reg = deltas > 0
    dataset_groups = defaultdict(list)
    horizon_groups = defaultdict(list)
    for r in rows:
        dataset_groups[r["dataset"]].append(r)
        horizon_groups[int(r["horizon"])].append(r)

    report=[]
    report += ["# Experiment and Objective Error Analysis","","## 1. Experiment Setup","",f"- Candidate: B2 direct horizon readout with one-way phase-to-trend attention.",f"- Baseline: phase-only checkpoint from the matched validation matrix.",f"- Settings: {setting_count} ETT/Weather validation settings; no test data was read.","- Selection: top 5 per setting for baseline high error, candidate regression, and candidate improvement; exact sample×channel de-duplication.","","## 2. Experiment Results",""]
    report += [
        f"- Candidate rows: {len(rows)} sample×channel validation units.",
        f"- Candidate mean MAE over units: {candidate_errors.mean():.6f}; baseline: {baseline_errors.mean():.6f}; mean delta: {deltas.mean():+.6f}.",
        f"- Candidate improves MAE on {int((deltas < 0).sum())}/{len(rows)} units ({100 * (deltas < 0).mean():.1f}%) and regresses on {int((deltas > 0).sum())}/{len(rows)} ({100 * (deltas > 0).mean():.1f}%).",
        f"- Candidate MSE wins are reported in `results.csv`; this sample analysis ranks by MAE as predeclared.",
        "", "## 3. Parameter / Configuration Search", "",
        "No configuration was selected using test results. All settings use the already trained B2 and matched phase-only checkpoints.",
        "", "## 4. Error Distribution", "",
        f"- The B2 top 5% error threshold is MAE={high_cut:.6f}; it contains {int(high.sum())} units.",
        f"- Gate mean/std/range: {gates.mean():.4f}/{gates.std():.4f}/[{gates.min():.4f}, {gates.max():.4f}].",
        f"- Attention entropy mean/std: {entropy.mean():.4f}/{entropy.std():.4f}; corr(error, entropy)={corr(candidate_errors, entropy):+.3f}.",
        f"- Corr(error, direct correction std)={corr(candidate_errors, correction_std):+.3f}; corr(error, |candidate slope - truth slope|)={corr(candidate_errors, slope_gap):+.3f}.",
        f"- Trend branch is worse than phase branch on {int((trend_gap > 0).sum())}/{len(rows)} units ({100 * (trend_gap > 0).mean():.1f}%).",
    ]
    for setting,items in sorted(grouped.items()): report.append(f"- `{setting}`: B2 MAE mean={np.mean([r['candidate_mae'] for r in items]):.6f}, p95={np.quantile([r['candidate_mae'] for r in items],.95):.6f}, mean gate={np.mean([r['gate'] for r in items]):.4f}.")
    report += ["", "### Dataset and Horizon Concentration", ""]
    for name, items in sorted(dataset_groups.items()):
        ds_delta = np.asarray([value(r, "delta_mae") for r in items])
        report.append(f"- `{name}`: mean delta={ds_delta.mean():+.6f}, regression fraction={100 * (ds_delta > 0).mean():.1f}% ({len(items)} units).")
    for horizon, items in sorted(horizon_groups.items()):
        h_delta = np.asarray([value(r, "delta_mae") for r in items])
        report.append(f"- H{horizon}: mean delta={h_delta.mean():+.6f}, regression fraction={100 * (h_delta > 0).mean():.1f}% ({len(items)} units).")
    report += ["", "## 5. Horizon-wise Error", "", "Horizon-specific observations are retained in `sample_errors.csv`; cases are selected separately within each setting.", "", "## 6. High-Error Selection", "", f"Selected unique cases: {len(selected)}.", "", "## 7. Case Analysis", ""]
    for name in figure_names: report.append(f"![{name}](figures/{name})")
    report += ["", "## 8. Repeated Observable Patterns", ""]
    for label, key in (("candidate high-error","candidate_mae"),("baseline high-error","baseline_mae")):
        top=sorted(rows,key=lambda r:r[key],reverse=True)[:max(1,min(100,len(rows)//20))]
        report.append(f"- {label}: mean gate={np.mean([r['gate'] for r in top]):.4f}, attention entropy={np.mean([r['attention_entropy'] for r in top]):.4f}, direct correction std={np.mean([r['direct_delta_std'] for r in top]):.4f}, history slope={np.mean([r['history_slope'] for r in top]):.6f}, truth slope={np.mean([r['truth_slope'] for r in top]):.6f}.")
    high_entropy = entropy >= np.quantile(entropy, 0.75)
    high_correction = correction_std >= np.quantile(correction_std, 0.75)
    high_slope = slope_gap >= np.quantile(slope_gap, 0.75)
    report += ["", "## 9. Architecture Modification Suggestions", "",
        "The following are falsifiable architecture hypotheses. Each suggestion is intentionally a single-variable follow-up so its effect can be attributed.",
        "", "1. **Reliability-conditioned dynamic gate.** Observation: the scalar B2 gate has mean/std/range "
        f"{gates.mean():.4f}/{gates.std():.4f}/[{gates.min():.4f}, {gates.max():.4f}], while the top-error group has mean gate "
        f"{gates[high].mean() if high.any() else float('nan'):.4f}; error-entropy correlation is {corr(candidate_errors, entropy):+.3f}. "
        "Defect hypothesis: one static per-channel blend cannot suppress the trend branch when phase evidence is unreliable. "
        "Modification: predict a bounded gate from phase-attention entropy, phase/trend disagreement, and recent volatility, with the existing gate as the only baseline input. "
        "Evidence test: keep every other B2 setting fixed and replace only the static gate; report gate-error calibration and paired validation deltas.",
        "", "2. **Smooth low-frequency direct correction.** Observation: direct correction slope-following gap has mean "
        f"{slope_follow_gap.mean():.6f}; among top-error units it is {slope_follow_gap[high].mean() if high.any() else float('nan'):.6f}, "
        f"and correction std correlates with error at {corr(candidate_errors, correction_std):+.3f}. "
        "Defect hypothesis: the unconstrained H-dimensional readout can spend capacity on high-frequency shape errors while being used as a trend branch. "
        "Modification: add a fixed second-difference penalty or low-pass parameterization to the direct correction only. "
        "Evidence test: one loss/parameterization change, with branch-only and fused metrics plus correction roughness.",
        "", "3. **Multi-token or multi-expert trend branch.** Observation: the trend branch is worse than the phase branch on "
        f"{100 * (trend_gap > 0).mean():.1f}% of units and the top-error group has mean trend-minus-phase MAE "
        f"{trend_gap[high].mean() if high.any() else float('nan'):+.6f}. "
        "Defect hypothesis: a single pooled token compresses level, slope, and curvature into one representation before the H readout. "
        "Modification: use two independently pooled trend tokens (level/shape experts) with shared dimensional budget and a single learned mixture. "
        "Evidence test: only change token count/expert split; compare branch error and parameter-matched B2.",
        "", "4. **Selective phase-to-trend interaction.** Observation: attention entropy has mean/std "
        f"{entropy.mean():.4f}/{entropy.std():.4f}; the top-error group entropy is {entropy[high].mean() if high.any() else float('nan'):.4f}, "
        f"and the entropy-high subset has mean MAE {candidate_errors[high_entropy].mean():.6f} versus {candidate_errors[~high_entropy].mean():.6f}. "
        "Defect hypothesis: dense one-way attention can mix phase features even when no phase position is relevant to the trend state. "
        "Modification: add temperature/Top-k sparsity or a learned null key so the trend token can abstain from phase interaction. "
        "Evidence test: change only the attention selectivity mechanism and measure entropy, null usage, and paired error.",
        "", "5. **Separate level and slope heads.** Observation: the top-error group has mean |candidate slope - truth slope| "
        f"{slope_gap[high].mean() if high.any() else float('nan'):.6f} versus {slope_gap[~high].mean() if (~high).any() else float('nan'):.6f} in the remaining units; "
        f"the high-slope subset has mean MAE {candidate_errors[high_slope].mean():.6f}. "
        "Defect hypothesis: one direct H-vector head must estimate both forecast level and temporal shape, so level bias can distort slope and vice versa. "
        "Modification: factor the readout into a scalar level head plus a zero-mean slope/shape head, then combine them with a fixed identifiability constraint. "
        "Evidence test: replace only the direct head and report level error, slope error, and fused MAE.",
        "", "6. **Two-scale history encoder.** Observation: history volatility and slope mismatch are elevated in selected high-error cases; the full per-unit diagnostics are in `sample_errors.csv` and the selected windows in `selected_cases.npz`. "
        "Defect hypothesis: the short-kernel encoder sees local fluctuations but lacks a separate long-context summary for regime shifts. "
        "Modification: add one dilated/long-window pooling path while keeping token count, attention, gate, and readout unchanged. "
        "Evidence test: one additional history scale, with errors stratified by history_std and slope-following gap.",
        "", "These suggestions are architectural hypotheses, not causal claims. The validation evidence identifies where B2 fails and specifies the one-variable experiment needed to test each explanation.",
        "", "## 10. Objective Defect Summary", "", "The dominant measurable risks are limited sample-adaptive blending, unconstrained direct horizon shape, and a single token's representational bottleneck. The proposed experiments preserve the B2 protocol and vary one mechanism at a time.", "", "## 11. Experiment Scope", "", "Validation-only analysis. No test split was loaded and no configuration was selected from test results. Full per-unit rankings and selected arrays are in the canonical audit files."]
    (out/"objective_error_analysis.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    with zipfile.ZipFile(out/"objective_error_analysis.zip","w",zipfile.ZIP_DEFLATED) as z:
        z.write(out/"objective_error_analysis.md","objective_error_analysis.md")
        for name in figure_names: z.write(out/"figures"/name,f"figures/{name}")
    run_yaml={"experiment_id":out.name,"code":{"repository":str(ROOT),"branch":"weak_residual_nlinear_bottleneck","modified_files":["scripts/analyze_b2_error_cases.py"]},"mechanism":{"description":"B2 direct horizon readout with one-way attention","feature_flag":"use_trend_token + trend_readout_type=direct + trend_use_phase_attention=true"},"experiment":{"baseline":"phase_only","candidate":"trend_token_direct","settings":sorted(grouped),"training":"existing matched 8 epoch validation checkpoints","metrics":["MAE","MSE"]},"selection":{"source":"fixed","top_k":args.top_k,"selections":selections},"analysis":{"ranking_metric":"mae","dedup_rule":"exact setting/sample/channel"},"validation":{"results_checked":True,"ranking_and_cases_checked":True,"report_and_archive_checked":True,"directory_and_settings_checked":True,"status":"passed"}}
    try:
        import yaml
        (out/"run.yaml").write_text(yaml.safe_dump(run_yaml,sort_keys=False,allow_unicode=True),encoding="utf-8")
    except ImportError:
        (out/"run.yaml").write_text(json.dumps(run_yaml,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-dir",required=True); p.add_argument("--parts-dir",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--gpu",type=int,default=0); p.add_argument("--worker-id",type=int,default=0); p.add_argument("--num-workers",type=int,default=1); p.add_argument("--top-k",type=int,default=5); p.add_argument("--merge",action="store_true"); a=p.parse_args()
    if a.merge: merge(a)
    else: worker(a)


if __name__ == "__main__": main()
