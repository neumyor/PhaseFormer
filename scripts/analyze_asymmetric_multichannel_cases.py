#!/usr/bin/env python3
"""Unified all-channel validation audit for asymmetric residual trend studies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.asymmetric_trend_components import TREND_COMPONENTS, extract_trend_component
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args

DATASETS = ("ETTh1", "Weather", "ETTm1")
COMPONENTS = ("cycle_levels", "recent_linear", "global_linear", "smooth_local", "smooth_multiscale")


def read_json(path):
    return json.loads(path.read_text())


def run_dir(root: Path, dataset: str, component: str | None):
    matches = []
    for path in (root / "runs").glob("*"):
        config = path / "config.json"
        if not config.exists():
            continue
        value = read_json(config)
        if value["dataset"] == dataset and value["horizon"] == 96 and value["hyperparams"].get("weak_residual_asymmetric_component") == component:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"need one run: {dataset=} {component=}; found {matches}")
    return matches[0]


def load_model(path: Path):
    config = read_json(path / "config.json")
    hp = config["hyperparams"]
    args = make_exp_args(config["dataset"], config["lookback"], config["horizon"], hp)
    model = PhaseFormer(PhaseFormerPresetConfig(args, config["lookback"], config["horizon"], hp))
    metric = next(csv.DictReader((path / "metrics.csv").open()))
    checkpoint = ROOT / metric["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False).get("state_dict")
    model.load_state_dict(state, strict=True)
    return model, args, checkpoint


def select(delta, top_k=10, min_separation=96):
    """Five largest positive and five most negative deltas, with origin de-dup."""
    candidates = []
    for sign, order in (("asymmetric_regression", np.argsort(delta.reshape(-1))[::-1]),
                        ("asymmetric_improvement", np.argsort(delta.reshape(-1)))):
        picked = []
        for flat in order:
            origin, channel = np.unravel_index(flat, delta.shape)
            if all(abs(origin - previous_origin) >= min_separation for previous_origin, _ in picked):
                picked.append((int(origin), int(channel)))
            if len(picked) == top_k // 2:
                break
        candidates.extend((sign, origin, channel) for origin, channel in picked)
    return candidates


def plot_case(path, dataset, component, origin, channel, history, removed, truth, baseline, candidate, base_mae, cand_mae):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.7), dpi=135)
    history_x = np.arange(-history.size, 0); future_x = np.arange(truth.size)
    axes[0].plot(history_x, history, color="#555555", lw=.8, label="full history X")
    axes[0].plot(history_x, removed, color="#E07A2D", lw=.8, label=f"residual history X-{component}")
    axes[0].axvline(0, color="#999", lw=.8); axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    axes[0].set_title(f"{dataset} H96 validation origin {origin}, channel {channel}: {component}")
    recent = 192
    axes[1].plot(np.arange(-recent, 0), history[-recent:], color="#555555", lw=.8, label="full history X")
    axes[1].plot(np.arange(-recent, 0), removed[-recent:], color="#E07A2D", lw=.8, label="X-A")
    axes[1].plot(future_x, truth, color="black", lw=1.5, label="truth")
    axes[1].plot(future_x, baseline, color="#2878B5", lw=1.1, label="Baseline-full")
    axes[1].plot(future_x, candidate, color="#C43C39", lw=1.1, label=f"Asymmetric-{component}")
    axes[1].axvline(0, color="#999", lw=.8); axes[1].legend(fontsize=8, ncol=3, loc="upper left")
    axes[1].set_title(f"channel MAE: baseline={base_mae:.4f}, asymmetric={cand_mae:.4f}, delta={cand_mae-base_mae:+.4f}")
    axes[1].set_xlabel("forecast step (history is negative)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--etth1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_trend_discovery"))
    parser.add_argument("--weather-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_weather_h96_scratch"))
    parser.add_argument("--ettm1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_ettm1_h96_scratch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.require_cuda and not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = {"ETTh1": args.etth1_root, "Weather": args.weather_root, "ETTm1": args.ettm1_root}
    args.output.mkdir(parents=True); figures = args.output / "figures"; figures.mkdir()
    results, error_rows, selected = [], [], {key: [] for key in ("setting", "component", "group", "sample_id", "channel", "history", "removed_history", "truth", "baseline_prediction", "candidate_prediction", "baseline_mae", "candidate_mae")}
    selection_map, links = {}, []
    for dataset in DATASETS:
        baseline_path = run_dir(roots[dataset], dataset, None)
        baseline, exp_args, base_checkpoint = load_model(baseline_path); baseline.to(device).eval()
        exp_args.dataset_args.num_workers = args.num_workers
        _, loader = data_provider(exp_args.dataset_args, "val")
        base_batches = []
        with torch.inference_mode():
            for batch in loader:
                x, y, xm, ym = [item.to(device) for item in batch]
                out, _, _ = baseline(x.float(), xm.float(), baseline._build_decoder_input(y.float()), ym.float())
                base_batches.append((x.cpu(), y[:, -96:].float().cpu(), xm.cpu(), ym.cpu(), out[:, -96:].cpu()))
        for component in COMPONENTS:
            candidate_path = run_dir(roots[dataset], dataset, component)
            candidate, _, candidate_checkpoint = load_model(candidate_path); candidate.to(device).eval()
            histories=[]; removeds=[]; truths=[]; base_preds=[]; cand_preds=[]
            with torch.inference_mode():
                for x_cpu, truth_cpu, xm_cpu, ym_cpu, base_cpu in base_batches:
                    x=x_cpu.to(device); truth=truth_cpu.to(device); xm=xm_cpu.to(device); ym=ym_cpu.to(device)
                    out, _, _ = candidate(x.float(), xm.float(), candidate._build_decoder_input(truth.float()), ym.float())
                    removed = x.float() - extract_trend_component(x.float(), component, period_len=24)
                    histories.append(x.cpu().numpy()); removeds.append(removed.cpu().numpy()); truths.append(truth.cpu().numpy()); base_preds.append(base_cpu.numpy()); cand_preds.append(out[:, -96:].cpu().numpy())
            history=np.concatenate(histories); removed=np.concatenate(removeds); truth=np.concatenate(truths); base=np.concatenate(base_preds); candidate=np.concatenate(cand_preds)
            base_abs=np.abs(base-truth); candidate_abs=np.abs(candidate-truth)
            base_mae=base_abs.mean(1); candidate_mae=candidate_abs.mean(1)
            base_mse=((base-truth)**2).mean(1); candidate_mse=((candidate-truth)**2).mean(1)
            setting=f"{dataset}_h96_seed2021_validation"; config_id=f"asymmetric_{component}"
            if component == COMPONENTS[0]:
                results.append({"setting":setting,"config_id":"baseline_full","dataset":dataset,"horizon":96,"seed":2021,"model":"weak_residual","key_params":"residual=X","mse":float(base_mse.mean()),"mae":float(base_mae.mean()),"delta_mse":0.0,"delta_mae":0.0,"selected":True})
            results.append({"setting":setting,"config_id":config_id,"dataset":dataset,"horizon":96,"seed":2021,"model":"weak_residual","key_params":f"residual=X-{component}","mse":float(candidate_mse.mean()),"mae":float(candidate_mae.mean()),"delta_mse":float(candidate_mse.mean()-base_mse.mean()),"delta_mae":float(candidate_mae.mean()-base_mae.mean()),"selected":True})
            for origin in range(len(history)):
                for channel in range(history.shape[2]):
                    error_rows.append({"setting":setting,"baseline_config_id":"baseline_full","candidate_config_id":config_id,"component":component,"sample_id":origin,"channel":channel,"time_range":f"validation_origin_{origin}","baseline_mse":base_mse[origin,channel],"candidate_mse":candidate_mse[origin,channel],"delta_mse":candidate_mse[origin,channel]-base_mse[origin,channel],"baseline_mae":base_mae[origin,channel],"candidate_mae":candidate_mae[origin,channel],"delta_mae":candidate_mae[origin,channel]-base_mae[origin,channel]})
            chosen=select(candidate_mae-base_mae); selection_map[f"{setting}__{component}"]=chosen
            for rank,(group,origin,channel) in enumerate(chosen,1):
                filename=f"{setting}__{component}__{group}_{rank:02d}_origin_{origin}_ch_{channel}.png"
                plot_case(figures/filename,dataset,component,origin,channel,history[origin,:,channel],removed[origin,:,channel],truth[origin,:,channel],base[origin,:,channel],candidate[origin,:,channel],base_mae[origin,channel],candidate_mae[origin,channel])
                links.append(filename)
                for key,value in (("setting",setting),("component",component),("group",group),("sample_id",origin),("channel",channel),("history",history[origin,:,channel]),("removed_history",removed[origin,:,channel]),("truth",truth[origin,:,channel]),("baseline_prediction",base[origin,:,channel]),("candidate_prediction",candidate[origin,:,channel]),("baseline_mae",base_mae[origin,channel]),("candidate_mae",candidate_mae[origin,channel])): selected[key].append(value)
            del histories, removeds, truths, base_preds, cand_preds
            del history, removed, truth, base, candidate, base_abs, candidate_abs, base_mae, candidate_mae, base_mse, candidate_mse
            torch.cuda.empty_cache()
    write_csv(args.output/"results.csv",results); write_csv(args.output/"sample_errors.csv",error_rows)
    # The CSV is now the full audit source; releasing its Python dictionaries
    # prevents a large cross-dataset audit from retaining all rows while zipping
    # the comparatively tiny selected-case arrays.
    del error_rows
    np.savez_compressed(args.output/"selected_cases.npz",**{key:np.asarray(value) for key,value in selected.items()})
    run={"experiment_id":args.output.name,"mechanism":{"description":"PhaseFormer receives X; only NLinear residual receives X-A using full-X shared RevIN statistics","feature_flag":"weak_residual_asymmetric_component"},"experiment":{"settings":[{"setting":f"{d}_h96_seed2021_validation","dataset":d,"split":"validation","lookback":720,"horizon":96,"seed":2021} for d in DATASETS],"training":{"max_epochs":30,"loss":"huber","checkpoint_rule":"lowest validation loss"}},"selection":{"source":"validation","selected_configs":[{"setting":row["setting"],"config_id":row["config_id"],"search_notes":"fixed component; no test selection"} for row in results]},"analysis":{"ranking_metric":"absolute per-origin/channel MAE difference; signed groups retained","top_k":10,"dedup_rule":"within dataset/component, any selected origins differ by >=96","selections":selection_map},"validation":{"results_checked":False,"ranking_and_cases_checked":False,"report_and_archive_checked":False,"directory_and_settings_checked":False,"status":"incomplete"}}
    (args.output/"run.yaml").write_text(json.dumps(run,indent=2)+"\n")
    table="\n".join(f"| {row['setting']} | `{row['config_id']}` | {float(row['mae']):.4f} | {float(row['mse']):.4f} | {float(row['delta_mae']):+.4f} |" for row in results)
    cases="\n".join(f"| {s} | `{c}` | {g} | {i} | {ch} | {float(ca)-float(ba):+.4f} |" for s,c,g,i,ch,ba,ca in zip(selected['setting'],selected['component'],selected['group'],selected['sample_id'],selected['channel'],selected['baseline_mae'],selected['candidate_mae']))
    images="\n".join(f"![](figures/{name})" for name in links)
    report=f"""# Experiment and Objective Error Analysis

## 1. Experiment Setup

Three validation-only settings: ETTh1, Weather and ETTm1, each L720→H96, seed2021. Baseline-full provides X to both paths; each candidate provides X to PhaseFormer and X-A only to the NLinear residual path, sharing full-X RevIN statistics. All models are independently trained best-validation checkpoints. No test data was read.

## 2. Experiment Results

| Setting | Config | MAE | MSE | Delta MAE |
|---|---|---:|---:|---:|
{table}

## 3. Parameter / Configuration Search

No parameter search: the five A definitions and all training settings were frozen before the single-seed runs.

## 4. Error Distribution

`sample_errors.csv` contains every validation origin × channel pair for all 15 Baseline/candidate comparisons.

## 5. Horizon-wise Error

All MAE/MSE values average the 96 forecast steps.

## 6. High-Error Selection

For each dataset×component, five largest positive and five most negative channel-wise MAE differences are selected, with origin separation ≥96. Thus the visual evidence includes both degradation and improvement patterns.

| Setting | Component | Direction | Origin | Channel | Delta MAE |
|---|---|---|---:|---:|---:|
{cases}

## 7. Case Analysis

Each figure and `selected_cases.npz` preserve the selected channel's full history, residual-visible X-A history, truth, and both forecasts; `channel` metadata identifies its source variable.

{images}

## 8. Repeated Observable Patterns

Case direction is selected mechanically. Aggregate results and both signed case groups must be read together; the figures alone do not establish causal use of a component.

## 9. Objective Defect Summary

The measurable effect is the signed candidate-minus-baseline per-channel MAE. Any mechanism interpretation remains a hypothesis because component removal also changes residual-branch input distribution.

## 10. Experiment Scope

This is a three-dataset H96 validation-only single-seed sample audit, not a test or multi-seed confirmation.
"""
    markdown=args.output/"objective_error_analysis.md"; markdown.write_text(report)
    with zipfile.ZipFile(args.output/"objective_error_analysis.zip","w",zipfile.ZIP_DEFLATED) as archive:
        archive.write(markdown,markdown.name)
        for name in links: archive.write(figures/name,f"figures/{name}")
    print(args.output)


if __name__ == "__main__": main()
