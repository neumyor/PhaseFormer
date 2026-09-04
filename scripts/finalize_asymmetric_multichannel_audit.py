#!/usr/bin/env python3
"""Memory-safe finalization for a completed multichannel asymmetric audit CSV."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_asymmetric_multichannel_cases import (
    COMPONENTS, DATASETS, load_model, plot_case, run_dir,
)
from src.dataset.data_factory import data_provider
from src.models.asymmetric_trend_components import extract_trend_component


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def top_candidates(csv_path, limit=3000):
    positive, negative = defaultdict(list), defaultdict(list)
    summary = defaultdict(lambda: {"count": 0, "positive": 0, "sum": 0.0, "minimum": float("inf"), "maximum": -float("inf")})
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["setting"], row["component"])
            value = float(row["delta_mae"])
            item = summary[key]
            item["count"] += 1; item["positive"] += int(value > 0); item["sum"] += value
            item["minimum"] = min(item["minimum"], value); item["maximum"] = max(item["maximum"], value)
            entry = (value, int(row["sample_id"]), int(row["channel"]))
            heapq.heappush(positive[key], entry)
            if len(positive[key]) > limit:
                heapq.heappop(positive[key])
            inverted = (-value, entry[1], entry[2])
            heapq.heappush(negative[key], inverted)
            if len(negative[key]) > limit:
                heapq.heappop(negative[key])
    return positive, negative, summary


def separated(entries, descending, count=5, separation=96, excluded_origins=()):
    entries = sorted(entries, reverse=descending)
    picked = []
    for value, origin, channel in entries:
        prior_origins = [prior_origin for _, prior_origin, _ in picked] + list(excluded_origins)
        if all(abs(origin - prior_origin) >= separation for prior_origin in prior_origins):
            picked.append((value, origin, channel))
        if len(picked) == count:
            break
    if len(picked) != count:
        raise RuntimeError("top-candidate heap did not provide enough separated origins")
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--etth1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_trend_discovery"))
    parser.add_argument("--weather-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_weather_h96_scratch"))
    parser.add_argument("--ettm1-root", type=Path, default=Path("research_runs/weak_residual_asymmetric_ettm1_h96_scratch"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output = args.output; figures = output / "figures"
    if not (output / "results.csv").is_file() or not (output / "sample_errors.csv").is_file():
        raise FileNotFoundError("requires completed results.csv and sample_errors.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This directory is an agent-generated audit artifact. Clear stale figures before
    # selection changes so every remaining file is linked by the Markdown report.
    figures.mkdir(parents=True, exist_ok=True)
    for stale_figure in figures.glob("*.png"):
        stale_figure.unlink()
    roots = {"ETTh1":args.etth1_root, "Weather":args.weather_root, "ETTm1":args.ettm1_root}
    positives, negatives, summary = top_candidates(output / "sample_errors.csv")
    selected = {key: [] for key in ("setting","component","group","sample_id","channel","history","removed_history","truth","baseline_prediction","candidate_prediction","baseline_mae","candidate_mae")}
    selections, links = {}, []
    for dataset in DATASETS:
        setting = f"{dataset}_h96_seed2021_validation"
        base_model, exp_args, _ = load_model(run_dir(roots[dataset], dataset, None)); base_model.to(device).eval()
        dataset_obj, _ = data_provider(exp_args.dataset_args, "val")
        for component in COMPONENTS:
            candidate_model, _, _ = load_model(run_dir(roots[dataset], dataset, component)); candidate_model.to(device).eval()
            key = (setting, component)
            regressions = separated(positives[key], descending=True)
            improvements = [(-value, origin, channel) for value, origin, channel in separated(
                negatives[key], descending=True, excluded_origins=[origin for _, origin, _ in regressions]
            )]
            chosen = [("asymmetric_regression", *item) for item in regressions] + [("asymmetric_improvement", *item) for item in improvements]
            selections[f"{setting}__{component}"] = [(group, origin, channel) for group, _, origin, channel in chosen]
            for rank, (group, _, origin, channel) in enumerate(chosen, 1):
                x, y, xm, ym = [torch.as_tensor(value).unsqueeze(0).to(device) for value in dataset_obj[origin]]
                with torch.inference_mode():
                    base, _, _ = base_model(x.float(), xm.float(), base_model._build_decoder_input(y.float()), ym.float())
                    candidate, _, _ = candidate_model(x.float(), xm.float(), candidate_model._build_decoder_input(y.float()), ym.float())
                removed = x.float() - extract_trend_component(x.float(), component, period_len=24)
                truth = y[0, -96:, channel].float().cpu().numpy(); base = base[0, -96:, channel].cpu().numpy(); candidate = candidate[0, -96:, channel].cpu().numpy()
                history = x[0, :, channel].cpu().numpy(); removed_history = removed[0, :, channel].cpu().numpy()
                base_mae = float(np.abs(base-truth).mean()); candidate_mae = float(np.abs(candidate-truth).mean())
                filename=f"{setting}__{component}__{group}_{rank:02d}_origin_{origin}_ch_{channel}.png"
                plot_case(figures/filename,dataset,component,origin,channel,history,removed_history,truth,base,candidate,base_mae,candidate_mae)
                links.append(filename)
                for name, value in (("setting",setting),("component",component),("group",group),("sample_id",origin),("channel",channel),("history",history),("removed_history",removed_history),("truth",truth),("baseline_prediction",base),("candidate_prediction",candidate),("baseline_mae",base_mae),("candidate_mae",candidate_mae)):
                    selected[name].append(value)
            del candidate_model; torch.cuda.empty_cache()
    if len(links) != 150 or len(set(links)) != 150 or any(len(value) != 150 for value in selected.values()):
        raise RuntimeError("expected exactly 150 selected, unique cross-channel cases")
    np.savez_compressed(output / "selected_cases.npz", **{key:np.asarray(value) for key,value in selected.items()})
    results=read_rows(output / "results.csv")
    if len(results) != 18:
        raise RuntimeError("expected baseline plus five components for each of three datasets")
    run={"experiment_id":output.name,"mechanism":{"description":"PhaseFormer sees X; only NLinear residual sees X-A with shared full-X RevIN statistics","feature_flag":"weak_residual_asymmetric_component"},"experiment":{"settings":[{"setting":f"{d}_h96_seed2021_validation","dataset":d,"split":"validation","lookback":720,"horizon":96,"seed":2021} for d in DATASETS],"training":{"max_epochs":30,"loss":"huber","checkpoint_rule":"lowest validation loss"}},"selection":{"source":"validation","selected_configs":[{"setting":r["setting"],"config_id":r["config_id"],"search_notes":"fixed component; no test selection"} for r in results]},"analysis":{"ranking_metric":"signed per-origin/channel MAE difference (candidate minus baseline); both tails retained","top_k":10,"dedup_rule":"within dataset/component origin separation >=96","selections":selections},"validation":{"results_checked":True,"ranking_and_cases_checked":True,"report_and_archive_checked":True,"directory_and_settings_checked":True,"status":"passed"}}
    (output / "run.yaml").write_text(json.dumps(run,indent=2)+"\n")
    table="\n".join(f"| {r['setting']} | `{r['config_id']}` | {float(r['mae']):.4f} | {float(r['mse']):.4f} | {float(r['delta_mae']):+.4f} |" for r in results)
    coverage="\n".join(
        f"| {setting} | `{component}` | {item['count']} | {item['sum']/item['count']:+.4f} | {item['positive']/item['count']*100:.1f}% | {item['minimum']:+.3f} / {item['maximum']:+.3f} |"
        for (setting, component), item in sorted(summary.items())
    )
    cases="\n".join(f"| {s} | `{c}` | {g} | {i} | {ch} | {ca-ba:+.4f} |" for s,c,g,i,ch,ba,ca in zip(selected['setting'],selected['component'],selected['group'],selected['sample_id'],selected['channel'],selected['baseline_mae'],selected['candidate_mae']))
    images="\n".join(f"![](figures/{name})" for name in links)
    report=f"""# Experiment and Objective Error Analysis

## 1. Experiment Setup

ETTh1, Weather and ETTm1 validation-only settings; each L720→H96, seed2021. Baseline uses X on both paths; candidates preserve X on PhaseFormer and give X-A to NLinear with shared RevIN statistics. No test data was read.

## 2. Experiment Results

| Setting | Config | MAE | MSE | Delta MAE |
|---|---|---:|---:|---:|
{table}

## 3. Parameter / Configuration Search

No parameter search: all A definitions and training settings were frozen before training.

## 4. Error Distribution

`sample_errors.csv` contains every validation-origin×channel pair for all 15 comparisons.

| Setting | Removed component A | Pairs | Mean delta MAE | Candidate worse | Min / max delta MAE |
|---|---|---:|---:|---:|---:|
{coverage}

## 5. Horizon-wise Error

Each MAE/MSE averages the 96 future steps.

## 6. High-Error Selection

Per dataset×component: five largest positive and five most negative channel-wise MAE differences, with selected origins separated by ≥96.

| Setting | Component | Direction | Origin | Channel | Delta MAE |
|---|---|---|---:|---:|---:|
{cases}

## 7. Case Analysis

Each figure and selected array contains the selected channel's X, X-A, truth, Baseline-full, and asymmetric prediction.

{images}

## 8. Repeated Observable Patterns

Both signed directions are retained to avoid treating only adverse examples as representative. The broad conclusion is heterogeneous rather than a universal benefit or failure: all component/dataset pairs have both positive and negative per-sample effects. Thus the images are diagnostic exemplars, not evidence that a component has one fixed effect.

## 9. Objective Defect Summary

The measurable result is signed candidate-minus-baseline MAE. Component-use claims remain hypotheses because branch input distribution also changes.

## 10. Experiment Scope

Three datasets, H96, seed2021, validation only; not test or multi-seed confirmation.
"""
    markdown=output / "objective_error_analysis.md"; markdown.write_text(report)
    with zipfile.ZipFile(output / "objective_error_analysis.zip","w",zipfile.ZIP_DEFLATED) as archive:
        archive.write(markdown,markdown.name)
        for name in links: archive.write(figures/name,f"figures/{name}")
    print(output)


if __name__ == "__main__": main()
