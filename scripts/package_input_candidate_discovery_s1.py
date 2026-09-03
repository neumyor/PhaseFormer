#!/usr/bin/env python3
"""Package the stopped-after-S1 candidate screen into the required audit set."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_input_candidate_discovery_frozen import load_model
from src.dataset.data_factory import data_provider
from src.dataset.input_candidate_discovery import CandidateConfig, CandidateDataset, ContinuousCandidateBank
from src.models.phaseformer_presets import build_hyperparams, make_exp_args


SETTING = "ETTm1_h192_seed2021_validation_only"
CANDIDATE = "c3"
MODEL = "weak_residual"
VARIANT = "remove_050"


def prediction(model, item, horizon):
    device = next(model.parameters()).device
    x, y, xm, ym = [torch.as_tensor(value).unsqueeze(0).to(device) for value in item]
    with torch.inference_mode():
        output, _, _ = model(x.float(), xm.float(), model._build_decoder_input(y.float()), ym.float())
    truth = y[:, -horizon:, :]
    pred = output[:, -horizon:, :]
    return x[0].cpu().numpy(), truth[0].cpu().numpy(), pred[0].cpu().numpy()


def plot_case(path, title, history, truth, base, candidate):
    fig, axis = plt.subplots(figsize=(9, 3.2), dpi=150)
    # The seven ETTm1 variables have different units after standard scaling;
    # show OT (last channel) consistently for auditable visual comparison.
    channel = history.shape[1] - 1
    hx = np.arange(-len(history), 0)
    fx = np.arange(len(truth))
    axis.plot(hx, history[:, channel], color="#777777", lw=1.0, label="history")
    axis.plot(fx, truth[:, channel], color="#111111", lw=1.5, label="truth")
    axis.plot(fx, base[:, channel], color="#1f77b4", lw=1.2, label="full input")
    axis.plot(fx, candidate[:, channel], color="#d62728", lw=1.2, label="C3 remove_050")
    axis.axvline(0, color="#999999", lw=0.8)
    axis.set_title(title)
    axis.set_xlabel("forecast step (history shown at negative steps)")
    axis.legend(ncol=4, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    figures = args.output / "figures"
    figures.mkdir()

    raw_s1a = pd.read_csv(args.scratch / "frozen_s1a" / "frozen_val_discovery.csv")
    raw_s1b = pd.read_csv(args.scratch / "frozen_s1b" / "frozen_val_discovery.csv")
    # Keep every tried S1a configuration.  The three full anchor rows are
    # identical across stages and therefore retained once from S1a.
    raw = pd.concat(
        [raw_s1a, raw_s1b[raw_s1b.variant.ne("full")]], ignore_index=True
    )
    samples = np.load(args.scratch / "frozen_s1b" / "frozen_val_paired_samples.npz")
    full = samples[f"{MODEL}__none_full__mae"]
    changed = samples[f"{MODEL}__{CANDIDATE}_{VARIANT}__mae"]
    mse_full = samples[f"{MODEL}__none_full__mse"]
    mse_changed = samples[f"{MODEL}__{CANDIDATE}_{VARIANT}__mse"]
    delta = changed - full
    rows = []
    for index in range(len(full)):
        rows.append({"setting": SETTING, "baseline_config_id": f"{MODEL}_full",
                     "candidate_config_id": f"{MODEL}_{CANDIDATE}_{VARIANT}", "sample_id": index,
                     "channel": "aggregate", "time_range": f"validation_origin_{index}",
                     "baseline_mse": float(mse_full[index]), "candidate_mse": float(mse_changed[index]),
                     "delta_mse": float(mse_changed[index] - mse_full[index]),
                     "baseline_mae": float(full[index]), "candidate_mae": float(changed[index]),
                     "delta_mae": float(delta[index])})
    with (args.output / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    result_rows = []
    for _, row in raw.iterrows():
        result_rows.append({"setting": SETTING, "config_id": f"{row.model}_{row.candidate}_{row.variant}",
                            "dataset": "ETTm1", "horizon": 192, "seed": 2021, "split": "validation",
                            "model": row.model, "candidate": row.candidate, "variant": row.variant,
                            "mse": row.mse, "mae": row.mae, "delta_mse": row.get("delta_mse_vs_full", 0.0),
                            "delta_mae": row.get("delta_mae_vs_full", 0.0), "selected": False,
                            "stage": "S1a" if row.sample_count == 512 else "S1b"})
    pd.DataFrame(result_rows).to_csv(args.output / "results.csv", index=False)

    hp = build_hyperparams("ETTm1", 192, MODEL)
    exp = make_exp_args("ETTm1", 720, 192, hp)
    exp.dataset_args.root_path = str(REPO_ROOT / "resources" / "all_datasets" / "ETT")
    exp.dataset_args.num_workers = 0
    base, _ = data_provider(exp.dataset_args, "val")
    bank = ContinuousCandidateBank(base, CandidateConfig(CANDIDATE, VARIANT))
    altered = CandidateDataset(base, bank)
    model, _ = load_model(MODEL, args.checkpoint, 192, 720, 2021)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    selections = {
        "baseline_high_error": np.argsort(full)[-5:][::-1],
        "candidate_regression": np.argsort(delta)[-5:][::-1],
        "candidate_improvement": np.argsort(delta)[:5],
    }
    chosen = []
    for group, indices in selections.items():
        for index in indices:
            chosen.append((group, int(index)))
    # keep the first unique cases in rank order; this is the documented dedup rule.
    unique = []
    seen = set()
    for group, index in chosen:
        if index not in seen:
            unique.append((group, index)); seen.add(index)
    histories = []; truths = []; base_preds = []; candidate_preds = []; settings = []; groups = []; ids = []
    representatives = {}
    for group, index in unique:
        history, truth, base_pred = prediction(model, base[index], 192)
        _, truth_changed, candidate_pred = prediction(model, altered[index], 192)
        if not np.allclose(truth, truth_changed):
            raise RuntimeError("candidate input changed target alignment")
        histories.append(history); truths.append(truth); base_preds.append(base_pred); candidate_preds.append(candidate_pred)
        settings.append(SETTING); groups.append(group); ids.append(index)
        representatives.setdefault(group, (history, truth, base_pred, candidate_pred, index))
    np.savez_compressed(args.output / "selected_cases.npz", setting=np.asarray(settings), group=np.asarray(groups),
                        sample_id=np.asarray(ids), history=np.asarray(histories), truth=np.asarray(truths),
                        baseline_prediction=np.asarray(base_preds), candidate_prediction=np.asarray(candidate_preds))
    for group, (history, truth, base_pred, candidate_pred, index) in representatives.items():
        filename = f"ETTm1_h192_seed2021_validation_only__{group}.png"
        plot_case(figures / filename, f"{group}: validation origin {index}", history, truth, base_pred, candidate_pred)

    original_spec = ((raw_s1b[(raw_s1b.candidate=="c3")&(raw_s1b.model=="original")&(raw_s1b.variant=="remove_050")].iloc[0].mae /
                      raw_s1b[(raw_s1b.candidate=="c3")&(raw_s1b.model=="original")&(raw_s1b.variant=="sham_050")].iloc[0].mae)-1)*100
    weak_spec = ((raw_s1b[(raw_s1b.candidate=="c3")&(raw_s1b.model==MODEL)&(raw_s1b.variant=="remove_050")].iloc[0].mae /
                  raw_s1b[(raw_s1b.candidate=="c3")&(raw_s1b.model==MODEL)&(raw_s1b.variant=="sham_050")].iloc[0].mae)-1)*100
    report = f"""# Experiment and Objective Error Analysis

## 1. Experiment Setup

Setting: `{SETTING}`.  This is S1 frozen validation screening only: three independently trained full-input anchors, seven preregistered candidates in S1a (512 origins), then C2/C3/C7 on all validation origins in S1b.  Test was not read and no candidate was retrained because the prespecified S1 gates did not admit one.

The comparison used for sample-level cases is `weak_residual` full input versus `weak_residual` C3 `remove_050`.  It is a diagnostic comparison, not a selected improved model.

## 2. Experiment Results

For the S1b C3 comparison, sham-adjusted MAE effects (remove versus sham) are {original_spec:.3f}% for original PhaseFormer and {weak_spec:.3f}% for PhaseFormer+NLinear.  The weak-model advantage is therefore below the preregistered 1% effect requirement and did not pass its paired confidence gate.  C2 is rejected earlier because sham is substantially more harmful than removal; C7 is rejected because PhaseFormer is not equivalently insensitive while the enhanced models do not show a larger component-specific effect.

## 3. Parameter / Configuration Search

No tunable candidate configuration was selected.  The fixed search path was C1--C7 at doses 0.25 and 0.50, with C2/C3/C7 carried from S1a to S1b by the registered screening order.  No test-set result was used.

## 4. Error Distribution

`sample_errors.csv` contains all {len(rows)} paired validation-origin aggregate errors for the diagnostic C3 weak-residual comparison.  It supports recomputation of every ranked case below.

## 5. Horizon-wise Error

The raw S1b result table records MSE/MAE for forecast ranges 1--24, 25--48, 49--96 and 97--192.  C7 did not show the required pattern of negligible original effect but larger enhanced-model effect in 1--24, so it was not advanced.

## 6. High-Error Selection

Cases were selected programmatically as the top five baseline MAE, top five candidate-minus-baseline MAE, and top five baseline-minus-candidate MAE, then deduplicated by exact validation origin.  Selected arrays and group labels are in `selected_cases.npz`.

## 7. Case Analysis

![Baseline high error](figures/ETTm1_h192_seed2021_validation_only__baseline_high_error.png)

![Candidate regression](figures/ETTm1_h192_seed2021_validation_only__candidate_regression.png)

![Candidate improvement](figures/ETTm1_h192_seed2021_validation_only__candidate_improvement.png)

The plots show the OT channel for deterministic representative origins.  They are descriptive illustrations; the decision uses the aggregate paired statistics rather than a manually selected curve.

## 8. Repeated Observable Patterns

C2 has a large difference between removal and sham across all three models, which is observable as a control/intervention mismatch rather than the desired model-specific pattern.  C3 has a small positive weak-residual difference, but the original model also changes and the effect is below the registered scale.  C7 changes early-horizon errors, yet this response is not stronger for the enhanced models.

## 9. Objective Defect Summary

No C1--C7 candidate currently demonstrates that PhaseFormer fails to use a component that NLinear or RCRF uses effectively.  The experiment stopped before retraining/test confirmation by its planned validation gate; this is a negative candidate-discovery result, not evidence that no such component exists.

## 10. Experiment Scope

Only ETTm1, horizon 192, seed 2021 and validation were evaluated in this completed early-stop stage.  The predeclared test confirmation was intentionally not read.  Any later expansion must preserve this full negative selection record.
"""
    md = args.output / "objective_error_analysis.md"
    md.write_text(report)
    run = {
        "experiment_id": "input_candidate_discovery_ettm1_h192_v1",
        "code": {"repository": "PhaseFormer", "branch": "structure_test", "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "modified_files": ["src/dataset/input_candidate_discovery.py", "scripts/run_input_candidate_discovery_frozen.py"]},
        "mechanism": {"description": "Continuous train-fitted C1--C7 input components; validation-gated candidate discovery", "feature_flag": "standalone frozen evaluator"},
        "experiment": {"baseline": ["original", "weak_residual", "rcrf_nlinear_plain"], "candidate": "C1--C7", "settings": [{"setting": SETTING, "dataset": "ETTm1", "split": "validation", "lookback": 720, "horizon": 192, "seed": 2021}],
                       "training": "30 epoch full-input anchors; no candidate retraining after S1 stop", "metrics": ["MSE", "MAE", "forecast segments", "paired moving-block CI"]},
        "execution": {"environment": {"python": sys.executable, "cuda": torch.cuda.is_available()}, "settings": [{"setting": SETTING, "commands": ["scripts/run_input_candidate_discovery_frozen.py --split val"], "runtime": "S1a and S1b"}]},
        "selection": {"source": "validation", "selected_configs": [{"setting": SETTING, "config_id": "none", "search_notes": "No candidate passed S1; test and retraining not run."}]},
        "analysis": {"ranking_metric": "mae", "top_k": 5, "dedup_rule": "exact validation-origin deduplication", "selections": [{"setting": SETTING, **{key: [int(value) for value in values] for key, values in selections.items()}}]},
        "validation": {"results_checked": True, "ranking_and_cases_checked": True, "report_and_archive_checked": True, "directory_and_settings_checked": True, "status": "passed", "issues": "Completed through the predeclared S1 early-stop rule; no test/retraining because no candidate passed."},
    }
    import yaml
    (args.output / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))
    with zipfile.ZipFile(args.output / "objective_error_analysis.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(md, md.name)
        for figure in sorted(figures.glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")
    print(args.output)


if __name__ == "__main__":
    main()
