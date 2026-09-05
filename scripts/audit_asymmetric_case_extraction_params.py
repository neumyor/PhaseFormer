#!/usr/bin/env python3
"""Write and validate the extraction-parameter ledger for joint-route cases."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_asymmetric_joint_route_cases import COMPONENTS, candidate_path
from scripts.export_asymmetric_prediction_divergence_cases import trend_kwargs
from scripts.probe_ssa_low_frequency_trend import SSA_BY_DATASET


OUTPUT = ROOT / "research_runs/asymmetric_prediction_divergence_cases/EXTRACTION_PARAMETERS.md"
DATASETS = ("ETTh1", "Weather", "ETTm1")


def _load(path: Path) -> dict:
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def _effective(config: dict) -> dict:
    """Exactly mirror the exporter call that produced the displayed A curve."""
    return trend_kwargs(config["hyperparams"])


def _description(component: str, params: dict) -> str:
    if component == "cycle_levels":
        return f"每 {params['period_len']} 步均值，按最后周期水平锚定"
    if component == "recent_linear":
        return f"最后 {params['recent_window']} 步 OLS 斜率，外推并末点锚定"
    if component == "global_linear":
        return "完整 720 步 OLS 斜率，末点锚定"
    if component == "smooth_local":
        return f"replicate padding Gaussian 平滑，σ={params['local_sigma']:g}"
    if component == "smooth_multiscale":
        return f"Gaussian(σ={params['local_sigma']:g}) − Gaussian(σ={params['long_sigma']:g})"
    if component == "causal_ema":
        return f"单侧 EMA，α={params['causal_ema_alpha']:g}"
    if component == "holt_local_linear":
        return f"单侧 Holt，α={params['holt_level_alpha']:g}，β={params['holt_trend_beta']:g}"
    raise ValueError(component)


def _compact(component: str, params: dict) -> str:
    if component == "cycle_levels":
        return f"P={params['period_len']}"
    if component == "recent_linear":
        return f"W={params['recent_window']}"
    if component == "global_linear":
        return "L=720"
    if component == "smooth_local":
        return f"σ={params['local_sigma']:g}"
    if component == "smooth_multiscale":
        return f"σshort={params['local_sigma']:g}; σlong={params['long_sigma']:g}"
    if component == "causal_ema":
        return f"α={params['causal_ema_alpha']:g}"
    if component == "holt_local_linear":
        return f"α={params['holt_level_alpha']:g}; β={params['holt_trend_beta']:g}"
    raise ValueError(component)


def main() -> None:
    root = OUTPUT.parent
    manifest_path = root / "manifest.csv"
    manifest = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    expected_rows = len(DATASETS) * len(COMPONENTS) * 3
    if len(manifest) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} manifest rows, found {len(manifest)}")
    pairs: dict[tuple[str, str], dict] = {}
    for dataset in DATASETS:
        for component in COMPONENTS:
            minus_path = candidate_path(dataset, component, "minus_component")
            only_path = candidate_path(dataset, component, "component_only")
            minus, only = _load(minus_path), _load(only_path)
            for config, expected_mode in ((minus, "minus_component"), (only, "component_only")):
                hp = config["hyperparams"]
                if (config["dataset"], config["lookback"], config["horizon"], config["seed"], config["period"],
                    hp.get("weak_residual_asymmetric_component"), hp.get("weak_residual_asymmetric_input_mode", "minus_component")) != (
                    dataset, 720, 96, 2021, 24, component, expected_mode):
                    raise RuntimeError(f"candidate config mismatch: {config}")
            minus_params, only_params = _effective(minus), _effective(only)
            if minus_params != only_params:
                raise RuntimeError(f"X-A/Only-A extraction mismatch: {dataset} {component}")
            pairs[dataset, component] = minus_params
    for dataset in DATASETS:
        for component in COMPONENTS:
            subset = [row for row in manifest if row["dataset"] == dataset and row["component"] == component]
            if len(subset) != 3 or any(not (root / row["figure"]).is_file() for row in subset):
                raise RuntimeError(f"manifest/figure mismatch: {dataset} {component}")

    lines = [
        "# Asymmetric prediction-divergence cases: extraction parameter ledger", "",
        "This ledger is an audit of the images and `manifest.csv` in this same directory. It does **not** infer values from the current experiment plan. For every existing component/dataset pair it reads the exact X-A and Only-A checkpoint `config.json` files selected by `scripts/export_asymmetric_joint_route_cases.py`, verifies their common effective extractor arguments, and verifies three manifest figures exist.", "",
        "## Verified scope", "",
        f"- {len(manifest)} manifest rows = 3 datasets × {len(COMPONENTS)} components × 3 channel-0 cases.",
        "- All runs: validation-only, L=720, H=96, seed=2021, period=24; displayed A was recomputed by the exporter using the X-A checkpoint's exact effective arguments.",
        "- X-A and Only-A configs agree on every extraction argument for every row below.",
        "",
        "## Existing case images: exact effective parameters", "",
        "| Component / extraction rule | ETTh1 | Weather | ETTm1 |",
        "|---|---:|---:|---:|",
    ]
    for component in COMPONENTS:
        params = pairs["ETTh1", component]
        label = f"`{component}` — {_description(component, params)}"
        lines.append("| " + label + " | " + " | ".join(_compact(component, pairs[dataset, component]) for dataset in DATASETS) + " |")
    lines += [
        "", "## SSA low-frequency trend: frozen prospective parameters", "",
        "`ssa_low_frequency` is not yet represented by prediction-divergence case images in this directory, so it is deliberately not presented as a parameter used for an existing result. The following values are now frozen for the next SSA X-A/Only-A run and are the values used by the separate validation-only SSA visual probe. Each sample/channel is embedded in a W×(L−W+1) trajectory matrix; the first R candidate eigentriples are Hankel-reconstructed, scored by Fourier energy at periods at least Pmin, the top r are summed, and the output is endpoint-anchored.", "",
        "| Dataset | W | retained r | candidate R | Pmin (steps) |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        p = SSA_BY_DATASET[dataset]
        lines.append(f"| {dataset} | {p['window']} | {p['rank']} | {p['candidate_rank']} | {p['min_period']} |")
    lines += [
        "", "## Important provenance boundary", "",
        "The existing ETTh1 `causal_ema` and `holt_local_linear` case images above use the delivered checkpoints' α=.024 (and Holt β=.006). They predate the separate slow-parameter ETTh1 retraining (α=.006, β=.0015), and must not be relabelled as that newer run until its images are regenerated.",
        "",
        "Validation command: `python scripts/audit_asymmetric_case_extraction_params.py`. The command fails if a checkpoint identity, mode, effective extraction arguments, manifest cardinality, or a referenced figure disagrees.",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
