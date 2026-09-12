#!/usr/bin/env python3
"""Experiment 4: frequency-band sensitivity probe.

For 3 model variants per setting -- full-rank baseline, a trained low-rank
model, and the fully causal-EMA-smoothed model (smooth_ratio=1.0) -- zero out
one frequency band of the real test-set input x at a time (via rfft/irfft,
leaving y untouched) and measure the resulting test MSE/MAE degradation vs.
the unperturbed baseline. Expectation: the fully-smoothed variant should be
roughly insensitive to high-frequency-band removal (that information was
already discarded going into the model), while the full-rank and low-rank
variants should both still show meaningful sensitivity -- the direct
empirical test of the "capacity axis vs. temporal-resolution axis"
hypothesis. Requires server-side checkpoints (the causal-EMA-smoothed
variant only exists under research_runs/causal_ema_smooth_sweep_v1 on the
remote server).

See docs/PhaseFormer_lowrank_mechanism_analysis.md for background,
disclosure, and results.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _weak_residual_analysis_common import (
    ROOT,
    SETTINGS,
    build_model_and_loader,
    causal_ema_row,
    evaluate_model,
    full_rank_row,
    load_checkpoint_state_dict,
    low_rank_rows,
    pick_device,
    setting_label,
)

OUTPUT_ROOT = ROOT / "research_runs/lowrank_mechanism_analysis_v1"
NUM_BANDS = 6
MAX_EVAL_SAMPLES = 2000  # caps 6 bands x 3 variants x 7 settings forward-pass cost


def make_band_mask(num_freqs: int, num_bands: int) -> list[slice]:
    # log-spaced band edges over the rfft bin axis, low -> high frequency
    edges = [0] + [
        int(round(num_freqs * (2 ** (band - num_bands)))) for band in range(1, num_bands)
    ] + [num_freqs]
    edges = sorted(set(edges))
    return [slice(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def zero_band_transform(band: slice):
    def transform(x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        spectrum[:, band, :] = 0
        return torch.fft.irfft(spectrum, n=x.shape[1], dim=1)

    return transform


def capped_loader(loader, max_samples: int):
    seen = 0
    for batch in loader:
        yield batch
        seen += batch[0].shape[0]
        if seen >= max_samples:
            return


def load_variant(kind: str, dataset: str, horizon: int):
    if kind == "full_rank":
        row = full_rank_row(dataset, horizon)
    elif kind == "low_rank":
        row = low_rank_rows(dataset, horizon)[0]  # smallest tested rank
    elif kind == "smoothed":
        row = causal_ema_row(dataset, horizon, smooth_ratio=1.0)
    else:
        raise ValueError(kind)
    model, loader = build_model_and_loader(row)
    model.load_state_dict(load_checkpoint_state_dict(row), strict=True)
    return model, loader


def evaluate_setting(dataset: str, horizon: int, device: torch.device) -> list[dict]:
    rows = []
    for variant in ("full_rank", "low_rank", "smoothed"):
        model, loader = load_variant(variant, dataset, horizon)
        capped = list(capped_loader(loader, MAX_EVAL_SAMPLES))
        baseline = evaluate_model(model, capped, horizon, device)
        sample_x = capped[0][0]
        num_freqs = torch.fft.rfft(sample_x, dim=1).shape[1]
        bands = make_band_mask(num_freqs, NUM_BANDS)
        for band_index, band in enumerate(bands):
            perturbed = evaluate_model(model, capped, horizon, device, x_transform=zero_band_transform(band))
            rows.append(
                {
                    "setting": setting_label(dataset, horizon),
                    "dataset": dataset,
                    "horizon": horizon,
                    "variant": variant,
                    "band_index": band_index,
                    "band_range": f"{band.start}:{band.stop}",
                    "baseline_mse": baseline["mse"],
                    "baseline_mae": baseline["mae"],
                    "perturbed_mse": perturbed["mse"],
                    "perturbed_mae": perturbed["mae"],
                    "delta_mse_pct": 100.0 * (perturbed["mse"] - baseline["mse"]) / baseline["mse"],
                    "delta_mae_pct": 100.0 * (perturbed["mae"] - baseline["mae"]) / baseline["mae"],
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(all_rows: list[dict], figures_dir: Path) -> None:
    settings = sorted({row["setting"] for row in all_rows})
    figure, axes = plt.subplots(len(settings), 1, figsize=(7, 3 * len(settings)), squeeze=False)
    for row_index, setting in enumerate(settings):
        axis = axes[row_index, 0]
        for variant, color in (("full_rank", "C0"), ("low_rank", "C1"), ("smoothed", "C2")):
            rows = sorted(
                (r for r in all_rows if r["setting"] == setting and r["variant"] == variant),
                key=lambda r: r["band_index"],
            )
            axis.plot(
                [r["band_index"] for r in rows],
                [r["delta_mse_pct"] for r in rows],
                marker="o", color=color, label=variant,
            )
        axis.set_title(f"{setting}: ΔMSE% vs. frequency band removed (low→high)", fontsize=9)
        axis.set_xlabel("band index (low → high frequency)")
        axis.set_ylabel("ΔMSE%")
        axis.grid(True, linewidth=0.3, alpha=0.5)
        if row_index == 0:
            axis.legend(fontsize=8)
    figure.tight_layout()
    out_path = figures_dir / "freq_band_sensitivity.png"
    figure.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    all_rows: list[dict] = []
    for setting in SETTINGS:
        all_rows.extend(evaluate_setting(setting["dataset"], setting["horizon"], device))
    csv_path = OUTPUT_ROOT / "freq_band_sensitivity.csv"
    write_csv(all_rows, csv_path)
    print(f"wrote {csv_path}")
    plot_comparison(all_rows, figures_dir)


if __name__ == "__main__":
    main()
