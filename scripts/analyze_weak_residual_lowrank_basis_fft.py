#!/usr/bin/env python3
"""Experiment 3: FFT analysis of the learned low-rank basis vectors.

Pure post-hoc analysis, no eval loop. For a representative trained low-rank
checkpoint per setting (smallest tested rank, to maximize interpretability of
each individual mode), take each row of encoder.weight (rank x pooled_len,
"what temporal pattern this mode reads from the input") and each column of
decoder.weight (pred_len x rank, "what temporal pattern this mode writes to
the output"), compute the FFT magnitude spectrum, and report the spectral
centroid and high-frequency energy fraction per mode. The hypothesis under
test: compression keeps modes that are still high-frequency/phase-sensitive
(unlike smoothing, which pre-filters the input itself).

See docs/PhaseFormer_lowrank_mechanism_analysis.md for background,
disclosure, and results.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _weak_residual_analysis_common import (
    ROOT,
    SETTINGS,
    load_checkpoint_state_dict,
    low_rank_rows,
    setting_label,
)

OUTPUT_ROOT = ROOT / "research_runs/lowrank_mechanism_analysis_v1"
HIGH_FREQ_FRACTION = 0.5  # top half of the spectrum counts as "high frequency"


def spectral_stats(vector: np.ndarray) -> tuple[float, float]:
    spectrum = np.abs(np.fft.rfft(vector))
    freqs = np.arange(len(spectrum))
    total_energy = spectrum.sum()
    if total_energy == 0:
        return 0.0, 0.0
    centroid = float((freqs * spectrum).sum() / total_energy)
    cutoff = int(len(spectrum) * (1 - HIGH_FREQ_FRACTION))
    high_freq_fraction = float(spectrum[cutoff:].sum() / total_energy)
    return centroid, high_freq_fraction


def analyze_setting(dataset: str, horizon: int) -> tuple[list[dict], dict]:
    row = low_rank_rows(dataset, horizon)[0]  # smallest tested rank
    state_dict = load_checkpoint_state_dict(row)
    encoder_weight = state_dict["weak_period_residual.encoder.weight"].numpy()  # rank x pooled_len
    decoder_weight = state_dict["weak_period_residual.decoder.weight"].numpy()  # pred_len x rank
    rank = row["rank"]
    rows = []
    for mode in range(rank):
        enc_centroid, enc_high_frac = spectral_stats(encoder_weight[mode, :])
        dec_centroid, dec_high_frac = spectral_stats(decoder_weight[:, mode])
        rows.append(
            {
                "setting": setting_label(dataset, horizon),
                "dataset": dataset,
                "horizon": horizon,
                "rank_used": rank,
                "mode_index": mode,
                "encoder_spectral_centroid": round(enc_centroid, 3),
                "encoder_high_freq_energy_frac": round(enc_high_frac, 4),
                "decoder_spectral_centroid": round(dec_centroid, 3),
                "decoder_high_freq_energy_frac": round(dec_high_frac, 4),
            }
        )
    return rows, {"encoder_weight": encoder_weight, "decoder_weight": decoder_weight, "rank": rank}


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_setting(dataset: str, horizon: int, arrays: dict, figures_dir: Path) -> None:
    rank = arrays["rank"]
    figure, axes = plt.subplots(rank, 2, figsize=(8, 2.2 * rank), squeeze=False)
    for mode in range(rank):
        enc_spectrum = np.abs(np.fft.rfft(arrays["encoder_weight"][mode, :]))
        dec_spectrum = np.abs(np.fft.rfft(arrays["decoder_weight"][:, mode]))
        axes[mode, 0].plot(enc_spectrum, color="C0")
        axes[mode, 0].set_title(f"mode {mode} — encoder |FFT|", fontsize=8)
        axes[mode, 1].plot(dec_spectrum, color="C1")
        axes[mode, 1].set_title(f"mode {mode} — decoder |FFT|", fontsize=8)
    figure.suptitle(f"{setting_label(dataset, horizon)}: rank-{rank} basis FFT", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = figures_dir / f"basis_fft_{dataset}_h{horizon}.png"
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for setting in SETTINGS:
        rows, arrays = analyze_setting(setting["dataset"], setting["horizon"])
        all_rows.extend(rows)
        plot_setting(setting["dataset"], setting["horizon"], arrays, figures_dir)
    csv_path = OUTPUT_ROOT / "lowrank_basis_fft_summary.csv"
    write_csv(all_rows, csv_path)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
