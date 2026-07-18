import csv
import json
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from scripts.ablate_phase_local_trend_ett import (
    ett_dates,
    make_model_config,
    variable_names,
)
from src.models.PhaseFormer import PhaseFormer


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stats(x):
    x = torch.as_tensor(x).float()
    diff = x[1:] - x[:-1]
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "amp": (x.max() - x.min()).item(),
        "start": x[0].item(),
        "end": x[-1].item(),
        "net": (x[-1] - x[0]).item(),
        "slope": (x[-1] - x[0]).item() / max(1, x.numel() - 1),
        "diff_abs_mean": diff.abs().mean().item() if diff.numel() else 0.0,
    }


def same_phase_corr(x, period_len):
    x = torch.as_tensor(x).float()
    n = (x.numel() // period_len) * period_len
    if n < period_len * 3:
        return float("nan")
    periods = x[-n:].view(-1, period_len)
    last = periods[-1] - periods[-1].mean()
    prev_template = periods[:-1].mean(dim=0)
    prev_template = prev_template - prev_template.mean()
    den = last.norm() * prev_template.norm()
    return (last @ prev_template / den).item() if den.item() != 0.0 else float("nan")


def load_model(dataset, horizon, variant, enable_phase_trend, prefix, args):
    run_dir = os.path.join(args.output_dir, f"{prefix}_{dataset.lower()}_{horizon}")
    _, _, exp_args, model_config, loaders = make_model_config(
        dataset,
        horizon,
        args,
        enable_phase_trend,
    )
    model = PhaseFormer(model_config)
    model.load_state_dict(torch.load(os.path.join(run_dir, f"{variant}.pt"), map_location="cpu"))
    model.eval()
    return model, exp_args, loaders[-2], loaders[-1]


def predict_one(model, batch_x, batch_y, batch_x_mark, batch_y_mark, use_head=None):
    old = model.use_phase_local_trend
    if use_head is not None:
        model.use_phase_local_trend = use_head
    with torch.inference_mode():
        dec_inp = model._build_decoder_input(batch_y.float())
        pred, _, _ = model(
            x_enc=batch_x.float(),
            x_mark_enc=batch_x_mark.float(),
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark.float(),
        )
    model.use_phase_local_trend = old
    return pred[:, -model.pred_len :, :].detach().cpu()


def extract_phase_head(model, batch_x, channel):
    with torch.inference_mode():
        x_in, stats_tuple = model.revin.normalize(batch_x.float())
        x = x_in.permute(0, 2, 1)
        if model.pad_seq_len > 0:
            x = torch.nn.functional.pad(x, (0, model.pad_seq_len), mode="circular")
        x_periods = x.view(x.size(0), x.size(1), model.num_periods_input, model.period_len)
        phase_series = model._to_phase_series(x_periods)
        if model.use_phase_uncertainty_shrinkage:
            phase_series = model.phase_uncertainty_shrinkage(phase_series)
        diagnostics = model.phase_local_trend.diagnostics(phase_series)
        correction = diagnostics["effective_gate"] * diagnostics["correction"]
        correction_periods = model._from_phase_steps_to_periods(correction)
        correction_seq_norm = correction_periods.reshape(
            x.size(0), x.size(1), -1
        )[..., : model.pred_len].permute(0, 2, 1)
        mu, sigma = stats_tuple
        correction_seq = correction_seq_norm * sigma

    return {
        "phase_series": phase_series[0, channel].detach().cpu(),
        "raw_phase_slope": diagnostics["raw_slope"][0, channel].detach().cpu(),
        "shape_phase_slope": diagnostics["shape_slope"][0, channel].detach().cpu(),
        "correction_phase_norm": correction[0, channel].detach().cpu(),
        "correction_seq": correction_seq[0, :, channel].detach().cpu(),
        "base_gate": diagnostics["base_gate"][0, channel, 0, 0].item(),
        "reliability": diagnostics["reliability"][0, channel, 0, 0].item(),
        "effective_gate": diagnostics["effective_gate"][0, channel, 0, 0].item(),
    }


def period_matrix(seq, period_len):
    n = (len(seq) // period_len) * period_len
    return torch.as_tensor(seq[:n]).float().view(-1, period_len) if n else torch.empty(0, period_len)


def plot_case(path, title, dates, input_values, true, baseline, trend_no_head, trend, phase_info, period_len):
    horizon = len(true)
    input_tail = min(288, len(input_values))
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), constrained_layout=True)

    ax = axes[0]
    x_in = list(range(-input_tail, 0))
    x_future = list(range(horizon))
    ax.plot(x_in, input_values[-input_tail:], color="#555555", linewidth=1.1, label="input tail")
    ax.plot(x_future, true, color="#111111", linewidth=1.5, label="true")
    ax.plot(x_future, baseline, color="#1f77b4", linewidth=1.2, label="baseline")
    ax.plot(x_future, trend_no_head, color="#ff7f0e", linewidth=1.2, linestyle="--", label="phase model, head off")
    ax.plot(x_future, trend, color="#d62728", linewidth=1.2, label="phase model, head on")
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("scaled value")
    ax.legend(loc="best", ncols=3, fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    direct = torch.as_tensor(trend) - torch.as_tensor(trend_no_head)
    ax.plot(x_future, direct, color="#d62728", label="direct head effect: on - off")
    ax.plot(x_future, torch.as_tensor(trend) - torch.as_tensor(true), color="#9467bd", alpha=0.75, label="phase-head error")
    ax.plot(x_future, torch.as_tensor(baseline) - torch.as_tensor(true), color="#1f77b4", alpha=0.75, label="baseline error")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_ylabel("delta / error")
    ax.legend(loc="best", ncols=3, fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[2]
    phase_series = phase_info["phase_series"]
    raw_slope = phase_info["raw_phase_slope"]
    shape_slope = phase_info["shape_phase_slope"]
    recent_count = min(3, phase_series.shape[-1])
    phase_axis = list(range(period_len))
    for i in range(recent_count):
        period_idx = phase_series.shape[-1] - recent_count + i
        ax.plot(phase_axis, phase_series[:, period_idx], linewidth=1.0, label=f"recent period {period_idx}")
    ax.plot(phase_axis, raw_slope, color="#d62728", linewidth=1.2, label="raw phase slope")
    ax.plot(phase_axis, shape_slope, color="#9467bd", linewidth=1.5, label="shape slope")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_ylabel("phase value / slope")
    ax.set_title(
        "Phase-local trend internals: "
        f"base_gate={phase_info['base_gate']:.4f}, "
        f"reliability={phase_info['reliability']:.4f}, "
        f"effective={phase_info['effective_gate']:.4f}"
    )
    ax.legend(loc="best", ncols=4, fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[3]
    correction_phase = phase_info["correction_phase_norm"]
    im = ax.imshow(correction_phase.numpy(), aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xlabel("future period index")
    ax.set_ylabel("phase slot")
    ax.set_title("Applied phase-local correction in RevIN-normalized phase space")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    prefix = "phase_local_trend_ett_full_20260716"
    args = SimpleNamespace(
        output_dir="research_runs",
        lookback=720,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        seed=2021,
        num_workers=0,
        phase_trend_window=3,
        phase_trend_gate_init=0.0,
        progress=False,
    )
    top_path = os.path.join(args.output_dir, f"{prefix}_overall_top10_worsened_samples.csv")
    top = pd.read_csv(top_path)
    output_dir = os.path.join(args.output_dir, f"{prefix}_top10_visual_analysis")
    os.makedirs(output_dir, exist_ok=True)

    cache = {}
    rows = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        dataset = str(row.dataset)
        horizon = int(row.horizon)
        sample_idx = int(row.sample_index)
        variable = str(row.worst_variable_name)
        key = (dataset, horizon)
        if key not in cache:
            baseline_model, exp_args, test_set, _ = load_model(
                dataset, horizon, "baseline", False, prefix, args
            )
            trend_model, _, _, _ = load_model(
                dataset, horizon, "phase_local_trend", True, prefix, args
            )
            cache[key] = (
                baseline_model,
                trend_model,
                exp_args,
                test_set,
                ett_dates(test_set, exp_args),
                variable_names(exp_args),
            )

        baseline_model, trend_model, exp_args, test_set, dates, names = cache[key]
        channel = names.index(variable)
        batch = test_set[sample_idx]
        batch_x, batch_y, batch_x_mark, batch_y_mark = [
            torch.as_tensor(item).unsqueeze(0) for item in batch
        ]
        baseline_pred = predict_one(baseline_model, batch_x, batch_y, batch_x_mark, batch_y_mark)
        trend_pred = predict_one(trend_model, batch_x, batch_y, batch_x_mark, batch_y_mark, use_head=True)
        trend_no_head = predict_one(
            trend_model, batch_x, batch_y, batch_x_mark, batch_y_mark, use_head=False
        )
        true = batch_y[:, -horizon:, :].float().detach().cpu()
        phase_info = extract_phase_head(trend_model, batch_x.float(), channel)

        inp = batch_x[0, :, channel].float().detach().cpu()
        y_true = true[0, :, channel]
        y_base = baseline_pred[0, :, channel]
        y_no_head = trend_no_head[0, :, channel]
        y_trend = trend_pred[0, :, channel]
        direct = y_trend - y_no_head
        corr_seq = phase_info["correction_seq"]

        s_in, s_true, s_base, s_no_head, s_trend, s_direct = [
            stats(values) for values in [inp, y_true, y_base, y_no_head, y_trend, direct]
        ]
        base_err = y_base - y_true
        trend_err = y_trend - y_true
        no_head_err = y_no_head - y_true
        direct_align = torch.dot(
            direct - direct.mean(),
            y_true - y_no_head - (y_true - y_no_head).mean(),
        )
        direct_align_den = (direct - direct.mean()).norm() * (
            y_true - y_no_head - (y_true - y_no_head).mean()
        ).norm()
        direct_target_corr = (
            (direct_align / direct_align_den).item() if direct_align_den.item() != 0.0 else float("nan")
        )

        metrics = {
            "rank": rank,
            "dataset": dataset,
            "horizon": horizon,
            "sample_index": sample_idx,
            "variable": variable,
            "forecast_start": str(dates.iloc[sample_idx + test_set.seq_len]),
            "forecast_end": str(dates.iloc[sample_idx + test_set.seq_len + horizon - 1]),
            "head_base_gate": phase_info["base_gate"],
            "head_reliability": phase_info["reliability"],
            "head_effective_gate": phase_info["effective_gate"],
            "input_same_phase_corr": same_phase_corr(inp, trend_model.period_len),
            "baseline_mse": torch.square(base_err).mean().item(),
            "phase_trend_mse": torch.square(trend_err).mean().item(),
            "phase_trend_no_head_mse": torch.square(no_head_err).mean().item(),
            "head_direct_mse_delta_vs_no_head": (
                torch.square(trend_err).mean() - torch.square(no_head_err).mean()
            ).item(),
            "baseline_bias": base_err.mean().item(),
            "phase_trend_bias": trend_err.mean().item(),
            "phase_trend_no_head_bias": no_head_err.mean().item(),
            "direct_effect_mean": s_direct["mean"],
            "direct_effect_net": s_direct["net"],
            "direct_effect_std": s_direct["std"],
            "correction_seq_mean": corr_seq.mean().item(),
            "correction_seq_std": corr_seq.std(unbiased=False).item(),
            "raw_phase_slope_mean": phase_info["raw_phase_slope"].mean().item(),
            "raw_phase_slope_std": phase_info["raw_phase_slope"].std(unbiased=False).item(),
            "raw_phase_slope_min": phase_info["raw_phase_slope"].min().item(),
            "raw_phase_slope_max": phase_info["raw_phase_slope"].max().item(),
            "shape_phase_slope_mean": phase_info["shape_phase_slope"].mean().item(),
            "shape_phase_slope_std": phase_info["shape_phase_slope"].std(unbiased=False).item(),
            "shape_phase_slope_min": phase_info["shape_phase_slope"].min().item(),
            "shape_phase_slope_max": phase_info["shape_phase_slope"].max().item(),
            "direct_target_residual_corr": direct_target_corr,
            "input_net": s_in["net"],
            "true_net": s_true["net"],
            "baseline_net": s_base["net"],
            "phase_trend_no_head_net": s_no_head["net"],
            "phase_trend_net": s_trend["net"],
            "true_amp": s_true["amp"],
            "baseline_amp": s_base["amp"],
            "phase_trend_no_head_amp": s_no_head["amp"],
            "phase_trend_amp": s_trend["amp"],
            "plot_path": os.path.join(
                output_dir, f"rank_{rank:02d}_{dataset}_{horizon}_{sample_idx}_{variable}.png"
            ),
        }
        rows.append(metrics)

        plot_case(
            metrics["plot_path"],
            f"Rank {rank}: {dataset}-{horizon} sample {sample_idx} var {variable}",
            dates,
            inp.numpy(),
            y_true.numpy(),
            y_base.numpy(),
            y_no_head.numpy(),
            y_trend.numpy(),
            phase_info,
            trend_model.period_len,
        )

        detail_path = os.path.join(
            output_dir, f"rank_{rank:02d}_{dataset}_{horizon}_{sample_idx}_{variable}.json"
        )
        with open(detail_path, "w") as f:
            serializable = dict(metrics)
            serializable["input_stats"] = s_in
            serializable["true_stats"] = s_true
            serializable["baseline_stats"] = s_base
            serializable["phase_trend_no_head_stats"] = s_no_head
            serializable["phase_trend_stats"] = s_trend
            serializable["direct_effect_stats"] = s_direct
            json.dump(serializable, f, indent=2)

    metrics_path = os.path.join(output_dir, "top10_visual_intermediate_metrics.csv")
    write_csv(metrics_path, rows, list(rows[0]))
    print(pd.DataFrame(rows)[
        [
            "rank",
            "dataset",
            "horizon",
            "sample_index",
            "variable",
            "head_base_gate",
            "head_reliability",
            "head_effective_gate",
            "input_same_phase_corr",
            "head_direct_mse_delta_vs_no_head",
            "direct_effect_mean",
            "baseline_bias",
            "phase_trend_no_head_bias",
            "phase_trend_bias",
            "direct_target_residual_corr",
            "plot_path",
        ]
    ].to_string(index=False))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
