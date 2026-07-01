import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytorch_lightning as pl
import torch
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import config.base_config as config_module
from src.dataset.data_factory import data_provider
from src.dataset.data_info import DATASET_INFO
from src.models.PhaseFormer import PhaseFormer


DEFAULT_NORM_HYPERS = dict(revin_affine=False, revin_eps=1e-5)


def get_best_config(dataset_name, horizon):
    if dataset_name == "Exchange":
        return {
            "layers": 2,
            "latent_dim": 8,
            "phase_encoder_hidden": 32,
            "predictor_hidden": 64,
            "phase_num_routers": 8,
            "learning_rate": 0.001,
            "phase_attn_heads": 1,
        }
    if dataset_name == "ETTh1":
        if horizon in [96, 192, 336]:
            return {
                "layers": 3,
                "latent_dim": 4,
                "phase_encoder_hidden": 16,
                "predictor_hidden": 32,
                "phase_num_routers": 8,
                "learning_rate": 0.001,
                "phase_attn_heads": 1,
            }
        return {
            "layers": 3,
            "latent_dim": 32,
            "phase_encoder_hidden": 128,
            "predictor_hidden": 256,
            "phase_num_routers": 16,
            "learning_rate": 0.00015,
            "phase_attn_heads": 2,
        }
    if dataset_name == "ETTh2":
        if horizon in [96, 192, 336]:
            return {
                "layers": 1,
                "latent_dim": 8,
                "phase_encoder_hidden": 32,
                "predictor_hidden": 64,
                "phase_num_routers": 8,
                "learning_rate": 0.001,
                "phase_attn_heads": 1,
            }
        return {
            "layers": 1,
            "latent_dim": 4,
            "phase_encoder_hidden": 8,
            "predictor_hidden": 8,
            "phase_num_routers": 4,
            "learning_rate": 0.001,
            "phase_attn_heads": 1,
        }
    if dataset_name == "ETTm1":
        return {
            "layers": 1 if horizon == 336 else 2,
            "latent_dim": 8,
            "phase_encoder_hidden": 32,
            "predictor_hidden": 64,
            "phase_num_routers": 8,
            "learning_rate": 0.001,
            "phase_attn_heads": 1,
        }
    if dataset_name == "ETTm2":
        return {
            "layers": 2 if horizon == 96 else 1,
            "latent_dim": 8,
            "phase_encoder_hidden": 32,
            "predictor_hidden": 64,
            "phase_num_routers": 8,
            "learning_rate": 0.001,
            "phase_attn_heads": 1,
        }
    if horizon == 96:
        return {
            "layers": 3,
            "latent_dim": 8,
            "phase_encoder_hidden": 32,
            "predictor_hidden": 64,
            "phase_num_routers": 8,
            "learning_rate": 0.001,
            "phase_attn_heads": 1,
        }
    return {
        "layers": 2,
        "latent_dim": 8,
        "phase_encoder_hidden": 32,
        "predictor_hidden": 64,
        "phase_num_routers": 8,
        "learning_rate": 0.001,
        "phase_attn_heads": 1,
    }


def get_frequency(dataset_name):
    if dataset_name == "Exchange":
        return "d"
    if dataset_name in ["ETTh1", "ETTh2"]:
        return "h"
    return "t"


class PhaseFormerConfig:
    def __init__(
        self,
        exp_args,
        best_config,
        lookback,
        horizon,
        variant,
        gate_init,
        time_mark_dim,
        period_len,
        phase_trend_window,
        phase_trend_gate_init,
        phase_jitter_gate_init,
        phase_reliability_min,
        phase_reliability_noise_threshold,
        phase_reliability_noise_temperature,
        phase_noise_hifreq_strength,
        phase_noise_hifreq_threshold,
        phase_noise_hifreq_temperature,
        phase_noise_hifreq_window,
        lowfreq_trend_window,
        lowfreq_trend_gate_init,
    ):
        self.seq_len = lookback
        self.pred_len = horizon
        self.enc_in = exp_args.model_args.num_variants
        self.period_len = period_len
        self.target_var_index = -1
        self.training_args = exp_args.training_args
        self.dataset_args = exp_args.dataset_args

        self.latent_dim = best_config["latent_dim"]
        self.phase_encoder_hidden = best_config["phase_encoder_hidden"]
        self.predictor_hidden = best_config["predictor_hidden"]
        self.phase_layers = best_config["layers"]
        self.phase_attn_heads = best_config["phase_attn_heads"]
        self.phase_attn_dropout = 0.1
        self.phase_attn_use_relpos = True
        self.phase_attn_window = None
        self.phase_attention_dim = None
        self.phase_num_routers = best_config["phase_num_routers"]
        self.phase_use_pos_embed = True
        self.phase_pos_dropout = 0.0

        self.use_revin = True
        self.revin_affine = DEFAULT_NORM_HYPERS["revin_affine"]
        self.revin_eps = DEFAULT_NORM_HYPERS["revin_eps"]
        self.use_huber_loss = exp_args.training_args.use_huber_loss
        self.huber_delta = exp_args.training_args.huber_delta

        self.use_weak_period_residual = variant in [
            "trend_residual",
            "phase_trend_residual",
            "adaptive_residual",
            "adaptive_phase_trend_residual",
            "channel_residual",
            "adaptive_channel_residual",
            "smooth_residual",
            "adaptive_smooth_residual",
            "phase_jitter_residual",
            "phase_jitter_smooth_residual",
            "phase_reliability_residual",
            "phase_reliability_smooth_residual",
            "phase_hifreq_residual",
            "phase_hifreq_smooth_residual",
            "lowfreq_trend_residual",
            "lowfreq_trend_smooth_residual",
        ]
        self.weak_period_residual_gate_init = gate_init
        self.weak_period_residual_head_type = (
            "channel"
            if variant in ["channel_residual", "adaptive_channel_residual"]
            else "lowpass"
            if variant
            in [
                "smooth_residual",
                "adaptive_smooth_residual",
                "phase_jitter_smooth_residual",
                "phase_reliability_smooth_residual",
                "phase_hifreq_smooth_residual",
                "lowfreq_trend_smooth_residual",
            ]
            else "shared"
        )
        self.weak_period_residual_smooth_window = best_config.get(
            "weak_period_residual_smooth_window", 25
        )
        self.use_adaptive_weak_period_gate = variant in [
            "adaptive_residual",
            "adaptive_phase_trend_residual",
            "adaptive_channel_residual",
            "adaptive_smooth_residual",
        ]
        self.adaptive_weak_period_gate_hidden = 8
        self.use_time_mark_adjustment = variant == "time_mark"
        self.time_mark_dim = time_mark_dim
        self.time_mark_hidden = 32
        self.use_phase_local_trend = variant in [
            "phase_trend",
            "phase_trend_residual",
            "adaptive_phase_trend_residual",
        ]
        self.phase_local_trend_window = phase_trend_window
        self.phase_local_trend_gate_init = phase_trend_gate_init
        self.use_phase_jitter_smoothing = variant in [
            "phase_jitter",
            "phase_jitter_residual",
            "phase_jitter_smooth_residual",
        ]
        self.phase_jitter_gate_init = phase_jitter_gate_init
        self.use_phase_reliability_damping = variant in [
            "phase_reliability",
            "phase_reliability_residual",
            "phase_reliability_smooth_residual",
        ]
        self.phase_reliability_min = phase_reliability_min
        self.phase_reliability_noise_threshold = phase_reliability_noise_threshold
        self.phase_reliability_noise_temperature = phase_reliability_noise_temperature
        self.use_phase_noise_hifreq_damping = variant in [
            "phase_hifreq",
            "phase_hifreq_residual",
            "phase_hifreq_smooth_residual",
        ]
        self.phase_noise_hifreq_strength = phase_noise_hifreq_strength
        self.phase_noise_hifreq_threshold = phase_noise_hifreq_threshold
        self.phase_noise_hifreq_temperature = phase_noise_hifreq_temperature
        self.phase_noise_hifreq_window = phase_noise_hifreq_window
        self.use_lowfreq_trend_correction = variant in [
            "lowfreq_trend",
            "lowfreq_trend_residual",
            "lowfreq_trend_smooth_residual",
        ]
        self.lowfreq_trend_window = lowfreq_trend_window
        self.lowfreq_trend_gate_init = lowfreq_trend_gate_init

    def get(self, key, default=None):
        return getattr(self, key, default)


def build_exp_args(
    dataset_name,
    lookback,
    horizon,
    epochs,
    batch_size,
    percent,
    lr,
    loss_func,
    use_huber_loss,
    huber_delta,
):
    exp_args = config_module.config
    exp_args.model_args.model = "PhaseFormer"
    exp_args.model_args.input_len = exp_args.dataset_args.seq_len = lookback
    exp_args.training_args.itr = 1
    exp_args.training_args.patience = min(8, max(2, epochs))
    exp_args.training_args.ema = False
    exp_args.training_args.train_epochs = epochs
    exp_args.training_args.lr_schedule_config.type = "type3"
    exp_args.training_args.loss_func = loss_func
    exp_args.training_args.use_huber_loss = use_huber_loss
    exp_args.training_args.huber_delta = huber_delta
    exp_args.training_args.learning_rate = lr
    exp_args.training_args.batch_size = batch_size

    exp_args.dataset_args.percent = percent
    exp_args.dataset_args.data = DATASET_INFO[dataset_name]["data"]
    exp_args.dataset_args.root_path = DATASET_INFO[dataset_name]["root_path"]
    exp_args.dataset_args.data_path = DATASET_INFO[dataset_name]["data_path"]
    exp_args.dataset_args.freq = get_frequency(dataset_name)
    exp_args.dataset_args.batch_size = batch_size
    exp_args.dataset_args.seq_len = lookback
    exp_args.dataset_args.pred_len = horizon
    exp_args.dataset_args.noisy_ratio = 0.0
    exp_args.dataset_args.var_needed = exp_args.model_args.num_variants = int(
        DATASET_INFO[dataset_name]["num_variants"]
    )
    return exp_args


def apply_overrides(best_config, args):
    best_config = dict(best_config)
    for key, value in {
        "layers": args.layers,
        "latent_dim": args.latent_dim,
        "phase_encoder_hidden": args.phase_encoder_hidden,
        "predictor_hidden": args.predictor_hidden,
        "phase_num_routers": args.phase_num_routers,
        "phase_attn_heads": args.phase_attn_heads,
        "learning_rate": args.learning_rate,
        "weak_period_residual_smooth_window": args.smooth_window,
    }.items():
        if value is not None:
            best_config[key] = value
    return best_config


def _dataset_variable_names(dataset):
    raw_path = os.path.join(dataset.root_path, dataset.data_path)
    columns = list(pd.read_csv(raw_path, nrows=0).columns[1:])
    if getattr(dataset, "var_needed", None) is not None:
        columns = columns[: int(dataset.var_needed)]
    return columns


def _dataset_dates(dataset):
    raw_path = os.path.join(dataset.root_path, dataset.data_path)
    dates = pd.to_datetime(pd.read_csv(raw_path, usecols=["date"])["date"])
    seq_len = int(dataset.seq_len)
    data_len = len(dataset.data_x)
    data_path = str(dataset.data_path)
    if data_path.startswith("ETTh"):
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
    elif data_path.startswith("ETTm"):
        border1s = [
            0,
            12 * 30 * 24 * 4 - seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
        ]
    else:
        num_train = int(len(dates) * 0.7)
        num_test = int(len(dates) * 0.2)
        border1s = [0, num_train - seq_len, len(dates) - num_test - seq_len]
    start = border1s[int(dataset.set_type)]
    return dates.iloc[start : start + data_len].reset_index(drop=True)


def _inverse_one(dataset, values, channel):
    values = values.detach().cpu()
    if not getattr(dataset, "scale", False):
        return values.numpy()
    channels = dataset.data_x.shape[-1]
    padded = torch.zeros(values.numel(), channels, dtype=values.dtype)
    padded[:, channel] = values.reshape(-1)
    restored = dataset.inverse_transform(padded.numpy())
    return restored[:, channel].reshape(values.shape)


def _case_pattern_metrics(pred, true, inp):
    err = pred - true
    pred_slope = pred[-1] - pred[0]
    true_slope = true[-1] - true[0]
    return {
        "mse": torch.square(err).mean(),
        "mae": torch.abs(err).mean(),
        "bias_abs": err.mean().abs(),
        "trend_mismatch": (pred_slope - true_slope).abs(),
        "peak_under": (true.max() - pred.max()).clamp_min(0.0),
        "valley_over": (pred.min() - true.min()).clamp_min(0.0),
        "volatility_mismatch": (pred.std(unbiased=False) - true.std(unbiased=False)).abs(),
        "late_mse": torch.square(err[-max(1, err.numel() // 4) :]).mean(),
        "input_volatility": inp.diff().abs().mean() if inp.numel() > 1 else torch.tensor(0.0),
    }


def collect_bad_cases(model, loader, pred_len, bad_case_limit, max_batches, run_dir=None):
    model.eval()
    device = next(model.parameters()).device
    dataset = loader.dataset
    var_names = _dataset_variable_names(dataset)
    dates = _dataset_dates(dataset)
    candidates = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec_inp = model._build_decoder_input(batch_y.float())
            outputs, _, _ = model(
                x_enc=batch_x.float(),
                x_mark_enc=batch_x_mark.float(),
                x_dec=dec_inp,
                x_mark_dec=batch_y_mark.float(),
            )
            pred = outputs[:, -pred_len:, :]
            true = batch_y.float()[:, -pred_len:, :]
            for local_idx in range(pred.size(0)):
                global_idx = batch_idx * loader.batch_size + local_idx
                for channel in range(pred.size(-1)):
                    metrics = _case_pattern_metrics(
                        pred[local_idx, :, channel].detach().cpu(),
                        true[local_idx, :, channel].detach().cpu(),
                        batch_x[local_idx, :, channel].detach().cpu(),
                    )
                    row = {
                        "batch_idx": batch_idx,
                        "sample_in_batch": int(local_idx),
                        "global_sample_index": int(global_idx),
                        "variable_index": int(channel),
                        "variable_name": var_names[channel] if channel < len(var_names) else str(channel),
                        "input_start": str(dates.iloc[global_idx]),
                        "input_end": str(dates.iloc[global_idx + loader.dataset.seq_len - 1]),
                        "forecast_start": str(dates.iloc[global_idx + loader.dataset.seq_len]),
                        "forecast_end": str(
                            dates.iloc[global_idx + loader.dataset.seq_len + pred_len - 1]
                        ),
                    }
                    row.update({key: float(value) for key, value in metrics.items()})
                    row["_pred"] = pred[local_idx, :, channel].detach().cpu()
                    row["_true"] = true[local_idx, :, channel].detach().cpu()
                    row["_input"] = batch_x[local_idx, :, channel].detach().cpu()
                    candidates.append(row)

    pattern_order = [
        ("highest_mse", "mse"),
        ("systematic_bias", "bias_abs"),
        ("trend_mismatch", "trend_mismatch"),
        ("peak_underfit", "peak_under"),
        ("valley_overfit", "valley_over"),
        ("volatility_mismatch", "volatility_mismatch"),
        ("late_horizon_drift", "late_mse"),
        ("volatile_input", "input_volatility"),
    ]
    selected = []
    seen = set()
    for pattern, key in pattern_order:
        for row in sorted(candidates, key=lambda item: item[key], reverse=True):
            identity = (row["global_sample_index"], row["variable_index"])
            if identity in seen:
                continue
            row = dict(row)
            row["error_pattern"] = pattern
            selected.append(row)
            seen.add(identity)
            break
        if len(selected) >= bad_case_limit:
            break

    if run_dir:
        windows_dir = os.path.join(run_dir, "bad_cases")
        os.makedirs(windows_dir, exist_ok=True)
        for case_id, row in enumerate(selected):
            global_idx = row["global_sample_index"]
            channel = row["variable_index"]
            pred_values = row.pop("_pred")
            true_values = row.pop("_true")
            input_values = row.pop("_input")
            input_original = _inverse_one(dataset, input_values, channel)
            true_original = _inverse_one(dataset, true_values, channel)
            pred_original = _inverse_one(dataset, pred_values, channel)
            window_rows = []
            for offset, value in enumerate(input_values.tolist()):
                date = dates.iloc[global_idx + offset]
                window_rows.append(
                    {
                        "segment": "input",
                        "offset": offset,
                        "date": str(date),
                        "value_scaled": value,
                        "value_original": float(input_original[offset]),
                        "prediction_scaled": "",
                        "prediction_original": "",
                        "error_scaled": "",
                    }
                )
            for offset, (pred_value, true_value) in enumerate(
                zip(pred_values.tolist(), true_values.tolist())
            ):
                date = dates.iloc[global_idx + dataset.seq_len + offset]
                window_rows.append(
                    {
                        "segment": "forecast",
                        "offset": offset,
                        "date": str(date),
                        "value_scaled": true_value,
                        "value_original": float(true_original[offset]),
                        "prediction_scaled": pred_value,
                        "prediction_original": float(pred_original[offset]),
                        "error_scaled": pred_value - true_value,
                    }
                )
            write_csv(
                os.path.join(windows_dir, f"case_{case_id:02d}_{row['error_pattern']}.csv"),
                window_rows,
                [
                    "segment",
                    "offset",
                    "date",
                    "value_scaled",
                    "value_original",
                    "prediction_scaled",
                    "prediction_original",
                    "error_scaled",
                ],
            )
    else:
        for row in selected:
            row.pop("_pred", None)
            row.pop("_true", None)
            row.pop("_input", None)

    return selected


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="Weather",
        choices=["Weather", "Exchange", "ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    )
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--period-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--variant",
        choices=[
            "baseline",
            "trend_residual",
            "time_mark",
            "phase_trend",
            "phase_trend_residual",
            "adaptive_residual",
            "adaptive_phase_trend_residual",
            "channel_residual",
            "adaptive_channel_residual",
            "smooth_residual",
            "adaptive_smooth_residual",
            "phase_jitter",
            "phase_jitter_residual",
            "phase_jitter_smooth_residual",
            "phase_reliability",
            "phase_reliability_residual",
            "phase_reliability_smooth_residual",
            "phase_hifreq",
            "phase_hifreq_residual",
            "phase_hifreq_smooth_residual",
            "lowfreq_trend",
            "lowfreq_trend_residual",
            "lowfreq_trend_smooth_residual",
        ],
        default="baseline",
    )
    parser.add_argument("--gate-init", type=float, default=0.2)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--phase-trend-window", type=int, default=3)
    parser.add_argument("--phase-trend-gate-init", type=float, default=0.1)
    parser.add_argument("--phase-jitter-gate-init", type=float, default=0.1)
    parser.add_argument("--phase-reliability-min", type=float, default=0.35)
    parser.add_argument("--phase-reliability-noise-threshold", type=float, default=0.0)
    parser.add_argument("--phase-reliability-noise-temperature", type=float, default=0.2)
    parser.add_argument("--phase-noise-hifreq-strength", type=float, default=0.5)
    parser.add_argument("--phase-noise-hifreq-threshold", type=float, default=1.0)
    parser.add_argument("--phase-noise-hifreq-temperature", type=float, default=0.2)
    parser.add_argument("--phase-noise-hifreq-window", type=int, default=7)
    parser.add_argument("--lowfreq-trend-window", type=int, default=25)
    parser.add_argument("--lowfreq-trend-gate-init", type=float, default=0.05)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--bad-case-limit", type=int, default=10)
    parser.add_argument("--bad-case-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--phase-encoder-hidden", type=int, default=None)
    parser.add_argument("--predictor-hidden", type=int, default=None)
    parser.add_argument("--phase-num-routers", type=int, default=None)
    parser.add_argument("--phase-attn-heads", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--loss-func", choices=["mse", "mae", "smae"], default="mse")
    parser.add_argument("--disable-huber", action="store_true")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    best_config = apply_overrides(get_best_config(args.dataset, args.horizon), args)
    exp_args = build_exp_args(
        args.dataset,
        args.lookback,
        args.horizon,
        args.epochs,
        args.batch_size,
        args.percent,
        best_config["learning_rate"],
        args.loss_func,
        not args.disable_huber,
        args.huber_delta,
    )
    exp_args.dataset_args.num_workers = args.num_workers
    exp_args.training_args.num_workers = args.num_workers

    run_id = args.run_id or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.dataset.lower()}_"
        f"{args.lookback}_{args.horizon}_{args.variant}_seed{args.seed}"
    )
    run_dir = os.path.join("research_runs", run_id)
    os.makedirs(run_dir, exist_ok=False)

    with open(os.path.join(run_dir, "commands.sh"), "w") as f:
        f.write(" ".join(["python", "scripts/research_weather_weak.py"] + os.sys.argv[1:]))
        f.write("\n")

    train_set, train_loader = data_provider(exp_args.dataset_args, "train")
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")
    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    time_mark_dim = int(train_set.data_stamp.shape[-1])

    model_config = PhaseFormerConfig(
        exp_args,
        best_config,
        args.lookback,
        args.horizon,
        args.variant,
        args.gate_init,
        time_mark_dim,
        args.period_len,
        args.phase_trend_window,
        args.phase_trend_gate_init,
        args.phase_jitter_gate_init,
        args.phase_reliability_min,
        args.phase_reliability_noise_threshold,
        args.phase_reliability_noise_temperature,
        args.phase_noise_hifreq_strength,
        args.phase_noise_hifreq_threshold,
        args.phase_noise_hifreq_temperature,
        args.phase_noise_hifreq_window,
        args.lowfreq_trend_window,
        args.lowfreq_trend_gate_init,
    )
    config_snapshot = {
        "args": vars(args),
        "dataset": {
            "root_path": exp_args.dataset_args.root_path,
            "data_path": exp_args.dataset_args.data_path,
            "features": exp_args.dataset_args.features,
            "num_variants": exp_args.model_args.num_variants,
        },
        "model": {
            "period_len": model_config.period_len,
            "phase_layers": model_config.phase_layers,
            "latent_dim": model_config.latent_dim,
            "phase_num_routers": model_config.phase_num_routers,
            "phase_attn_heads": model_config.phase_attn_heads,
            "use_weak_period_residual": model_config.use_weak_period_residual,
            "weak_period_residual_gate_init": model_config.weak_period_residual_gate_init,
            "weak_period_residual_head_type": model_config.weak_period_residual_head_type,
            "weak_period_residual_smooth_window": model_config.weak_period_residual_smooth_window,
            "use_adaptive_weak_period_gate": model_config.use_adaptive_weak_period_gate,
            "adaptive_weak_period_gate_hidden": model_config.adaptive_weak_period_gate_hidden,
            "use_time_mark_adjustment": model_config.use_time_mark_adjustment,
            "time_mark_dim": model_config.time_mark_dim,
            "time_mark_hidden": model_config.time_mark_hidden,
            "use_phase_local_trend": model_config.use_phase_local_trend,
            "phase_local_trend_window": model_config.phase_local_trend_window,
            "phase_local_trend_gate_init": model_config.phase_local_trend_gate_init,
            "use_phase_jitter_smoothing": model_config.use_phase_jitter_smoothing,
            "phase_jitter_gate_init": model_config.phase_jitter_gate_init,
            "use_phase_reliability_damping": model_config.use_phase_reliability_damping,
            "phase_reliability_min": model_config.phase_reliability_min,
            "phase_reliability_noise_threshold": model_config.phase_reliability_noise_threshold,
            "phase_reliability_noise_temperature": model_config.phase_reliability_noise_temperature,
            "use_phase_noise_hifreq_damping": model_config.use_phase_noise_hifreq_damping,
            "phase_noise_hifreq_strength": model_config.phase_noise_hifreq_strength,
            "phase_noise_hifreq_threshold": model_config.phase_noise_hifreq_threshold,
            "phase_noise_hifreq_temperature": model_config.phase_noise_hifreq_temperature,
            "phase_noise_hifreq_window": model_config.phase_noise_hifreq_window,
            "use_lowfreq_trend_correction": model_config.use_lowfreq_trend_correction,
            "lowfreq_trend_window": model_config.lowfreq_trend_window,
            "lowfreq_trend_gate_init": model_config.lowfreq_trend_gate_init,
        },
        "training": {
            "learning_rate": exp_args.training_args.learning_rate,
            "loss_func": exp_args.training_args.loss_func,
            "use_huber_loss": exp_args.training_args.use_huber_loss,
            "huber_delta": exp_args.training_args.huber_delta,
            "patience": exp_args.training_args.patience,
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    model = PhaseFormer(model_config)
    logger = CSVLogger(
        save_dir=os.path.join(run_dir, "lightning"),
        name="PhaseFormer",
        version=args.variant,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=exp_args.training_args.patience)],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        log_every_n_steps=1,
        deterministic=True,
    )

    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    elapsed = time.time() - start
    test_metrics = test_result[0] if test_result else {}
    bad_cases = collect_bad_cases(
        model,
        test_loader,
        args.horizon,
        args.bad_case_limit,
        args.bad_case_batches,
        run_dir,
    )

    metrics_row = {
        "run_id": run_id,
        "dataset": args.dataset,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "variant": args.variant,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": int(trainer.current_epoch),
        "test_mae": float(test_metrics.get("test_mae", float("nan"))),
        "test_mse": float(test_metrics.get("test_mse", float("nan"))),
        "elapsed_sec": elapsed,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
    }
    write_csv(os.path.join(run_dir, "metrics.csv"), [metrics_row], list(metrics_row))
    write_csv(
        os.path.join(run_dir, "bad_cases.csv"),
        bad_cases,
        [
            "batch_idx",
            "sample_in_batch",
            "global_sample_index",
            "variable_index",
            "variable_name",
            "input_start",
            "input_end",
            "forecast_start",
            "forecast_end",
            "error_pattern",
            "mse",
            "mae",
            "bias_abs",
            "trend_mismatch",
            "peak_under",
            "valley_over",
            "volatility_mismatch",
            "late_mse",
            "input_volatility",
        ],
    )

    runtime = {
        "elapsed_sec": elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(run_dir, "runtime.md"), "w") as f:
        for key, value in runtime.items():
            f.write(f"- {key}: {value}\n")

    print(json.dumps(metrics_row, indent=2))


if __name__ == "__main__":
    main()
