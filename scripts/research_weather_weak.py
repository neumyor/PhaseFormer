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
            "layers": 2 if horizon == 336 else 1,
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
        ]
        self.weak_period_residual_gate_init = gate_init
        self.weak_period_residual_head_type = (
            "channel"
            if variant in ["channel_residual", "adaptive_channel_residual"]
            else "lowpass"
            if variant
            in ["smooth_residual", "adaptive_smooth_residual", "phase_jitter_smooth_residual"]
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


def collect_bad_cases(model, loader, pred_len, bad_case_limit, max_batches):
    model.eval()
    device = next(model.parameters()).device
    bad_cases = []

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
            err = pred - true

            sample_mse = torch.square(err).mean(dim=(1, 2)).detach().cpu()
            topk = min(bad_case_limit, sample_mse.numel())
            values, indices = torch.topk(sample_mse, k=topk)
            for value, local_idx in zip(values.tolist(), indices.tolist()):
                bad_cases.append(
                    {
                        "batch_idx": batch_idx,
                        "sample_in_batch": int(local_idx),
                        "sample_mse": float(value),
                        "sample_mae": float(
                            torch.abs(err[local_idx]).mean().detach().cpu()
                        ),
                    }
                )

    bad_cases = sorted(bad_cases, key=lambda row: row["sample_mse"], reverse=True)[
        :bad_case_limit
    ]
    return bad_cases


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
        ],
        default="baseline",
    )
    parser.add_argument("--gate-init", type=float, default=0.2)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--phase-trend-window", type=int, default=3)
    parser.add_argument("--phase-trend-gate-init", type=float, default=0.1)
    parser.add_argument("--phase-jitter-gate-init", type=float, default=0.1)
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
        ["batch_idx", "sample_in_batch", "sample_mse", "sample_mae"],
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
