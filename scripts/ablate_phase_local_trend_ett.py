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

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.training.runner import restore_best_checkpoint
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    get_dataset_horizons,
    make_exp_args,
)


ETT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_horizons(value):
    if value == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def make_model_config(dataset, horizon, args, enable_phase_trend):
    hyperparams = build_hyperparams(dataset, horizon, "original")
    if args.epochs is not None:
        hyperparams["train_epochs"] = args.epochs
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if enable_phase_trend:
        hyperparams["scheme_name"] = "phase_local_trend"
        hyperparams["use_phase_local_trend"] = True
        hyperparams["phase_local_trend_window"] = args.phase_trend_window
        hyperparams["phase_local_trend_gate_init"] = args.phase_trend_gate_init
    else:
        hyperparams["scheme_name"] = "baseline"

    seed = int(args.seed if args.seed is not None else hyperparams.get("seed", 2021))
    exp_args = make_exp_args(
        dataset,
        args.lookback,
        horizon,
        hyperparams,
        batch_size=args.batch_size,
    )
    exp_args.dataset_args.num_workers = args.num_workers
    exp_args.training_args.num_workers = args.num_workers

    train_set, train_loader = data_provider(exp_args.dataset_args, "train")
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")
    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    if hasattr(train_set, "data_stamp"):
        hyperparams["time_mark_dim"] = int(train_set.data_stamp.shape[-1])

    model_config = PhaseFormerPresetConfig(
        exp_args,
        args.lookback,
        horizon,
        hyperparams,
    )
    return hyperparams, seed, exp_args, model_config, (train_set, train_loader, val_loader, test_set, test_loader)


def train_or_load_model(dataset, horizon, variant, enable_phase_trend, run_dir, args):
    hyperparams, seed, exp_args, model_config, loaders = make_model_config(
        dataset, horizon, args, enable_phase_trend
    )
    train_set, train_loader, val_loader, test_set, test_loader = loaders
    ckpt_path = os.path.join(run_dir, f"{variant}.pt")
    metrics_path = os.path.join(run_dir, f"{variant}_metrics.csv")

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    model = PhaseFormer(model_config)

    if args.resume and os.path.exists(ckpt_path) and os.path.exists(metrics_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        metrics = read_csv(metrics_path)[0]
        return model, metrics, test_loader, test_set, exp_args, hyperparams

    logger = CSVLogger(
        save_dir=os.path.join(run_dir, "lightning"),
        name="PhaseFormer",
        version=variant,
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints", variant),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=exp_args.training_args.train_epochs,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=exp_args.training_args.patience),
            checkpoint,
        ],
        accelerator="auto",
        devices=1,
        enable_progress_bar=args.progress,
        log_every_n_steps=1,
        deterministic=True,
    )

    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Restore the lowest-val-loss checkpoint manually: Lightning 2.1's test()
    # loads checkpoints via torch.load with the torch>=2.6 default
    # weights_only=True, which rejects the bundled model config. The shared
    # restore helper passes weights_only=False explicitly.
    restore_best_checkpoint(model, checkpoint)
    test_result = trainer.test(
        model,
        dataloaders=test_loader,
        verbose=False,
    )
    elapsed = time.time() - start
    test_metrics = test_result[0] if test_result else {}
    metrics = {
        "dataset": dataset,
        "horizon": horizon,
        "variant": variant,
        "seed": seed,
        "epochs_requested": exp_args.training_args.train_epochs,
        "epochs_completed": int(trainer.current_epoch),
        "learning_rate": exp_args.training_args.learning_rate,
        "loss_func": exp_args.training_args.loss_func,
        "use_huber_loss": exp_args.training_args.use_huber_loss,
        "test_mae": float(test_metrics.get("test_mae", float("nan"))),
        "test_mse": float(test_metrics.get("test_mse", float("nan"))),
        "elapsed_sec": elapsed,
        "train_size": len(train_set),
        "test_size": len(test_set),
    }
    torch.save(model.state_dict(), ckpt_path)
    write_csv(metrics_path, [metrics], list(metrics))
    return model, metrics, test_loader, test_set, exp_args, hyperparams


def ett_dates(dataset, exp_args):
    raw = pd.read_csv(os.path.join(exp_args.dataset_args.root_path, exp_args.dataset_args.data_path))
    dates = pd.to_datetime(raw["date"])
    seq_len = exp_args.dataset_args.seq_len
    data_path = exp_args.dataset_args.data_path
    if data_path.startswith("ETTh"):
        test_start = 12 * 30 * 24 + 4 * 30 * 24 - seq_len
    else:
        test_start = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len
    return dates.iloc[test_start : test_start + len(dataset.data_x)].reset_index(drop=True)


def variable_names(exp_args):
    raw = pd.read_csv(os.path.join(exp_args.dataset_args.root_path, exp_args.dataset_args.data_path), nrows=1)
    return list(raw.columns[1 : 1 + int(exp_args.model_args.num_variants)])


def predict_all(model, loader, pred_len):
    model.eval()
    device = next(model.parameters()).device
    preds = []
    trues = []
    with torch.inference_mode():
        for batch in loader:
            batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec_inp = model._build_decoder_input(batch_y.float())
            outputs, _, _ = model(
                x_enc=batch_x.float(),
                x_mark_enc=batch_x_mark.float(),
                x_dec=dec_inp,
                x_mark_dec=batch_y_mark.float(),
            )
            preds.append(outputs[:, -pred_len:, :].detach().cpu())
            trues.append(batch_y.float()[:, -pred_len:, :].detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def collect_pair_errors(dataset, horizon, run_dir, args):
    baseline_model, baseline_metrics, test_loader, test_set, exp_args, _ = train_or_load_model(
        dataset, horizon, "baseline", False, run_dir, args
    )
    trend_model, trend_metrics, _, _, _, _ = train_or_load_model(
        dataset, horizon, "phase_local_trend", True, run_dir, args
    )

    baseline_pred, true = predict_all(baseline_model, test_loader, horizon)
    trend_pred, true_again = predict_all(trend_model, test_loader, horizon)
    if not torch.allclose(true, true_again):
        raise RuntimeError("Baseline and phase-trend test targets are not aligned")

    dates = ett_dates(test_set, exp_args)
    names = variable_names(exp_args)
    baseline_err = baseline_pred - true
    trend_err = trend_pred - true
    baseline_sq = torch.square(baseline_err)
    trend_sq = torch.square(trend_err)
    baseline_abs = torch.abs(baseline_err)
    trend_abs = torch.abs(trend_err)

    per_variable_rows = []
    per_sample_rows = []
    top_variable_for_sample = {}
    for sample_idx in range(true.size(0)):
        var_deltas = []
        for channel in range(true.size(2)):
            base_mse = baseline_sq[sample_idx, :, channel].mean().item()
            trend_mse = trend_sq[sample_idx, :, channel].mean().item()
            base_mae = baseline_abs[sample_idx, :, channel].mean().item()
            trend_mae = trend_abs[sample_idx, :, channel].mean().item()
            row = {
                "dataset": dataset,
                "horizon": horizon,
                "sample_index": sample_idx,
                "variable_index": channel,
                "variable_name": names[channel] if channel < len(names) else str(channel),
                "input_start": str(dates.iloc[sample_idx]),
                "input_end": str(dates.iloc[sample_idx + test_set.seq_len - 1]),
                "forecast_start": str(dates.iloc[sample_idx + test_set.seq_len]),
                "forecast_end": str(dates.iloc[sample_idx + test_set.seq_len + horizon - 1]),
                "baseline_mse": base_mse,
                "phase_trend_mse": trend_mse,
                "mse_delta": trend_mse - base_mse,
                "baseline_mae": base_mae,
                "phase_trend_mae": trend_mae,
                "mae_delta": trend_mae - base_mae,
            }
            per_variable_rows.append(row)
            var_deltas.append(row)
        worst_var = max(var_deltas, key=lambda row: row["mse_delta"])
        top_variable_for_sample[sample_idx] = worst_var

        base_mse = baseline_sq[sample_idx].mean().item()
        trend_mse = trend_sq[sample_idx].mean().item()
        base_mae = baseline_abs[sample_idx].mean().item()
        trend_mae = trend_abs[sample_idx].mean().item()
        per_sample_rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "sample_index": sample_idx,
                "input_start": str(dates.iloc[sample_idx]),
                "input_end": str(dates.iloc[sample_idx + test_set.seq_len - 1]),
                "forecast_start": str(dates.iloc[sample_idx + test_set.seq_len]),
                "forecast_end": str(dates.iloc[sample_idx + test_set.seq_len + horizon - 1]),
                "baseline_mse": base_mse,
                "phase_trend_mse": trend_mse,
                "mse_delta": trend_mse - base_mse,
                "mse_delta_pct": (trend_mse - base_mse) / base_mse * 100.0 if base_mse else float("nan"),
                "baseline_mae": base_mae,
                "phase_trend_mae": trend_mae,
                "mae_delta": trend_mae - base_mae,
                "mae_delta_pct": (trend_mae - base_mae) / base_mae * 100.0 if base_mae else float("nan"),
                "worst_variable_index": worst_var["variable_index"],
                "worst_variable_name": worst_var["variable_name"],
                "worst_variable_mse_delta": worst_var["mse_delta"],
            }
        )

    top10_rows = sorted(per_sample_rows, key=lambda row: row["mse_delta"], reverse=True)[:10]
    write_csv(os.path.join(run_dir, "per_sample_errors.csv"), per_sample_rows, list(per_sample_rows[0]))
    write_csv(
        os.path.join(run_dir, "per_sample_variable_errors.csv"),
        per_variable_rows,
        list(per_variable_rows[0]),
    )
    write_csv(os.path.join(run_dir, "top10_worsened_samples.csv"), top10_rows, list(top10_rows[0]))

    summary = {
        "dataset": dataset,
        "horizon": horizon,
        "baseline_test_mae": baseline_metrics["test_mae"],
        "phase_trend_test_mae": trend_metrics["test_mae"],
        "test_mae_delta": float(trend_metrics["test_mae"]) - float(baseline_metrics["test_mae"]),
        "baseline_test_mse": baseline_metrics["test_mse"],
        "phase_trend_test_mse": trend_metrics["test_mse"],
        "test_mse_delta": float(trend_metrics["test_mse"]) - float(baseline_metrics["test_mse"]),
        "num_test_samples": len(per_sample_rows),
        "num_worsened_samples": sum(1 for row in per_sample_rows if row["mse_delta"] > 0.0),
        "mean_sample_mse_delta": sum(row["mse_delta"] for row in per_sample_rows) / len(per_sample_rows),
        "top_worsened_sample_index": top10_rows[0]["sample_index"] if top10_rows else "",
        "top_worsened_mse_delta": top10_rows[0]["mse_delta"] if top10_rows else "",
    }
    write_csv(os.path.join(run_dir, "pair_summary.csv"), [summary], list(summary))
    return summary


def summarize(output_dir, run_prefix, rows):
    summary_path = os.path.join(output_dir, f"{run_prefix}_summary.csv")
    fieldnames = list(rows[0]) if rows else [
        "dataset",
        "horizon",
        "baseline_test_mae",
        "phase_trend_test_mae",
        "test_mae_delta",
        "baseline_test_mse",
        "phase_trend_test_mse",
        "test_mse_delta",
        "num_test_samples",
        "num_worsened_samples",
        "mean_sample_mse_delta",
        "top_worsened_sample_index",
        "top_worsened_mse_delta",
    ]
    write_csv(summary_path, rows, fieldnames)
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(ETT_DATASETS))
    parser.add_argument("--horizons", default="all")
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--phase-trend-window", type=int, default=3)
    parser.add_argument("--phase-trend-gate-init", type=float, default=0.0)
    parser.add_argument("--output-dir", default="research_runs")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.run_prefix is None:
        args.run_prefix = f"phase_local_trend_ett_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    datasets = parse_csv_list(args.datasets)
    horizons_arg = parse_horizons(args.horizons)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for dataset in datasets:
        if dataset not in ETT_DATASETS:
            raise ValueError(f"Only ETT datasets are supported, got {dataset}")
        horizons = horizons_arg or get_dataset_horizons(dataset)
        for horizon in horizons:
            run_id = f"{args.run_prefix}_{dataset.lower()}_{horizon}"
            run_dir = os.path.join(args.output_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(vars(args) | {"dataset": dataset, "horizon": horizon}, f, indent=2)
            with open(os.path.join(run_dir, "runtime.md"), "w") as f:
                f.write(f"- python: {platform.python_version()}\n")
                f.write(f"- torch: {torch.__version__}\n")
                f.write(f"- cuda_available: {torch.cuda.is_available()}\n")
                f.write(f"- cuda_device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\n")
            summary_path = os.path.join(run_dir, "pair_summary.csv")
            if args.resume and os.path.exists(summary_path):
                summary = read_csv(summary_path)[0]
            else:
                summary = collect_pair_errors(dataset, horizon, run_dir, args)
            rows.append(summary)
            print(json.dumps(summary, indent=2))
            print(f"Current summary: {summarize(args.output_dir, args.run_prefix, rows)}")

    print(f"Final summary: {summarize(args.output_dir, args.run_prefix, rows)}")


if __name__ == "__main__":
    main()
