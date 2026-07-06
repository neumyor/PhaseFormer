import argparse
import csv
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)


DEFAULT_HORIZONS = [96, 192, 336, 720]


def parse_horizons(value):
    if value == "all":
        return list(DEFAULT_HORIZONS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_summary(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "dataset",
        "mode",
        "scheme",
        "lookback",
        "horizon",
        "seed",
        "epochs_requested",
        "epochs_completed",
        "learning_rate",
        "loss_func",
        "use_huber_loss",
        "test_mae",
        "test_mse",
        "log_dir",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(
    dataset_name,
    horizons=None,
    mode="latest",
    lookback=720,
    batch_size=None,
    num_workers=0,
    progress=True,
):
    torch.set_float32_matmul_precision("medium")
    horizons = list(DEFAULT_HORIZONS if horizons is None else horizons)
    summary_rows = []

    for horizon in horizons:
        hyperparams = build_hyperparams(dataset_name, horizon, mode)
        seed = int(hyperparams.get("seed", 2021))
        pl.seed_everything(seed, workers=True)

        exp_args = make_exp_args(
            dataset_name,
            lookback,
            horizon,
            hyperparams,
            batch_size=batch_size,
        )
        exp_args.dataset_args.num_workers = num_workers
        exp_args.training_args.num_workers = num_workers

        train_set, train_loader = data_provider(exp_args.dataset_args, "train")
        _, val_loader = data_provider(exp_args.dataset_args, "val")
        _, test_loader = data_provider(exp_args.dataset_args, "test")
        if hasattr(train_set, "data_stamp"):
            hyperparams["time_mark_dim"] = int(train_set.data_stamp.shape[-1])

        model_config = PhaseFormerPresetConfig(
            exp_args,
            lookback,
            horizon,
            hyperparams,
        )
        model = PhaseFormer(model_config)

        scheme = hyperparams.get("scheme_name", mode)
        logger_version = (
            f"{dataset_name}-{lookback}-{horizon}-PhaseFormer-{mode}-{scheme}"
            f"-seed{seed}"
        )
        logger = CSVLogger(
            save_dir="./log/training_results",
            name="PhaseFormer",
            version=logger_version,
        )

        print(f"\n{'=' * 72}")
        print(
            f"{dataset_name} {lookback}->{horizon} mode={mode} scheme={scheme} "
            f"seed={seed} lr={exp_args.training_args.learning_rate} "
            f"loss={exp_args.training_args.loss_func}"
        )
        print(f"{'=' * 72}")

        trainer = pl.Trainer(
            max_epochs=exp_args.training_args.train_epochs,
            logger=logger,
            enable_checkpointing=True,
            callbacks=[EarlyStopping(monitor="val_loss", patience=exp_args.training_args.patience)],
            accelerator="auto",
            devices=1,
            enable_progress_bar=progress,
            log_every_n_steps=1,
            deterministic=True,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        test_result = trainer.test(model, dataloaders=test_loader, verbose=True)
        metrics = test_result[0] if test_result else {}

        summary_rows.append(
            {
                "dataset": dataset_name,
                "mode": mode,
                "scheme": scheme,
                "lookback": lookback,
                "horizon": horizon,
                "seed": seed,
                "epochs_requested": exp_args.training_args.train_epochs,
                "epochs_completed": int(trainer.current_epoch),
                "learning_rate": exp_args.training_args.learning_rate,
                "loss_func": exp_args.training_args.loss_func,
                "use_huber_loss": exp_args.training_args.use_huber_loss,
                "test_mae": float(metrics.get("test_mae", float("nan"))),
                "test_mse": float(metrics.get("test_mse", float("nan"))),
                "log_dir": logger.log_dir,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        "./log/training_results/PhaseFormer",
        f"summary_{dataset_name.lower()}_{mode}_{timestamp}.csv",
    )
    write_summary(summary_path, summary_rows)
    print(f"\nSummary written to: {summary_path}")
    for row in summary_rows:
        print(
            f"{row['dataset']} {row['lookback']}->{row['horizon']} "
            f"{row['scheme']}: MAE={row['test_mae']:.6f}, MSE={row['test_mse']:.6f}"
        )
    return summary_rows


def main(dataset_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", default="all", help="all or comma-separated horizons")
    parser.add_argument("--mode", choices=["latest", "original"], default="latest")
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    run_dataset(
        dataset_name,
        horizons=parse_horizons(args.horizons),
        mode=args.mode,
        lookback=args.lookback,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress=not args.no_progress,
    )
