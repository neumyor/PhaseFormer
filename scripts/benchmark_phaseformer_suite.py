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
from src.training.runner import build_logger, build_trainer

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    PhaseFormerPresetConfig,
    build_hyperparams,
    get_dataset_horizons,
    make_exp_args,
)


DEFAULT_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Exchange",
    "Electricity",
    "Traffic",
    "Weather",
]


def parse_int_list(value):
    if value == "all":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
                        "sample_mae": float(torch.abs(err[local_idx]).mean().detach().cpu()),
                    }
                )
    return sorted(bad_cases, key=lambda row: row["sample_mse"], reverse=True)[
        :bad_case_limit
    ]


def read_existing_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def build_run_id(prefix, mode, dataset, horizon, hyperparams, seed):
    scheme = hyperparams.get("scheme_name", mode)
    return f"{prefix}_{mode}_{scheme}_{dataset.lower()}_{horizon}_seed{seed}"


def run_one(args, mode, dataset, horizon):
    hyperparams = build_hyperparams(dataset, horizon, mode)
    if args.epochs is not None:
        hyperparams["train_epochs"] = args.epochs
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = None
    seed = int(hyperparams.get("seed", args.seed))
    run_id = build_run_id(args.run_prefix, mode, dataset, horizon, hyperparams, seed)
    run_dir = os.path.join(args.output_dir, run_id)

    if args.resume and os.path.exists(os.path.join(run_dir, "metrics.csv")):
        metrics = read_existing_metrics(run_dir)
        metrics["resumed"] = "true"
        return metrics
    os.makedirs(run_dir, exist_ok=False)

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    exp_args = make_exp_args(
        dataset,
        args.lookback,
        horizon,
        hyperparams,
        batch_size=batch_size,
    )
    exp_args.dataset_args.num_workers = args.num_workers
    exp_args.training_args.num_workers = args.num_workers

    with open(os.path.join(run_dir, "commands.sh"), "w") as f:
        f.write(" ".join(["python", "scripts/benchmark_phaseformer_suite.py"] + sys.argv[1:]))
        f.write("\n")

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
    config_snapshot = {
        "mode": mode,
        "dataset": dataset,
        "horizon": horizon,
        "lookback": args.lookback,
        "seed": seed,
        "hyperparams": hyperparams,
        "dataset_args": {
            "root_path": exp_args.dataset_args.root_path,
            "data_path": exp_args.dataset_args.data_path,
            "freq": exp_args.dataset_args.freq,
            "features": exp_args.dataset_args.features,
            "num_variants": exp_args.model_args.num_variants,
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    model = PhaseFormer(model_config)
    logger = build_logger(
        os.path.join(run_dir, "lightning"),
        name="PhaseFormer",
        version=mode,
    )
    trainer, checkpoint = build_trainer(
        max_epochs=exp_args.training_args.train_epochs,
        logger=logger,
        patience=exp_args.training_args.patience,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        progress=args.progress,
    )

    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_result = trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path="best",
        verbose=False,
        weights_only=False,
    )
    elapsed = time.time() - start
    test_metrics = test_result[0] if test_result else {}
    bad_cases = collect_bad_cases(
        model,
        test_loader,
        horizon,
        args.bad_case_limit,
        args.bad_case_batches,
    )

    metrics_row = {
        "run_id": run_id,
        "mode": mode,
        "scheme": hyperparams.get("scheme_name", mode),
        "dataset": dataset,
        "lookback": args.lookback,
        "horizon": horizon,
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
        "val_size": len(val_set),
        "test_size": len(test_set),
        "resumed": "false",
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
    return metrics_row


def summarize(rows, output_dir, run_prefix):
    summary_path = os.path.join(output_dir, f"{run_prefix}_summary.csv")
    fieldnames = [
        "run_id",
        "mode",
        "scheme",
        "dataset",
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
        "elapsed_sec",
        "train_size",
        "val_size",
        "test_size",
        "resumed",
    ]
    write_csv(summary_path, rows, fieldnames)

    keyed = {}
    for row in rows:
        keyed[(row["dataset"], str(row["horizon"]), row["mode"])] = row
    compare_rows = []
    comparable_modes = {"latest", "best_nonresidual"}
    for dataset, horizon, mode in sorted(keyed):
        if mode not in comparable_modes:
            continue
        original = keyed.get((dataset, horizon, "original"))
        candidate = keyed[(dataset, horizon, mode)]
        if not original:
            continue
        orig_mae = float(original["test_mae"])
        orig_mse = float(original["test_mse"])
        candidate_mae = float(candidate["test_mae"])
        candidate_mse = float(candidate["test_mse"])
        compare_rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "mode": mode,
                "original_mae": orig_mae,
                "candidate_mae": candidate_mae,
                "mae_delta_pct": (candidate_mae - orig_mae) / orig_mae * 100.0,
                "original_mse": orig_mse,
                "candidate_mse": candidate_mse,
                "mse_delta_pct": (candidate_mse - orig_mse) / orig_mse * 100.0,
                "candidate_scheme": candidate["scheme"],
            }
        )
    compare_path = os.path.join(output_dir, f"{run_prefix}_comparison.csv")
    if compare_rows:
        write_csv(compare_path, compare_rows, list(compare_rows[0]))
    return summary_path, compare_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--horizons", default="all")
    parser.add_argument("--modes", default="original,latest")
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bad-case-limit", type=int, default=10)
    parser.add_argument("--bad-case-batches", type=int, default=8)
    parser.add_argument("--output-dir", default="research_runs")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.run_prefix is None:
        args.run_prefix = f"phaseformer_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    selected_horizons = parse_int_list(args.horizons)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for dataset in datasets:
        horizons = selected_horizons or get_dataset_horizons(dataset)
        for horizon in horizons:
            for mode in modes:
                rows.append(run_one(args, mode, dataset, horizon))
                summary_path, compare_path = summarize(rows, args.output_dir, args.run_prefix)
                print(f"Current summary: {summary_path}")
                print(f"Current comparison: {compare_path}")

    summary_path, compare_path = summarize(rows, args.output_dir, args.run_prefix)
    print(f"Final summary: {summary_path}")
    print(f"Final comparison: {compare_path}")


if __name__ == "__main__":
    main()
