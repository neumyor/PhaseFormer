#!/usr/bin/env python3
"""Matched warm-start validation experiments for the three low-rank hypotheses."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A factorial normalization/scale control precedes the two routing hypotheses.
ARMS = {
    "i0_continue": {},
    "pt_post": {"trend_interaction_mode": "phase_to_trend"},
    "pt_pre": {"trend_interaction_mode": "phase_to_trend", "lowrank_update_style": "pre_norm"},
    "pt_scaled": {"trend_interaction_mode": "phase_to_trend", "lowrank_update_style": "scaled_pre_norm"},
    "tp_post": {"trend_interaction_mode": "trend_to_phase"},
    "tp_pre": {"trend_interaction_mode": "trend_to_phase", "lowrank_update_style": "pre_norm"},
    "tp_scaled": {"trend_interaction_mode": "trend_to_phase", "lowrank_update_style": "scaled_pre_norm"},
    "pt_null": {"trend_interaction_mode": "phase_to_trend", "lowrank_update_style": "scaled_pre_norm", "lowrank_null_read": True},
    "tp_selective": {"trend_interaction_mode": "trend_to_phase", "lowrank_update_style": "scaled_pre_norm", "lowrank_selective_write": True},
}
DEFAULT_SETTINGS = "ETTh1:96:2021,ETTh1:96:2022,Weather:720:2021,Weather:720:2022,ETTm2:720:2021,ETTm2:720:2022"


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discover(source, settings):
    requested = {tuple(s.split(":")) for s in settings.split(",")}
    found = {}
    for path in sorted((source / "search" / "runs").glob("*/config.json")):
        config = json.loads(path.read_text())
        key = tuple(str(config[k]) for k in ("dataset", "horizon", "seed"))
        if key not in requested or config["mechanism"] != "phaseformer_l_i0_late_fusion":
            continue
        with (path.parent / "metrics.csv").open() as f:
            row = list(csv.DictReader(f))[-1]
        if key in found:
            raise ValueError(f"ambiguous source for {key}")
        hp = config["hyperparams"]
        assert hp["use_lowrank_interaction"] and not hp.get("use_trend_token", False)
        assert hp["trend_interaction_mode"] == "none"
        assert hp["weak_period_residual_head_type"] == "pooled_lowrank"
        assert hp["weak_period_residual_rank"] == int(key[1]) // 8
        assert hp["weak_period_residual_pool_factor"] == 1
        assert config["lookback"] == 720 and config["period"] == 24
        assert config["loss"] == "huber" and not config["evaluate_test"]
        ckpt = Path(row["checkpoint"])
        config["checkpoint"] = str(ckpt if ckpt.is_absolute() else source.parent.parent / ckpt)
        if not Path(config["checkpoint"]).is_file():
            raise FileNotFoundError(config["checkpoint"])
        setting = f"{key[0]}_H{key[1]}_S{key[2]}_val"
        found[key] = {"setting": setting, "config": config, "source_metrics": row,
                      "source_checkpoint_sha256": digest(config["checkpoint"])}
    if set(found) != requested:
        raise ValueError(f"missing sources: {requested - set(found)}")
    return list(found.values())


def prepare_config(source_config, arm, epochs):
    config = copy.deepcopy(source_config)
    hp = config["hyperparams"]
    hp.update(ARMS[arm])
    hp["train_epochs"] = epochs
    config.update(max_epochs=epochs, evaluate_test=False)
    return config


def construct(config, workers=2):
    from src.models.PhaseFormer import PhaseFormer
    from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args
    hp = config["hyperparams"]
    args = make_exp_args(config["dataset"], config["lookback"], config["horizon"], hp,
                         batch_size=config["batch_size"])
    args.dataset_args.percent = config["percent"]
    args.dataset_args.num_workers = workers
    for key in ("input_hypothesis", "input_variant", "intervention_seed",
                "input_d1_period", "input_d1_sigma", "input_d2_recent_length", "input_d3_component"):
        setattr(args.dataset_args, key, config[key])
    args.dataset_args.input_period_len = config["period"]
    if not Path(args.dataset_args.root_path).exists() and config["dataset"].startswith("ETT"):
        args.dataset_args.root_path = str(ROOT / "resources/all_datasets/ETT-small")
    return PhaseFormer(PhaseFormerPresetConfig(args, config["lookback"], config["horizon"], hp)), args


def load_i0(model, state):
    import torch
    current = model.state_dict()
    prefix = "lowrank_interaction_branch."
    missing = set(current) - set(state)
    unexpected = set(state) - set(current)
    if unexpected or any(not k.startswith(prefix) or k.startswith(prefix + "lowrank_head.") for k in missing):
        raise ValueError(f"invalid warm start: missing={missing}, unexpected={unexpected}")
    result = model.load_state_dict(state, strict=False)
    assert set(result.missing_keys) == missing and not result.unexpected_keys
    assert all(torch.equal(model.state_dict()[k], value) for k, value in state.items())
    return sorted(missing)


def forecast(model, batch, device):
    x, y, xm, ym = [v.float().to(device) for v in batch]
    pred, _, _ = model(x, xm, model._build_decoder_input(y), ym)
    return pred[:, -model.pred_len:, :], y[:, -model.pred_len:, :]


def train_job(path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import pytorch_lightning as pl
    from src.dataset.data_factory import data_provider
    from src.training.runner import build_logger, build_trainer, restore_best_checkpoint
    from scripts.search_phaseformer import evaluate
    job = json.loads(Path(path).read_text())
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("stage 2 requires CUDA")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("medium")
    cfg = prepare_config(job["source"]["config"], job["arm"], job["epochs"])
    if digest(cfg["checkpoint"]) != job["source"]["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint changed since manifest creation")
    pl.seed_everything(cfg["seed"], workers=True)
    model, args = construct(cfg)
    state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    missing = load_i0(model, state)
    baseline, _ = construct(job["source"]["config"])
    baseline.load_state_dict(state, strict=True)
    train_set, train_loader = data_provider(args.dataset_args, "train")
    val_set, val_loader = data_provider(args.dataset_args, "val")
    audit_batch = [v[:2] for v in next(iter(val_loader))]
    device = torch.device("cuda:0")
    model.to(device).eval()
    baseline.to(device).eval()
    with torch.inference_mode():
        initial, _ = forecast(model, audit_batch, device)
        reference, _ = forecast(baseline, audit_batch, device)
    max_abs = (initial - reference).abs().max().item()
    identity_required = job["arm"] in {"i0_continue", "pt_scaled", "tp_scaled", "pt_null", "tp_selective"}
    if identity_required and max_abs != 0.0:
        raise AssertionError(f"initial i0 identity failed: {max_abs}")
    del baseline, state
    save_json(out / "initialization.json", {
        "max_abs_vs_i0": max_abs, "identity_required": identity_required,
        "missing_keys": missing, "source_checkpoint_sha256": job["source"]["source_checkpoint_sha256"],
        "rank": model.lowrank_interaction_branch.rank,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    })
    logger = build_logger(str(out / "lightning"), version="train")
    trainer, checkpoint = build_trainer(max_epochs=job["epochs"], logger=logger,
        patience=job["epochs"] + 1, checkpoint_dir=str(out / "checkpoints"), progress=False)
    if job["smoke"]:
        trainer.limit_train_batches = 2
        trainer.limit_val_batches = 2
    # Reset after all construction/audits, so all arms use matched batch order.
    pl.seed_everything(cfg["seed"], workers=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.monotonic() - started
    restore_best_checkpoint(model, checkpoint)
    model.to(device)
    metric = evaluate(model, val_loader, val_set, "val", cfg["horizon"], out)
    cfg["checkpoint"] = checkpoint.best_model_path
    save_json(out / "config.json", cfg)
    save_json(out / "result.json", {
        "setting": job["source"]["setting"], "arm": job["arm"], **metric,
        "elapsed_sec": elapsed, "epochs": trainer.current_epoch,
        "best_val_loss": float(checkpoint.best_model_score),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_max_abs": max_abs, "checkpoint": checkpoint.best_model_path,
        "interaction_scale": float(model.lowrank_interaction_branch.interaction_scale.detach())
            if hasattr(model.lowrank_interaction_branch, "interaction_scale") else None,
        "null_score": float(model.lowrank_interaction_branch.null_score.detach())
            if hasattr(model.lowrank_interaction_branch, "null_score") else None,
        "python": sys.version, "torch": torch.__version__, "lightning": pl.__version__,
        "gpu": torch.cuda.get_device_name(0), "smoke": job["smoke"],
    })


def launch(job_path, gpu):
    job = json.loads(job_path.read_text())
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    status = out / "status.json"
    if (out / "result.json").exists():
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [sys.executable, str(Path(__file__).resolve()), "--job", str(job_path)]
    save_json(status, {"status": "running", "gpu": gpu, "command": command})
    with (out / "process.log").open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    save_json(status, {"status": "completed" if result.returncode == 0 else "failed",
                       "gpu": gpu, "returncode": result.returncode, "command": command})
    if result.returncode:
        raise RuntimeError(f"job failed: {job_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.job:
        train_job(args.job)
        return
    import yaml
    sources = discover(args.source_dir.resolve(), args.settings)
    scratch = args.scratch_dir.resolve()
    audit = args.audit_dir.resolve()
    jobs = []
    for arm in ARMS:
        for source in sources:
            path = scratch / "jobs" / f"{source['setting']}__{arm}.json"
            job = {"source": source, "arm": arm, "epochs": args.epochs, "smoke": args.smoke,
                   "output": str(scratch / "runs" / f"{source['setting']}__{arm}")}
            if path.exists() and json.loads(path.read_text()) != job:
                raise ValueError(f"resume configuration changed: {path}")
            save_json(path, job)
            jobs.append(path)
    manifest = {"experiment_id": audit.name, "sources": sources, "arms": ARMS,
                "epochs": args.epochs, "smoke": args.smoke, "jobs": [str(p) for p in jobs],
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    save_json(scratch / "manifest.json", manifest)
    audit.mkdir(parents=True, exist_ok=True)
    settings = [{"setting": s["setting"], "dataset": s["config"]["dataset"], "horizon": s["config"]["horizon"],
                 "seed": s["config"]["seed"], "lookback": 720, "split": "validation"} for s in sources]
    run = {"experiment_id": audit.name,
           "code": {"repository": "PhaseFormer", "branch": "codex/trend-token-cross-attention", "commit": manifest["code_commit"]},
           "mechanism": {"description": "Identity residual, null read, phase-conditional write", "feature_flag": list(ARMS)},
           "experiment": {"baseline": "i0_continue", "candidate": list(ARMS)[1:], "settings": settings,
                          "training": {"epochs": args.epochs, "warm_start": "paired stage-1 i0 weights", "optimizer": "fresh Adam; source learning rate and scheduler", "checkpoint": "lowest validation Huber loss after training", "hyperparameters": {s['setting']: s['config']['hyperparams'] for s in sources}}, "metrics": ["MSE", "MAE"]},
           "execution": {"environment": {"python": sys.executable}, "settings": [{"setting": s['setting'], "commands": [[sys.executable, str(Path(__file__).resolve()), "--job", str(p)] for p in jobs if p.name.startswith(s['setting'] + '__')]} for s in sources]},
           "selection": {"source": "validation", "selected_configs": [{"setting": s['setting'], "config_id": list(ARMS), "search_notes": "All fixed configurations retained; checkpoint selection only; no test"} for s in sources]},
           "analysis": {"ranking_metric": "mae", "top_k": 5, "dedup_rule": "within each setting/arm/category/channel separate sample starts by >= horizon"},
           "validation": {"status": "incomplete", "issues": ["training and paired audit pending"]}}
    (audit / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    print(json.dumps({"jobs": len(jobs), "settings": len(sources), "scratch": str(scratch)}, indent=2), flush=True)
    if args.prepare_only:
        return
    gpus = [int(x) for x in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU ids must be nonempty and unique")
    # A barrier keeps the two routing hypotheses after the update-rule controls.
    for stage_arms in (set(ARMS) - {"pt_null", "tp_selective"}, {"pt_null", "tp_selective"}):
        stage = [p for p in jobs if json.loads(p.read_text())["arm"] in stage_arms]
        def worker(gpu, bucket):
            for path in bucket:
                launch(path, gpu)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu, stage[i::len(gpus)]) for i, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
    if not args.smoke:
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_lowrank_stage2.py"),
                        "--scratch-dir", str(scratch), "--audit-dir", str(audit), "--gpus", args.gpus], check=True)


if __name__ == "__main__":
    main()
