"""Train and evaluate one bi-axial-stage job.

Protocol (same as the E14 phase-only protocol): input length 720, Adam at 1e-3,
Huber loss (delta 1), at most 30 epochs, early stopping with patience 8 on the
validation Huber loss, batch 256 (dataset default for the small datasets), the
standardized data of the repository's own dataset classes. The best-validation
state is evaluated on validation and test; selection uses validation only.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.biaxial.model import Cfg, PhaseFormerPlus, count_parameters  # noqa: E402

LOOKBACK = 720
MAX_EPOCHS = 30
PATIENCE = 8
LR = 1e-3
COMPILE_SAMPLES = 150_000   # training rows x channels above which the model is compiled
CFG_FIELDS = {f.name for f in fields(Cfg)}


def load_splits(dataset, horizon, data_root=None):
    from src.dataset.data_factory import data_provider
    from src.models.phaseformer_presets import make_exp_args
    args = make_exp_args(dataset, LOOKBACK, horizon, {'learning_rate': LR})
    if data_root:
        args.dataset_args.root_path = str(Path(data_root) / Path(args.dataset_args.root_path).name)
    args.dataset_args.num_workers = 0
    splits = {}
    for flag in ('train', 'val', 'test'):
        ds, _ = data_provider(args.dataset_args, flag)
        splits[flag] = torch.as_tensor(ds.data_x, dtype=torch.float32)
    return splits, args.training_args.batch_size


def windows(data, index, horizon):
    """Input windows (B, L, C) and target (B, H, C) for sample indices ``index``."""
    arange_x = index.unsqueeze(1) + torch.arange(LOOKBACK, device=data.device)
    arange_y = index.unsqueeze(1) + LOOKBACK + torch.arange(horizon, device=data.device)
    return data[arange_x], data[arange_y]


@torch.inference_mode()
def evaluate(model, data, horizon, batch):
    model.eval()
    n = data.shape[0] - LOOKBACK - horizon + 1
    sq = ab = hub = 0.0
    count = 0
    for start in range(0, n, batch):
        index = torch.arange(start, min(start + batch, n), device=data.device)
        x, y = windows(data, index, horizon)
        err = model(x) - y
        sq += err.square().sum().item()
        ab += err.abs().sum().item()
        hub += F.huber_loss(err, torch.zeros_like(err), reduction='sum').item()
        count += err.numel()
    return {'mse': sq / count, 'mae': ab / count, 'huber': hub / count}


def build_cfg(job, channels):
    overrides = {k: v for k, v in job.get('model', {}).items()}
    unknown = set(overrides) - CFG_FIELDS
    if unknown:
        raise ValueError(f'unknown model fields {unknown}')
    return Cfg(seq_len=LOOKBACK, pred_len=job['horizon'], channels=channels,
               period=job['period'], **overrides)


def flops_per_channel(cfg):
    from torch.utils.flop_counter import FlopCounterMode
    single = copy.deepcopy(cfg)
    single.channels = 1
    model = PhaseFormerPlus(single).eval()
    with FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, cfg.seq_len, 1))
    return counter.get_total_flops()


def run_job(job, splits, batch, device):
    torch.manual_seed(job['seed'])
    torch.cuda.manual_seed_all(job['seed'])
    train = splits['train'].to(device)
    val = splits['val'].to(device)
    test = splits['test'].to(device)
    horizon = job['horizon']
    cfg = build_cfg(job, train.shape[1])
    model = PhaseFormerPlus(cfg).to(device)
    # Compiling pays off only when an epoch is long; short jobs finish before the compile time is repaid.
    run = torch.compile(model) if train.shape[0] * train.shape[1] > COMPILE_SAMPLES and device.type == 'cuda' else model
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_train = train.shape[0] - LOOKBACK - horizon + 1
    best, best_state, bad, epochs = float('inf'), None, 0, 0
    started = time.monotonic()
    generator = torch.Generator(device=device)
    generator.manual_seed(job['seed'])
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_train, device=device, generator=generator)
        for start in range(0, n_train - batch + 1, batch):
            x, y = windows(train, order[start:start + batch], horizon)
            loss = F.huber_loss(run(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epochs = epoch + 1
        score = evaluate(run, val, horizon, batch)['huber']
        if score < best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    train_seconds = time.monotonic() - started
    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val, horizon, batch)
    test_metrics = evaluate(model, test, horizon, batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.monotonic()
    evaluate(model, test, horizon, batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return {
        'id': job['id'], 'group': job['group'], 'dataset': job['dataset'], 'horizon': horizon,
        'seed': job['seed'], 'period': job['period'], 'model': job.get('model', {}),
        'val_mse': val_metrics['mse'], 'val_mae': val_metrics['mae'],
        'test_mse': test_metrics['mse'], 'test_mae': test_metrics['mae'],
        'params': count_parameters(model), 'flops_per_channel': flops_per_channel(cfg),
        'epochs': epochs, 'train_seconds': train_seconds,
        'infer_seconds': time.monotonic() - t0,
        'state': best_state,
    }
