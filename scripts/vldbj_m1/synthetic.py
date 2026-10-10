#!/usr/bin/env python3
"""V1: M1 vs b0 on synthetic multiplicative-seasonality series (mechanism check).

Generates X_kp = l_k + (1 + alpha*l_k) s_p + e (random-walk level, centered
seasonal shape) and trains the compressed-branch b0 and M1 from scratch with a
plain torch loop.  Pre-registered reading: the paired M1-b0 validation gap
grows with alpha; at alpha=0 both arms are statistically indistinguishable.

This is a mechanism check, not a benchmark-protocol result: no Lightning, no
lr schedule, synthetic data only.  Runs on CPU.

Usage: python scripts/vldbj_m1/synthetic.py [--quick]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from src.models.phaseformer_presets import get_base_hyperparams  # noqa: E402
from scripts.vldbj_m1.model import prepare_config, construct  # noqa: E402

PERIOD = 24
LOOKBACK = 720
HORIZON = 96
CYCLES = 3000
ALPHAS = (0.0, 0.25, 0.5, 1.0)
SEEDS = (2021, 2022)
EPOCHS = 30
BATCH = 64


def series(alpha, seed, channels=7):
    """(length, channels) series; every channel has its own level path."""
    rng = np.random.default_rng(seed)
    length = CYCLES * PERIOD
    phase = np.arange(length) % PERIOD
    shape = np.sin(2 * np.pi * phase / PERIOD)
    out = []
    for c in range(channels):
        level = np.cumsum(rng.normal(0, 0.05, length))
        out.append(level + (1 + alpha * level) * shape + rng.normal(0, 0.1, length))
    return np.stack(out, -1).astype('float32')


def windows(x, split):
    cut = int(len(x) * (0.7 if split == 'train' else 0.75))
    part = x[cut:] if split != 'train' else x[:cut]
    count = (len(part) - LOOKBACK - HORIZON) // PERIOD + 1
    data = np.stack([part[i * PERIOD:i * PERIOD + LOOKBACK + HORIZON] for i in range(count)])
    return torch.from_numpy(data)


def source_config(seed):
    hp = get_base_hyperparams('ETTh1', HORIZON)
    hp.update(seed=seed, scheme_name='no_residual', period_len=24, loss_func='huber',
              use_huber_loss=True, train_epochs=EPOCHS, use_residual_head=False)
    return dict(dataset='ETTh1', horizon=HORIZON, lookback=720, mechanism='no_residual',
                period=24, percent=100, max_epochs=EPOCHS, seed=seed, loss='huber',
                batch_size=BATCH, evaluate_test=False, input_hypothesis='none',
                input_variant='full', intervention_seed=9102, input_d1_period=0.0,
                input_d1_sigma=0.0, input_d2_recent_length=0, input_d3_component='',
                init_checkpoint='', hyperparams=hp, config_hash='synthetic')


def frame(batch):
    """Give the (window, future) pair the (batch, time, channel=1) layout."""
    return batch[..., None] if batch.ndim == 2 else batch


def run_one(alpha, seed, arm, quick):
    torch.manual_seed(seed)
    x = windows(series(alpha, seed), 'train')
    v = windows(series(alpha, seed), 'val')
    model, _ = construct(prepare_config(source_config(seed), arm, 'S2', False))
    device = torch.device('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    delta = torch.tensor(1.0)
    best, best_state = float('inf'), None
    epochs = 2 if quick else EPOCHS
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for start in range(0, len(x), BATCH):
            batch = frame(x[order[start:start + BATCH]])
            win, future = batch[:, :LOOKBACK], batch[:, LOOKBACK:]
            pred = model(win, torch.zeros(len(batch), LOOKBACK, 4),
                         model._build_decoder_input(future),
                         torch.zeros(len(batch), HORIZON, 4))[0][:, -HORIZON:]
            err = pred - future
            loss = torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=delta)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            errs = []
            for start in range(0, len(v), BATCH):
                batch = frame(v[start:start + BATCH])
                win, future = batch[:, :LOOKBACK], batch[:, LOOKBACK:]
                pred = model(win, torch.zeros(len(batch), LOOKBACK, 4),
                             model._build_decoder_input(future),
                             torch.zeros(len(batch), HORIZON, 4))[0][:, -HORIZON:]
                errs.append((pred - future).square().mean(1).mean())
            val = torch.stack(errs).mean().item()
        if val < best:
            best, best_state = val, {k: p.detach().clone() for k, p in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'research_runs/vldbj_m1_synthetic')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            b0 = run_one(alpha, seed, 'b0', args.quick)
            m1 = run_one(alpha, seed, 'm1', args.quick)
            rows.append({'alpha': alpha, 'seed': seed, 'b0_val_mse': b0, 'm1_val_mse': m1,
                         'm1_minus_b0': m1 - b0})
            print(rows[-1], flush=True)
    (args.output / 'synthetic.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
