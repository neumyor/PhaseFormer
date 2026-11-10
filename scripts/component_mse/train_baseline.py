#!/usr/bin/env python3
"""Train one efficient baseline (official implementation) and dump its test predictions.

The protocol mirrors the PhaseFormer conference runs so the component
comparison is like for like: lookback 720, the datasets' 70/10/20 split and
train-fitted StandardScaler, Adam at 1e-3, Huber loss (delta 1), batch 16,
30 epochs with patience 8 on the validation loss, and the best-validation
checkpoint read on the test split. Architecture hyperparameters come from each
paper's scripts (see ``models.py``).

The loaders are rebuilt here around the shared dataset so every sample also
carries its index, which CycleNet needs for its cycle queue; windows, order and
batch size are otherwise those of ``data_provider``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from torch.utils.data import Dataset  # noqa: E402

LOOKBACK = 720
BATCH_SIZE = 16
MAX_EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 1e-3
HUBER_DELTA = 1.0
NUM_WORKERS = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dump-root', type=Path, required=True)
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--limit-batches', type=int, default=0,
                        help='smoke testing only: stop each epoch after this many batches')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help='diagnostics only: override the shared 1e-3 protocol')
    parser.add_argument('--loss', default='huber', choices=('huber', 'mse'),
                        help='diagnostics only: override the shared Huber protocol')
    parser.add_argument('--fits-order', type=int, default=0,
                        help='diagnostics only: FITS harmonic order (cut_freq = (L//24+1)*order+10)')
    return parser.parse_args()


class Indexed(Dataset):
    """The shared dataset plus the sample index, for CycleNet's cycle queue."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base[index]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, index


def main():
    args = parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    import numpy as np
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from scripts.component_mse import dump, models
    from src.dataset.data_factory import data_provider
    from src.models.phaseformer_presets import make_exp_args

    pl.seed_everything(args.seed, workers=True)
    # Eight concurrent tasks share the node; without this torch spawns one BLAS
    # thread per core in every process and the batches are too small to profit.
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda:0')

    learning_rate = args.learning_rate
    hyperparams = {'learning_rate': learning_rate, 'patience': PATIENCE,
                   'train_epochs': MAX_EPOCHS, 'loss_func': 'huber'}
    exp_args = make_exp_args(args.dataset, LOOKBACK, args.horizon, hyperparams,
                             batch_size=BATCH_SIZE)

    splits = {}
    for flag in ('train', 'val', 'test'):
        base, _ = data_provider(exp_args.dataset_args, flag)
        splits[flag] = (base, DataLoader(
            Indexed(base), batch_size=BATCH_SIZE, shuffle=flag == 'train',
            num_workers=NUM_WORKERS, drop_last=flag == 'train'))
    train_set = splits['train'][0]
    train_loader, val_loader, test_loader = (splits[f][1] for f in ('train', 'val', 'test'))

    channel_count = int(train_set.var_needed)
    if args.fits_order and args.model == 'FITS':
        models.FITS_CUTOFF[args.dataset] = models.fits_cutoff(LOOKBACK, args.fits_order)
    model = models.build(args.model, LOOKBACK, args.horizon, channel_count, args.dataset).to(device)
    needs_cycle = args.model in models.NEEDS_CYCLE_INDEX
    if needs_cycle:
        # timestamps start at the split's first window, so origin i sits at i + seq_len.
        phase_tables = {flag: models.cycle_phase(splits[flag][0].timestamps, args.dataset)
                        for flag in splits}
    criterion = nn.HuberLoss(delta=HUBER_DELTA) if args.loss == 'huber' else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def forward(batch, flag):
        bx = batch[0].float().to(device)
        if needs_cycle:
            index = batch[4].to(device)
            phases_flag = phase_tables[flag][index.cpu().numpy() + LOOKBACK]
            return model(bx, torch.from_numpy(phases_flag).to(device).long())
        return model(bx)

    def read(flag, collect=False):
        loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}[flag]
        model.eval()
        total, count = 0.0, 0
        predictions, truths = [], []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if args.limit_batches and index >= args.limit_batches:
                    break
                pred = forward(batch, flag)[:, -args.horizon:]
                true = batch[1].float().to(device)[:, -args.horizon:]
                total += float(criterion(pred, true)) * pred.numel()
                count += pred.numel()
                if collect:
                    predictions.append(pred.cpu().numpy())
                    truths.append(true.cpu().numpy())
        if collect:
            return total / count, np.concatenate(predictions), np.concatenate(truths)
        return total / count

    best_loss, best_state, stale, used_epochs = float('inf'), None, 0, 0
    started = time.monotonic()
    for epoch in range(args.max_epochs):
        model.train()
        for index, batch in enumerate(train_loader):
            if args.limit_batches and index >= args.limit_batches:
                break
            pred = forward(batch, 'train')[:, -args.horizon:]
            true = batch[1].float().to(device)[:, -args.horizon:]
            loss = criterion(pred, true)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        used_epochs = epoch + 1
        val_loss = read('val')
        print(f'epoch {used_epochs} val {val_loss:.6f} {time.monotonic() - started:.0f}s', flush=True)
        if val_loss < best_loss - 1e-9:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    elapsed = time.monotonic() - started

    model.load_state_dict(best_state)
    model.eval()
    _, prediction, truth = read('test', collect=True)
    test_mse, test_mae = dump.test_arrays(prediction, truth)
    dump.save_truth(args.dump_root, args.dataset, args.horizon, truth, splits['test'][0].scaler)
    dump.save_prediction(args.dump_root, args.model, args.dataset, args.horizon, args.seed, prediction)

    record = {
        'model': args.model, 'dataset': args.dataset, 'horizon': args.horizon, 'seed': args.seed,
        'lookback': LOOKBACK, 'batch_size': BATCH_SIZE, 'learning_rate': learning_rate,
        'loss': args.loss, 'epochs': used_epochs, 'best_val_loss': best_loss,
        'test_mse': test_mse, 'test_mae': test_mae, 'test_size': int(prediction.shape[0]),
        'channels': channel_count, 'parameters': models.parameter_count(model),
        'elapsed_sec': elapsed, 'implementation': 'official',
        'space': 'standardized (train-fitted StandardScaler)',
    }
    dump.save_result(args.dump_root, args.model, args.dataset, args.horizon, args.seed, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
