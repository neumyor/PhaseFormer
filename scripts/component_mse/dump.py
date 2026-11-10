"""Shared on-disk layout for the component-MSE experiment.

Everything is kept in the StandardScaler space the models are trained in, which
is also the space the paper's MSE is reported in; the scaler's per-channel mean
and scale are stored next to the truth so raw units can be rebuilt later.

Layout under ``dump_root``::

    truth/<dataset>_H<horizon>.npz        truth (N, H, C) float32, mean, scale
    predictions/<model>/<model>__<dataset>_H<horizon>_S<seed>.npz
    results/<model>__<dataset>_H<horizon>_S<seed>.json

Windows are appended in dataloader order, which is stable because the test
loader is not shuffled and does not drop the last batch; every model therefore
sees the same N rows in the same order.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def truth_path(root, dataset, horizon):
    return Path(root) / 'truth' / f'{dataset}_H{horizon}.npz'


def prediction_path(root, model, dataset, horizon, seed):
    return Path(root) / 'predictions' / model / f'{model}__{dataset}_H{horizon}_S{seed}.npz'


def result_path(root, model, dataset, horizon, seed):
    return Path(root) / 'results' / f'{model}__{dataset}_H{horizon}_S{seed}.json'


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends .npz unless the name already ends with it, so the
    # temporary keeps the suffix and only the unique tag is added.
    temporary = path.parent / f'{path.stem}.{os.getpid()}.tmp{path.suffix}'
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def save_truth(root, dataset, horizon, truth, scaler):
    """Write the shared truth array once; identical for every model."""
    path = truth_path(root, dataset, horizon)
    if path.exists():
        return
    _atomic_savez(path, truth=truth.astype('float32'),
                  scaler_mean=scaler.mean_.astype('float64'),
                  scaler_scale=scaler.scale_.astype('float64'))


def save_prediction(root, model, dataset, horizon, seed, prediction):
    _atomic_savez(prediction_path(root, model, dataset, horizon, seed),
                  prediction=prediction.astype('float32'))


def save_result(root, model, dataset, horizon, seed, record):
    path = result_path(root, model, dataset, horizon, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))


def load_truth(root, dataset, horizon):
    with np.load(truth_path(root, dataset, horizon)) as handle:
        return handle['truth']


def load_prediction(root, model, dataset, horizon, seed):
    with np.load(prediction_path(root, model, dataset, horizon, seed)) as handle:
        return handle['prediction']


def test_arrays(prediction, truth):
    """Element-mean MSE/MAE, the same reduction the paper's tables use."""
    prediction = prediction.astype('float64')
    truth = truth.astype('float64')
    error = prediction - truth
    return float(np.square(error).mean()), float(np.abs(error).mean())
