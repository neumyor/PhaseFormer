#!/usr/bin/env python3
"""Case study of the E1 replication (07_case_study): where E1 corrects PhaseFormer on ETTm2-96.

Inputs: the E1 predictions of 03_trend_errors/e1_predictions (ETTm2-96, both selections), the PhaseFormer predictions of the
trend-error experiment (PF-e14, seed 2021, the checkpoint of the paper's case study) and the B0H+ predictions
(model 'B0H+', seed 2021) of ~/niuyiming/component_mse_20261023/predictions, all in standardized units, same test windows.

Writes under research_runs/e1_paper_v1/07_case_study/:
  stats_<model>.json   mean trend / periodic errors over all test windows of PhaseFormer and the model, and the share of
                       windows and of (window, channel) pairs on which the model has the lower trend error
  pool_<model>.npz     the 2400 (window, channel) pairs with the largest trend-error reduction (format of the paper's pool)
  paper_windows_<model>.npz   the three windows of the paper figure, (5054, 3), (9208, 3), (4222, 0), forecast by the model
  eligible_<model>.json       pairs meeting the selection criterion of the paper (periodic error at most 15% above PhaseFormer,
                       trend error down by at least 65%), best 20 by reduction
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402
from scripts.component_mse import dump, trend_definitions as td  # noqa: E402
from scripts.vldbj_m1.proto import build_model  # noqa: E402
from src.dataset.data_factory import data_provider  # noqa: E402

BASE = ROOT / 'research_runs/e1_paper_v1'
OUT = BASE / '07_case_study'
DUMP = Path.home() / 'niuyiming/component_mse_20261023/predictions'
PAPER = ((5054, 3), (9208, 3), (4222, 0))
P = 24


def split(error):
    trend = td.apply_trend(error.astype('float32'), 'cycle-level', P)
    return np.square(trend).mean(1), np.square(error - trend).mean(1)       # (N, C) each


def main():
    sa.register_heads()
    OUT.mkdir(parents=True, exist_ok=True)
    truth = dump.load_truth(DUMP, 'ETTm2', 96)
    pf = dump.load_prediction(DUMP, 'PF-e14', 'ETTm2', 96, 2021)
    tr_pf, pe_pf = split(pf - truth)
    models = {'E1-test': np.load(BASE / '03_trend_errors/e1_predictions/E1-test__ETTm2_H96.npz')['prediction'],
              'E1-val': np.load(BASE / '03_trend_errors/e1_predictions/E1-val__ETTm2_H96.npz')['prediction'],
              'B0H+': dump.load_prediction(DUMP, 'B0H+', 'ETTm2', 96, 2021)}
    job = None
    for r in csv.DictReader((BASE / '00_protocol/e1_final_configs_28.csv').open()):
        if (r['dataset'], int(r['horizon']), r['selection']) == ('ETTm2', 96, 'test'):
            job = dict(json.loads(Path(r['job']).read_text()), smoke=True)
    job.pop('source_hashes', None)
    job.pop('code_commit', None)
    _, args, _, _ = build_model(job)
    test_set, _ = data_provider(args.dataset_args, 'test')
    assert len(test_set) == len(truth), (len(test_set), len(truth))
    for name, pred in models.items():
        tr, pe = split(pred - truth)
        k = tr.size
        stats = dict(windows=int(pred.shape[0]), trend_phaseformer=float(tr_pf.mean()), trend_model=float(tr.mean()),
                     periodic_phaseformer=float(pe_pf.mean()), periodic_model=float(pe.mean()),
                     window_share=float((tr.mean(1) < tr_pf.mean(1)).mean()), pair_share=float((tr < tr_pf).sum() / k))
        (OUT / f'stats_{name}.json').write_text(json.dumps(stats, indent=1))
        reduction = tr_pf - tr
        flat = np.argsort(-reduction.reshape(-1))[:2400]
        win, ch = np.unravel_index(flat, tr.shape)

        def pack(windows, channels):
            hist = np.stack([test_set[int(w)][0][-96:, int(c)] for w, c in zip(windows, channels)])
            return dict(hist=hist.astype('float32'), truth=truth[windows, :, channels], pf=pf[windows, :, channels],
                        plus=pred[windows, :, channels], window=np.asarray(windows), channel=np.asarray(channels),
                        tr_pf=tr_pf[windows, channels], tr_plus=tr[windows, channels], pe_pf=pe_pf[windows, channels],
                        pe_plus=pe[windows, channels])

        np.savez_compressed(OUT / f'pool_{name}.npz', **pack(win, ch))
        pw = np.array([w for w, _ in PAPER])
        pc = np.array([c for _, c in PAPER])
        np.savez_compressed(OUT / f'paper_windows_{name}.npz', **pack(pw, pc))
        eligible = (pe <= 1.15 * pe_pf) & (tr <= 0.35 * tr_pf)
        idx = np.argwhere(eligible)
        order = np.argsort(-reduction[eligible])[:20]
        top = [dict(window=int(idx[i][0]), channel=int(idx[i][1]), trend_pf=float(tr_pf[tuple(idx[i])]),
                    trend_model=float(tr[tuple(idx[i])])) for i in order]
        (OUT / f'eligible_{name}.json').write_text(json.dumps(dict(n_eligible=int(eligible.sum()), best20=top), indent=1))
        print(name, json.dumps(stats), 'eligible', int(eligible.sum()), flush=True)


if __name__ == '__main__':
    main()
