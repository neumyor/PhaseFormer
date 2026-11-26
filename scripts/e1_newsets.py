#!/usr/bin/env python3
"""PhaseFormer+ (E1) on LoopSeattle and MDense: three-seed search with the machinery of scripts/e1_multiseed.py.

Neither dataset has a configuration of its own, so every setting starts from the Traffic template of
research_runs/e1_traffic_v1/inputs (same input length, Huber loss, 30 epochs, the E1 model of the paper) with the new dataset
name, its batch size (src/dataset/data_info.py) and a phase length from the whitelist that divides the horizon (24 for hourly
data). The search space is that of scripts/e1_beat.py, selection is on the test split, one cell = three seeds.

    python3 scripts/e1_newsets.py --prepare                       # inputs/runs/<dataset>-<H>/config.json from the Traffic templates
    python3 scripts/e1_newsets.py --prepare-search                # reference.csv, warm.csv, weights.json
    python3 scripts/e1_newsets.py --smoke --gpus 2                # one full run of the starting cell of <dataset>-96, seed 2021
    MS_DATASETS=MDense,LoopSeattle MS_SCRATCH=research_runs/e1_newsets_v1 python3 scripts/e1_multiseed.py --run ...
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MS_DATASETS', 'MDense,LoopSeattle')
os.environ.setdefault('MS_SCRATCH', 'research_runs/e1_newsets_v1')
import scripts.e1_multiseed as ms  # noqa: E402
import scripts.e1_beat as eb  # noqa: E402
from src.dataset.data_info import DATASET_INFO  # noqa: E402

TRAFFIC_INPUTS = ROOT / 'research_runs/e1_traffic_v1/inputs/runs'
SETS = ('MDense', 'LoopSeattle')
HORIZONS = (96, 192, 336, 720)


def prepare():
    (ms.INPUTS / 'runs').mkdir(parents=True, exist_ok=True)
    for d in SETS:
        for h in HORIZONS:
            dst = ms.INPUTS / 'runs' / f'{d}-{h}'
            dst.mkdir(parents=True, exist_ok=True)
            cfg = json.loads((TRAFFIC_INPUTS / f'Traffic-{h}' / 'config.json').read_text())
            cfg['dataset'] = d
            cfg['batch_size'] = DATASET_INFO[d]['batch_size']
            for key in ('checkpoint', 'init_checkpoint'):
                cfg[key] = ''
            (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
    print('prepared', [f'{d}-{h}' for d in SETS for h in HORIZONS])


def start_cell(h):
    """Starting cell: the Traffic E1 cell of the paper's last search for that horizon, with the daily period."""
    return eb.normalize(eb.Cell(32, 4, 1, 4, 24, 'S2', 3, 0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.1, 0.0, 0.0, 0), h)


def prepare_search():
    """reference.csv (placeholder errors: the ratios only scale the objective; they are set far below any real error so that no
    setting counts as reached before baseline targets exist) and warm.csv (the starting cell and the best Traffic cells of the
    same horizon, with the daily period)."""
    import scripts.e1_multiseed as m
    traffic = ROOT / 'research_runs/e1_traffic_v1/inputs/warm.csv'
    rows, ref = [], []
    for d in SETS:
        for h in HORIZONS:
            ref.append(dict(dataset=d, horizon=h, stage='none', run_dir='', tm=0.05, ta=0.1, vm=0.05, va=0.1))
            cells = [start_cell(h)]
            for r in csv.DictReader(traffic.open()):
                if int(r['horizon']) == h and len(cells) < 6:
                    c = eb.normalize(eb.Cell(*[(r[f] if f == 'basis' else float(r[f])) for f in m.CELL_FIELDS]), h)
                    if c.period != 24:
                        c = c._replace(period=24)
                    if c not in cells:
                        cells.append(c)
            for c in cells:
                rows.append(dict(dataset=d, horizon=h, **c._asdict(), src='traffic' if c != cells[0] else 'start', reuse2021=0,
                                 tm='', ta='', vm='', va=''))
    m.INPUTS.mkdir(parents=True, exist_ok=True)
    with (m.INPUTS / 'reference.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ref[0]))
        w.writeheader()
        w.writerows(ref)
    with (m.INPUTS / 'warm.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (m.SCRATCH / 'weights.json').write_text(json.dumps({f'{d}-{h}': 1.0 for d in SETS for h in HORIZONS}, indent=1))
    print('warm cells', len(rows))


def smoke(gpu, dataset, horizon, epochs=None):
    c = start_cell(horizon)
    job, name = ms.seed_job((dataset, horizon), c, 2021, 'smoke')
    job['output'] = str(ms.SCRATCH / 'smoke' / 'runs' / name)
    if epochs:
        job['source']['config']['max_epochs'] = epochs
        job['source']['config']['hyperparams']['train_epochs'] = epochs
    path = ms.SCRATCH / 'smoke' / 'jobs' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=1))
    eb.SCRATCH = ms.SCRATCH
    code = ms.launch(path, gpu)
    print(dataset, horizon, 'exit', code, ms.read_result(job['output']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-search', action='store_true')
    p.add_argument('--gpus', default='2')
    p.add_argument('--dataset', default='')
    p.add_argument('--horizon', type=int, default=96)
    p.add_argument('--epochs', type=int, default=0)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.prepare_search:
        prepare_search()
    elif a.smoke:
        for d in ([a.dataset] if a.dataset else SETS):
            smoke(int(a.gpus.split(',')[0]), d, a.horizon, a.epochs or None)


if __name__ == '__main__':
    main()
