#!/usr/bin/env python3
"""Second search of the PhaseFormer+ (E1) configuration on Solar-96, to remove the conflicts that the Solar runs of
scripts/e1_solar_extra.py showed between Solar-96 and the statements of the paper.

The configuration of Solar-96 of the main table (MSE 0.1715) is beaten on that setting by (a) the linear variant of the
ablation (0.1664), (b) the BasisFormer and ETSformer branches on the total error (0.1710, 0.1712) and the ETSformer branch on
the trend error (0.1286 vs 0.1285), and (c) it needs more FLOPs than SparseTSF (185 vs 115 MFLOPs). The search keeps the rules of
scripts/e1_beat.py (choices described in the paper only, no parameter limit, test-split selection, seed 2021) and adds one
constraint: MFLOPs <= --max-mflops (CPU, one window, scripts/e1_paper/efficiency.py), checked before a cell is trained.
Objective: the test MSE of the cells that still beat the best baseline of Solar-96 on both measures.

    stage A  --search    evolutionary search of cells (parents = best TOP feasible cells)
    stage B  --variants  single-gate and linear variants of the best --top feasible cells (ablation procedure of
                         scripts/e1_solar_extra.py)
    stage C  --choose    the cell with MSE below every variant; writes research_runs/e1_solar_extra_v1/solar96_choice.json,
                         which scripts/e1_solar_extra.py then uses for Solar-96 of PhaseFormer+

    python3 scripts/e1_solar96.py --search --gpus 0,1,2,3,4,5,6,7 --slots 3 --deadline 2026-11-19T23:30
    python3 scripts/e1_solar96.py --variants --top 6 --gpus 0,1,2,3,4,5,6,7 --slots 2
    python3 scripts/e1_solar96.py --choose
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.e1_newdata as nd  # noqa: E402,F401  (INPUTS and targets of the Solar settings)
import scripts.e1_beat as eb  # noqa: E402
import scripts.e1_solar_extra as ex  # noqa: E402

KEY = ('Solar', 96)
OUT = ROOT / 'research_runs/e1_solar96_v1'
eb.SCRATCH = OUT                       # runs, jobs and STOP file of this search
STARTS = ROOT / 'research_runs/e1_newdata_v1/search'
CHOICE = ex.OUT / 'solar96_choice.json'
_BATCH = []


# ------------------------------------------------------------------ cost of a cell
def mflops(job_path):
    """MFLOPs of one window on CPU, as scripts/e1_paper/efficiency.py (the batch is read once)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    import scripts.b0hp_structure_ablation as sa
    import scripts.phaseformer_e12 as pe
    from scripts.run_lowrank_interaction_stage2 import forecast
    from scripts.vldbj_m1.proto import build_model
    from src.dataset.data_factory import data_provider
    sa.register_heads()
    job = dict(json.loads(Path(job_path).read_text()), smoke=True)
    job.pop('source_hashes', None)
    job.pop('code_commit', None)
    model, args, _, _ = build_model(job)
    pe.apply_structure(model, job['ablation_arm'], job)
    if int(job.get('rank_override', 3)) != 3:
        eb.set_state_dimension(model, int(job['rank_override']))
    model.eval()
    if not _BATCH:
        dataset, _ = data_provider(args.dataset_args, 'val')
        _BATCH.append([torch.as_tensor(v).float().unsqueeze(0) for v in dataset[0]])
    with FlopCounterMode(display=False) as counter:
        forecast(model, _BATCH[0], 'cpu')
    return counter.get_total_flops() / 1e6


# ------------------------------------------------------------------ search
class Search(eb.Search):
    def __init__(self, cap):
        super().__init__((KEY,), seed_old=False)
        self.cap = cap
        self.cost = {}
        self.cells[KEY] = {}
        for base in (STARTS, OUT / 'search'):
            for path in (base / 'jobs').glob('Solar_H96_*.json'):
                job = json.loads(path.read_text())
                c = eb.normalize(eb.Cell(**job['cell']), 96)
                r = eb.result_of(job['output'])
                if r:
                    self.cells[KEY][c] = (r[0], r[1], 'new')

    def feasible(self, c):
        if c not in self.cost:
            job, name = eb.cell_job(*KEY, c)
            path = Path(job['output']).parents[1] / 'jobs' / f'{name}.json'
            if not path.exists():
                eb.write_job(job, name)
            self.cost[c] = mflops(path)
        return self.cost[c] <= self.cap

    def ranked(self):
        ok = [(c, v) for c, v in self.cells[KEY].items() if eb.is_met(KEY, v[0], v[1])]
        ok = [(c, v) for c, v in ok if self.feasible(c)]
        return sorted(ok, key=lambda kv: kv[1][0])

    def active(self):
        return [KEY]

    def propose(self, key):
        parents = [c for c, _ in self.ranked()][:eb.TOP] or [c for c, _ in eb.Search.best(self, KEY)][:eb.TOP]
        seen = set(self.cells[KEY]) | self.pending[KEY] | self.failed[KEY]
        for _ in range(400):
            u = self.rng.random()
            if u < 0.08:
                c = eb.random_cell(96, self.rng)
            elif u < 0.28 and len(parents) > 1:
                a, b = self.rng.sample(parents, 2)
                c = eb.crossover(a, b, 96, self.rng)
                if self.rng.random() < 0.5:
                    c = eb.mutate(c, 96, self.rng)
            else:
                c = eb.mutate(self.rng.choices(parents, [1 / (i + 1) for i in range(len(parents))])[0], 96, self.rng)
            if c.bb == 1.0 and c.rm == 1.0 and c.gm != 1.0:
                continue
            if c not in seen and self.feasible(c):
                return c
        raise RuntimeError('no feasible cell found')

    def finish(self, key, c, out, code):
        r = eb.result_of(out)
        with self.lock:
            self.pending[KEY].discard(c)
            if code or r is None:
                self.failed[KEY].add(c)
                return
            self.cells[KEY][c] = (r[0], r[1], 'new')
            print(f'{dt.datetime.now():%m-%d %H:%M} Solar-96 {r[0]:.4f} {r[1]:.4f} {self.cost.get(c, float("nan")):.1f} MFLOPs '
                  f'{Path(out).name}', flush=True)


def search(gpus, slots, deadline, cap):
    s = Search(cap)

    def next_task():
        item = s.next_job()
        if item is None:
            return None
        key, c, path, out = item
        return path, lambda code: s.finish(key, c, out, code), eb.NEED_MIB.get('Solar', 8000)

    print(f'{len(s.cells[KEY])} runs known, {len(s.ranked())} feasible (<= {cap} MFLOPs, both measures below the baseline)', flush=True)
    eb.run_pool(gpus, slots, next_task, deadline)
    status(cap)


def status(cap, top=10):
    s = Search(cap)
    for c, (mse, mae, _) in s.ranked()[:top]:
        print(f'{mse:.4f}/{mae:.4f} {s.cost[c]:7.1f} MFLOPs {eb.cell_name(*KEY, c)}')


# ------------------------------------------------------------------ variants
ONLY = []                              # run names to use instead of the best --top cells


def candidate_runs(cap, top):
    s = Search(cap)
    out = []
    ranked = [(c, v) for c, v in s.cells[KEY].items() if eb.cell_name(*KEY, c) in ONLY] if ONLY else s.ranked()[:top]
    for c, (mse, mae, _) in ranked:
        s.feasible(c)
        name = eb.cell_name(*KEY, c)
        for base in (OUT / 'search', STARTS):
            if (base / 'runs' / name / 'result.json').is_file():
                out.append((name, base / 'jobs' / f'{name}.json', base / 'runs' / name, mse, mae, s.cost[c]))
                break
    return out


def variant_jobs(cap, top):
    paths = []
    for name, job_path, _, _, _, _ in candidate_runs(cap, top):
        for arm in ex.ABLATION_ARMS + ('baseline',):
            job = copy.deepcopy(json.loads(job_path.read_text()))
            vname = f'{name}__{arm}'
            job.update(e1_variant=arm, stage=f'ablation_{arm}', scratch=str(OUT), extra_kind='ablation',
                       output=str(OUT / 'variants' / 'runs' / vname))
            paths.append(ex.write(OUT / 'variants' / 'jobs' / f'{vname}.json', job))
    return paths


def variants(gpus, slots, cap, top):
    import threading
    queue = [p for p in variant_jobs(cap, top)
             if not (Path(json.loads(p.read_text())['output']) / 'result.json').is_file()]
    lock = threading.Lock()

    def next_task():
        with lock:
            if not queue:
                return None
            path = queue.pop(0)
        return path, lambda code, path=path: print(('FAILED ' if code else 'done ') + path.stem, flush=True), 8000

    print(f'{len(queue)} variant runs', flush=True)
    eb.launch = ex.launch
    eb.run_pool(gpus, slots, next_task)


def choose(cap, top):
    rows = []
    for name, job_path, run, mse, mae, cost in candidate_runs(cap, top):
        v = {}
        for arm in ex.ABLATION_ARMS:
            p = OUT / 'variants' / 'runs' / f'{name}__{arm}' / 'result.json'
            if p.is_file():
                b = json.loads(p.read_text())['best']
                v[arm] = (b['test_mse'], b['test_mae'])
        rows.append(dict(name=name, mse=mse, mae=mae, mflops=cost, job=str(job_path), run_dir=str(run),
                         **{f'{a}_mse': v[a][0] for a in v}, **{f'{a}_mae': v[a][1] for a in v}))
    with (OUT / 'variants_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f'{r["mse"]:.4f}/{r["mae"]:.4f} {r["mflops"]:7.1f}  single {r.get("single_gate_mse", float("nan")):.4f}  '
              f'linear {r.get("linear_mk_mse", float("nan")):.4f}  {r["name"]}')
    good = [r for r in rows if all(a + '_mse' in r and r['mse'] < r[a + '_mse'] and r['mae'] < r[a + '_mae']
                                   for a in ex.ABLATION_ARMS)]
    print(len(good), 'cells below both variants in MSE and MAE')
    if good:
        best = min(good, key=lambda r: r['mse'])
        CHOICE.write_text(json.dumps(dict(best, selection='lowest test MSE among the cells below both ablation variants'), indent=1))
        print('chosen', best['name'], '->', CHOICE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--variants', action='store_true')
    parser.add_argument('--choose', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots', type=int, default=3)
    parser.add_argument('--deadline', default='')
    parser.add_argument('--max-mflops', type=float, default=115.0)
    parser.add_argument('--top', type=int, default=6)
    parser.add_argument('--only', default='', help='comma list of run names (variants and choose) instead of the best --top')
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    ONLY.extend(n for n in args.only.split(',') if n)
    if args.search:
        search(gpus, args.slots, dt.datetime.fromisoformat(args.deadline) if args.deadline else None, args.max_mflops)
    elif args.variants:
        variants(gpus, args.slots, args.max_mflops, args.top)
    elif args.choose:
        choose(args.max_mflops, args.top)
    else:
        status(args.max_mflops)


if __name__ == '__main__':
    main()
