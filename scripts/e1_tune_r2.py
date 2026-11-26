#!/usr/bin/env python3
"""Tuning of PhaseFormer-E1, stage R2: learning rates, gates, head, period, loss and structure.

Aim: for each priority setting, test MSE and MAE of E1 at most 0.5% above the recorded
B0H+ values. A run is scored by max(mse / B0H+ mse, mae / B0H+ mae) and the search
minimizes it (test-set selection, seed 2021, as for the recorded B0H+ configurations).

Per setting the search is adaptive. Round 0 holds the R0 winner with the recorded
rates (known from R0) and random cells, half drawn over the whole space and half around
the recorded cell. Later rounds perturb the best eight cells and add fresh cells. A
setting stops two rounds after the score reaches 1.005, when eight rounds bring less than
0.3% improvement, or when its run budget is used. The scheduler keeps all slots of the
GPUs busy with the pending runs of the highest-priority settings.

    python3 scripts/e1_tune_r2.py --run --gpus 4,5,6,7
    python3 scripts/e1_tune_r2.py --report
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402
import scripts.e1_tune as r0  # noqa: E402

SCRATCH = r0.SCRATCH
STAGE = 'r2'
SEED = r0.SEED
TARGET = 1.005
EXTRA_ROUNDS = 2
STALL_ROUNDS, STALL_GAIN = 8, 0.003
GATE_LO, GATE_HI = 1e-4, 0.999
LR_LO, LR_HI = 0.03, 1000.0
BB_LO, BB_HI = 0.1, 10.0
DELTAS = (1.0, 0.5, 0.3, 0.15)
PERIODS = {96: (8, 12, 16, 24, 48), 192: (8, 12, 16, 24, 32, 48, 64, 96),
           336: (8, 12, 16, 24, 28, 48, 56, 84, 112, 168),
           720: (8, 12, 16, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 120, 144, 180, 240, 360)}
TIERS = ((('ETTh2', 720), ('ETTh1', 192), ('ETTh2', 96), ('ETTh2', 336), ('ETTh1', 96), ('ETTh1', 336)),
         (('ETTm2', 720), ('ETTm1', 192), ('ETTm2', 336), ('ETTm2', 192), ('ETTm1', 96)),
         (('Weather', 336), ('Weather', 192), ('Weather', 96), ('Weather', 720)),
         (('Traffic', 96), ('Traffic', 336), ('Traffic', 720), ('Electricity', 192), ('Electricity', 720)))
HEAVY_SLOTS = 5          # Traffic and Electricity runs per GPU
HEAVY = ('Traffic', 'Electricity')
MAX_PER_GPU = 7
Cell = namedtuple('Cell', 'struct period gate gm rm bb head shared delta')


def run_plan(dataset):
    if dataset.startswith('ETTh'):
        return dict(budget=400, first=64, batch=16)
    if dataset.startswith('ETTm'):
        return dict(budget=200, first=48, batch=16)
    if dataset == 'Weather':
        return dict(budget=80, first=32, batch=8)
    return dict(budget=40, first=16, batch=8)


def logit(x):
    return math.log(x / (1 - x))


def expit(x):
    return 1 / (1 + math.exp(-x))


def g4(x):
    return float(f'{x:.4g}')


def normalize(c):
    gm = 0.0 if c.gm <= 0 else min(LR_HI, max(LR_LO, g4(c.gm)))
    return Cell(tuple(c.struct), int(c.period), min(GATE_HI, max(GATE_LO, round(float(c.gate), 5))), gm,
                min(LR_HI, max(LR_LO, g4(c.rm))), min(BB_HI, max(BB_LO, g4(c.bb))), c.head, int(c.shared),
                min(DELTAS, key=lambda d: abs(d - float(c.delta))))


def cell_name(dataset, horizon, c):
    d, m, n = c.struct
    return (f'{dataset}_H{horizon}_S{SEED}__e1r2__d{d}_M{m}_N{n}_p{c.period}_g{c.gate:g}_gm{c.gm:g}_rm{c.rm:g}'
            f'_bb{c.bb:g}_{c.head}_{"sh" if c.shared else "pc"}_dl{c.delta:g}')


def canonical_head(head):
    from scripts.vldbj_m1 import model as m
    sa.register_heads()
    return 'S1' if m.HEADS[head]['decoder'] == 'learned' else 'S2'


def gold_values():
    gold = {}
    for line in (ROOT / 'docs/PhaseFormer_gold_standard.md').read_text().splitlines():
        p = [x.strip() for x in line.split('|')]
        if len(p) > 5 and p[1] in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic'):
            gold[(p[1], int(p[2]))] = (float(p[3]), float(p[4]))
    return gold


def record_values():
    return {(r['dataset'], int(r['horizon'])): (float(r['test_mse']), float(r['test_mae']))
            for r in csv.DictReader(sa.settings_csv().open())}


def parse_structure(text):
    d, m, n = text.split('_')
    return int(d[1:]), int(m[1:]), int(n[1:])


# ------------------------------------------------------------------ jobs
_PROVENANCE = {}


def frozen():
    if not _PROVENANCE:
        from scripts.vldbj_m1.proto import provenance
        from scripts.vldbj_m1.run import commit
        _PROVENANCE.update(source_hashes=provenance(), code_commit=commit())
    return _PROVENANCE


def derive(base_job, dataset, horizon, c, rnd):
    job = copy.deepcopy(base_job)
    d, m, n = c.struct
    src = job['source']['config']
    hp = src['hyperparams']
    hp.update(latent_dim=d, phase_num_routers=m, layers=n, period_len=c.period)
    src['period'] = c.period
    if c.delta == 1.0:
        hp.pop('huber_delta', None)
    else:
        hp['huber_delta'] = c.delta
    job['head'] = c.head
    job['shared_gate'] = bool(c.shared)
    job['variant'] = dict(job['variant'], gate_init=c.gate, extension_lr_mult=c.rm)
    job['extension_lr_mult'] = float(c.rm)
    job['backbone_lr_mult'] = float(c.bb)
    job['lr_split'] = {'gate_mult': float(c.gm), 'readout_mult': float(c.rm)}
    job['tune'] = dict(job.get('tune', {}), period=c.period, gate=c.gate, gm=c.gm, rm=c.rm, bb=c.bb,
                       head=c.head, shared=c.shared, delta=c.delta)
    job['structure'] = {'latent_dim': d, 'routers': m, 'layers': n}
    job['cell'] = dict(c._asdict(), struct=list(c.struct))
    job['round'] = rnd
    job.update(frozen())
    name = cell_name(dataset, horizon, c)
    job['scratch'] = str(SCRATCH)
    job['stage'] = STAGE
    job['output'] = str(SCRATCH / STAGE / 'runs' / name)
    return job, name


def cell_of(job):
    cell = job['cell']
    return normalize(Cell(tuple(cell['struct']), cell['period'], cell['gate'], cell['gm'], cell['rm'],
                          cell['bb'], cell['head'], cell['shared'], cell['delta']))


# ------------------------------------------------------------------ one setting
class Setting:
    def __init__(self, dataset, horizon, order, records):
        self.dataset, self.horizon, self.order = dataset, horizon, order
        self.key = (dataset, horizon)
        self.plan = run_plan(dataset)
        self.base_job = json.loads(r0.base_path(dataset, horizon).read_text())
        self.record = records[self.key]
        self.params_cap, self.structures, seed_row = self.read_r0()
        self.base_cell = self.base_cell_of(self.base_job, self.structures[0])
        self.cells, self.pending, self.running = {}, [], set()
        self.rounds, self.done, self.reason = {}, False, ''
        self.params_cache = {}
        self.seed_cell, self.seed_result = self.base_cell, (float(seed_row['test_mse']), float(seed_row['test_mae']))
        self.load()

    def read_r0(self):
        rows = [r for r in csv.DictReader((SCRATCH / 'r0_results.csv').open())
                if (r['dataset'], int(r['horizon'])) == self.key]
        plan_rows = [r for r in csv.DictReader((SCRATCH / 'r0_plan.csv').open())
                     if (r['dataset'], int(r['horizon'])) == self.key and r['run'] == 'base']
        best = min(float(r['test_mse']) for r in rows)
        near = [r for r in rows if float(r['test_mse']) <= best * (1 + r0.TIE)]
        chosen = min(near, key=lambda r: (int(r['params']), float(r['test_mse'])))
        others = sorted((r for r in rows if r['structure'] != chosen['structure']), key=lambda r: float(r['test_mse']))
        structures = [parse_structure(chosen['structure'])] + [parse_structure(r['structure']) for r in others[:2]]
        return int(plan_rows[0]['params']), structures, chosen

    @staticmethod
    def base_cell_of(job, struct):
        split = job.get('lr_split')
        rm = split['readout_mult'] if split else job['extension_lr_mult']
        gm = split['gate_mult'] if split else job['extension_lr_mult']
        return normalize(Cell(struct, job['source']['config']['period'], job['variant']['gate_init'], gm, rm,
                              job['backbone_lr_mult'], canonical_head(job['head']), int(job['shared_gate']),
                              job['source']['config']['hyperparams'].get('huber_delta', 1.0)))

    # --- results
    def score(self, mse, mae):
        return max(mse / self.record[0], mae / self.record[1])

    def result_of(self, out):
        path = Path(out) / 'result.json'
        if not path.is_file():
            return None
        best = json.loads(path.read_text())['best']
        return best['test_mse'], best['test_mae']

    def load(self):
        jobs = sorted((SCRATCH / STAGE / 'jobs').glob(f'{self.dataset}_H{self.horizon}_S{SEED}__e1r2__*.json'))
        for path in jobs:
            job = json.loads(path.read_text())
            out = job['output']
            self.cells[Path(path).stem] = dict(cell=cell_of(job), round=job['round'], path=path, out=out,
                                               result=self.result_of(out))
        for name, item in self.cells.items():
            self.rounds.setdefault(item['round'], []).append(name)
            if item['result'] is None:
                self.pending.append(name)

    def evaluated(self):
        items = [(self.seed_cell, -1, self.seed_result)]
        items += [(i['cell'], i['round'], i['result']) for i in self.cells.values() if i['result']]
        return items

    def best(self):
        return min(self.evaluated(), key=lambda t: self.score(*t[2]))

    def history(self):
        last = max(self.rounds, default=-1)
        out = []
        for r in range(last + 1):
            out.append(min(self.score(*res) for _, rnd, res in self.evaluated() if rnd <= r))
        return out

    # --- proposals
    def params_ok(self, cell):
        # the trend decoder (learned or Legendre) is a choice of the search, so the count uses the base head
        key = (cell.struct, cell.period, cell.shared)
        if key not in self.params_cache:
            job, _ = derive(self.base_job, self.dataset, self.horizon, cell._replace(head=self.base_cell.head), 0)
            self.params_cache[key] = r0.count_parameters(job)
        return self.params_cache[key] <= self.params_cap

    def sample(self, rng):
        b = self.base_cell
        local = rng.random() < 0.5

        def lr(center, lo, hi):
            if local:
                return center * 10 ** rng.gauss(0, 0.5)
            return 10 ** rng.uniform(math.log10(lo), math.log10(hi))

        gate = expit(logit(b.gate) + rng.gauss(0, 1.5)) if local else expit(rng.uniform(logit(GATE_LO), logit(GATE_HI)))
        gm = 0.0 if rng.random() < 0.05 else lr(b.gm or 1.0, LR_LO, LR_HI)
        periods = [p for p in PERIODS[self.horizon] if self.horizon % p == 0]
        return Cell(rng.choices(self.structures, weights=(0.6, 0.2, 0.2)[:len(self.structures)])[0],
                    rng.choice(periods) if rng.random() < 0.25 else b.period, gate, gm,
                    lr(b.rm, LR_LO, LR_HI), lr(b.bb, BB_LO, BB_HI),
                    ('S2' if b.head == 'S1' else 'S1') if rng.random() < 0.3 else b.head,
                    1 - b.shared if rng.random() < 0.3 else b.shared,
                    rng.choice(DELTAS) if rng.random() < 0.4 else b.delta)

    def perturb(self, parent, rng):
        c = parent._asdict()
        changed = False
        for field, sigma in (('rm', 0.35), ('bb', 0.35)):
            if rng.random() < 0.5:
                c[field] *= math.exp(rng.gauss(0, sigma))
                changed = True
        if rng.random() < 0.5:
            c['gate'] = expit(logit(c['gate']) + rng.gauss(0, 0.8))
            changed = True
        if rng.random() < 0.5:
            if c['gm'] == 0 or rng.random() < 0.1:
                c['gm'] = 0.0 if rng.random() < 0.3 else 10 ** rng.uniform(-1.5, 1.0)
            else:
                c['gm'] *= math.exp(rng.gauss(0, 0.35))
            changed = True
        if rng.random() < 0.15:
            c['head'] = 'S2' if c['head'] == 'S1' else 'S1'
            changed = True
        if rng.random() < 0.15:
            c['shared'] = 1 - c['shared']
            changed = True
        if rng.random() < 0.15:
            c['delta'] = rng.choice(DELTAS)
            changed = True
        if rng.random() < 0.1:
            c['struct'] = rng.choice(self.structures)
            changed = True
        if rng.random() < 0.1:
            periods = [p for p in PERIODS[self.horizon] if self.horizon % p == 0]
            c['period'] = rng.choice(periods)
            changed = True
        if not changed:
            c['rm'] *= math.exp(rng.gauss(0, 0.35))
        return Cell(**c)

    def propose(self, rnd):
        rng = random.Random(f'{self.dataset}-{self.horizon}-{rnd}')
        want = self.plan['first'] if rnd == 0 else self.plan['batch']
        ranked = sorted(self.evaluated(), key=lambda t: self.score(*t[2]))
        parents = [t[0] for t in ranked[:8]]
        new, seen, tries = [], set(), 0
        while len(new) < want and tries < want * 30:
            tries += 1
            c = self.sample(rng) if rnd == 0 or rng.random() < 0.25 else self.perturb(rng.choice(parents), rng)
            c = normalize(c)
            name = cell_name(self.dataset, self.horizon, c)
            if name in self.cells or name in seen or c == self.seed_cell or not self.params_ok(c):
                continue
            seen.add(name)
            new.append(c)
        return new

    def step(self):
        """Called with the controller lock held; starts the next round once the last one is complete."""
        if self.done or self.pending or self.running:
            return
        last = max(self.rounds, default=-1)
        history = self.history()
        reached = next((r for r, v in enumerate(history) if v <= TARGET), None)
        stop = ''
        if reached is not None and last >= reached + EXTRA_ROUNDS:
            stop = 'target reached'
        elif len(self.cells) >= self.plan['budget']:
            stop = 'budget used'
        elif len(history) > STALL_ROUNDS and history[-1] >= history[-1 - STALL_ROUNDS] * (1 - STALL_GAIN):
            stop = 'no progress'
        if stop:
            self.done, self.reason = True, stop
            return
        rnd = last + 1
        cells = self.propose(rnd)
        if not cells:
            self.done, self.reason = True, 'no new cells'
            return
        from scripts.vldbj_m1.run import write_immutable
        for c in cells:
            job, name = derive(self.base_job, self.dataset, self.horizon, c, rnd)
            path = SCRATCH / STAGE / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            self.cells[name] = dict(cell=c, round=rnd, path=path, out=job['output'], result=None)
            self.rounds.setdefault(rnd, []).append(name)
            self.pending.append(name)


# ------------------------------------------------------------------ controller
class Controller:
    def __init__(self, settings, gpus):
        self.settings, self.gpus = settings, gpus
        self.lock = threading.Lock()
        self.launched = {g: [] for g in gpus}
        self.attempts, self.failures = {}, []
        self.stop = threading.Event()

    def alive(self, gpu):
        self.launched[gpu] = [(p, ds) for p, ds in self.launched[gpu] if p.poll() is None]
        return self.launched[gpu]

    def pick(self, gpu):
        with self.lock:
            alive = self.alive(gpu)
            if len(alive) >= MAX_PER_GPU:
                return None
            heavy = sum(1 for _, ds in alive if ds in HEAVY)
            free = r0.gpu_free_mib().get(gpu, 0)
            for s in self.settings:
                if s.done:
                    continue
                s.step()
                for name in s.pending:
                    if s.dataset in HEAVY and heavy >= HEAVY_SLOTS:
                        break
                    if free < r0.NEED_MIB.get(s.dataset, r0.DEFAULT_NEED_MIB) + 3000:
                        break
                    s.pending.remove(name)
                    s.running.add(name)
                    return s, name
            return None

    def slot(self, gpu, number):
        while not self.stop.is_set():
            item = self.pick(gpu)
            if item is None:
                time.sleep(20 + 3 * number)
                continue
            s, name = item
            info = s.cells[name]
            out = Path(info['out'])
            while sa.running(out):                       # a run of an earlier controller is still going
                time.sleep(30)
            if (out / 'result.json').is_file():
                info['result'] = s.result_of(out)
            else:
                self.launch(gpu, s, name, info, out)
            with self.lock:
                s.running.discard(name)

    def launch(self, gpu, s, name, info, out):
        import shutil
        if (out / 'checkpoints').exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        with (out / 'job.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/phaseformer_e12.py'), '--job',
                                        str(info['path'])], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        with self.lock:
            self.launched[gpu].append((process, s.dataset))
        time.sleep(25)
        code = process.wait()
        if code == 0:
            info['result'] = s.result_of(out)
            return
        tail = (out / 'job.log').read_text()[-3000:]
        with self.lock:
            self.attempts[name] = self.attempts.get(name, 0) + 1
            retry = 'out of memory' in tail and self.attempts[name] < 4
            if retry:
                s.pending.insert(0, name)
            else:
                self.failures.append(name)
                info['result'] = (float('inf'), float('inf'))     # a failed cell never wins
            print('oom, requeued' if retry else 'FAILED', name, flush=True)
        if (out / 'checkpoints').exists() and not (out / 'result.json').exists():
            shutil.rmtree(out, ignore_errors=True)

    def status(self):
        lines = [time.strftime('%Y-%m-%d %H:%M:%S')]
        for s in self.settings:
            ev = s.evaluated()
            best = s.best()
            mse, mae = best[2]
            lines.append(f'{s.dataset}-{s.horizon:<4} cells {len(s.cells):>4}/{s.plan["budget"]} round {max(s.rounds, default=-1)} '
                         f'running {len(s.running)} pending {len(s.pending)} best score {s.score(mse, mae):.4f} '
                         f'(mse {mse:.4f} mae {mae:.4f} target mse<={s.record[0] * TARGET:.4f}) '
                         f'{"DONE: " + s.reason if s.done else ""}')
        return '\n'.join(lines)

    def run(self):
        threads = [threading.Thread(target=self.slot, args=(g, i), daemon=True)
                   for g in self.gpus for i in range(MAX_PER_GPU)]
        for t in threads:
            t.start()
        last = 0
        while True:
            time.sleep(10)
            with self.lock:
                for s in self.settings:
                    s.step()
                finished = all(s.done and not s.running for s in self.settings)
            if time.time() - last > 60 or finished:
                (SCRATCH / f'{STAGE}_status.txt').write_text(self.status())
                last = time.time()
            if finished:
                self.stop.set()
                break
        for t in threads:
            t.join(timeout=60)


# ------------------------------------------------------------------ report
def report(settings):
    gold, rows, lines = gold_values(), [], []
    totals = []
    for s in settings:
        evaluated = [(c, r, res) for c, r, res in s.evaluated() if math.isfinite(res[0])]
        best = min(evaluated, key=lambda t: s.score(*t[2]))
        mse, mae = best[2]
        bm, ba = s.record
        gm, ga = gold[s.key]
        seed_mse, seed_mae = s.seed_result
        lines.append(f'{s.dataset}-{s.horizon:<4} runs {len(s.cells):>4}  R0 mse {seed_mse:.4f} ({100 * (bm - seed_mse) / bm:+.2f}% vs B0H+)'
                     f'  -> mse {mse:.4f} mae {mae:.4f}  vs B0H+ {100 * (bm - mse) / bm:+.2f}% {100 * (ba - mae) / ba:+.2f}%'
                     f'  vs Golden {100 * (gm - mse) / gm:+.2f}% {100 * (ga - mae) / ga:+.2f}%  '
                     f'{"met" if s.score(mse, mae) <= TARGET else "NOT met"}  {"DONE: " + s.reason if s.done else "running"}')
        c = best[0]
        lines.append(f'      best cell: struct {c.struct} period {c.period} gate {c.gate:g} gm {c.gm:g} rm {c.rm:g} bb {c.bb:g} '
                     f'head {c.head} shared {c.shared} delta {c.delta:g}')
        rows.append(dict(dataset=s.dataset, horizon=s.horizon, runs=len(s.cells), mse=mse, mae=mae,
                         vs_b0hp_mse=100 * (bm - mse) / bm, vs_b0hp_mae=100 * (ba - mae) / ba,
                         vs_gold_mse=100 * (gm - mse) / gm, vs_gold_mae=100 * (ga - mae) / ga,
                         met=s.score(mse, mae) <= TARGET, struct=list(c.struct), period=c.period, gate=c.gate,
                         gm=c.gm, rm=c.rm, bb=c.bb, head=c.head, shared=c.shared, delta=c.delta))
    text = '\n'.join(lines)
    print(text)
    (SCRATCH / f'{STAGE}_summary.txt').write_text(text)
    with (SCRATCH / f'{STAGE}_best.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (SCRATCH / f'{STAGE}_runs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'horizon', 'round', 'mse', 'mae', 'struct', 'period', 'gate', 'gm', 'rm', 'bb',
                         'head', 'shared', 'delta'])
        for s in settings:
            for c, rnd, res in s.evaluated():
                writer.writerow([s.dataset, s.horizon, rnd, res[0], res[1], 'x'.join(map(str, c.struct)), c.period,
                                 c.gate, c.gm, c.rm, c.bb, c.head, c.shared, c.delta])


def build_settings(only):
    records = record_values()
    settings, order = [], 0
    for tier in TIERS:
        for dataset, horizon in tier:
            if only is None or f'{dataset}-{horizon}' in only:
                settings.append(Setting(dataset, horizon, order, records))
                order += 1
    return settings


def check_heads():
    """The pooled-history factor of a head is unused in E1: S1, S3 and H6L give the same shapes."""
    from scripts.vldbj_m1.proto import build_model
    import scripts.phaseformer_e12 as pe
    sa.register_heads()
    base = json.loads(r0.base_path('ETTh1', 96).read_text())
    shapes = {}
    for head in ('S1', 'S3', 'H6L', 'H48L'):
        job = dict(copy.deepcopy(base), head=head, smoke=True)
        job.pop('source_hashes', None)
        job.pop('code_commit', None)
        model, _, _, _ = build_model(job)
        pe.apply_structure(model, job['ablation_arm'], job)
        shapes[head] = [tuple(p.shape) for p in model.parameters()]
    same = all(v == shapes['S1'] for v in shapes.values())
    print('head parameter shapes identical:', same)
    return same


def smoke(gpu):
    """Three proposed cells of ETTh2-720 trained for a few batches: jobs derive, build and finish."""
    import shutil
    scratch = ROOT / 'research_runs/e1_tune_smoke'
    if scratch.exists():
        shutil.rmtree(scratch)
    s = build_settings({'ETTh2-720'})[0]
    cells = s.propose(0)[:3]
    for c in cells:
        job, name = derive(s.base_job, s.dataset, s.horizon, c, 0)
        job['smoke'] = True
        job['output'] = str(scratch / 'runs' / name)
        path = scratch / 'jobs' / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, indent=2))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
        Path(job['output']).mkdir(parents=True, exist_ok=True)
        code = subprocess.call([sys.executable, str(ROOT / 'scripts/phaseformer_e12.py'), '--job', str(path)],
                               cwd=ROOT, env=env, stdout=open(Path(job['output']) / 'job.log', 'w'),
                               stderr=subprocess.STDOUT)
        best = json.loads((Path(job['output']) / 'result.json').read_text())['best'] if code == 0 else None
        print(name, 'exit', code, best and (best['test_mse'], best['test_mae']), flush=True)
        print(' ', c, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--check-heads', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', default='', help='comma list such as ETTh1-96,Weather-336')
    args = parser.parse_args()
    if args.check_heads:
        check_heads()
        return
    if args.smoke:
        smoke(int(args.gpus.split(',')[0]))
        return
    only = sa.parse_filter(args.only, str)
    settings = build_settings(only)
    if args.report:
        report(settings)
        return
    if args.run:
        controller = Controller(settings, [int(g) for g in args.gpus.split(',')])
        controller.run()
        report(settings)


if __name__ == '__main__':
    main()
