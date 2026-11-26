#!/usr/bin/env python3
"""Tuning of PhaseFormer-E1, stage R3: wider rates, regularization, routing capacity, state dimension.

R3 continues R2 for the settings whose best score is still above 1.005 (score = max(mse / B0H+ mse,
mae / B0H+ mae), test-set selection, seed 2021). A setting becomes active once its R2 search is
finished; its R2 and R0 runs are the starting population. Changes against R2:

  wider rates     bb up to 100, gate and readout multipliers up to 1e4, readout multiplier down to 0.003,
                  gate_init up to 0.9999
  fixed           the period of the record; no switch between E, E1 and E2
  regularization  attention/MLP dropout, position dropout, predictor dropout, attention heads, RevIN affine
  state           dimension r of the trend state in {2, 3, 4, 6} for the learned decoder head
  parameters      at most twice the parameters of the E1 base of the setting (R0 plan);
                  structures are drawn from a grid of (d, M, N_r), not only from the R0 winners

The scheduler and the stopping rules are those of R2 (scripts/e1_tune_r2.py).

    python3 scripts/e1_tune_r3.py --check
    python3 scripts/e1_tune_r3.py --run --gpus 4,5,6,7
    python3 scripts/e1_tune_r3.py --report
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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402
import scripts.e1_tune as r0  # noqa: E402
import scripts.e1_tune_r2 as r2  # noqa: E402

SCRATCH = r0.SCRATCH
STAGE = 'r3'
SEED = r0.SEED
TARGET = r2.TARGET
EXTRA_ROUNDS = r2.EXTRA_ROUNDS
STALL_ROUNDS, STALL_GAIN = r2.STALL_ROUNDS, r2.STALL_GAIN
PARAM_FACTOR = 2.0
GATE_LO, GATE_HI = 1e-4, 0.9999
GM_LO, GM_HI = 0.03, 1e4
RM_LO, RM_HI = 0.003, 1e4
BB_LO, BB_HI = 0.1, 100.0
DELTAS = r2.DELTAS
ATTN_DROPOUTS, POS_DROPOUTS, PRED_DROPOUTS = (0.0, 0.05, 0.1, 0.2, 0.3), (0.0, 0.1, 0.2), (0.0, 0.1, 0.2)
DEFAULT_ATTN_DROPOUT = 0.1
RANKS = (2, 3, 4, 6)
DEFAULT_RANK = 3
D_VALUES, M_VALUES, N_VALUES = (2, 3, 4, 6, 8, 12, 16, 24, 32), (2, 4, 6, 8, 16), (1, 2, 3)
CHANNELS = {'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7, 'Weather': 21, 'Electricity': 321, 'Traffic': 862}
EXTRA_SETTINGS = ((('Traffic', 192), False),)      # not part of R2 (score 1.007 at R0)
Cell = namedtuple('Cell', 'struct gate gm rm bb head shared delta attn_drop pos_drop pred_drop heads affine rank')
SCRIPT = ROOT / 'scripts/phaseformer_e13.py'


def run_plan(dataset):
    if dataset.startswith('ETT'):
        return dict(budget=300, first=48, batch=16)
    if dataset == 'Weather':
        return dict(budget=80, first=24, batch=8)
    return dict(budget=40, first=12, batch=8)


def estimate_params(dataset, horizon, period, struct, head, shared, rank, affine, position=True):
    """Parameter count of an E1 model (checked against the counted models in --check)."""
    d, m, n = struct
    c = CHANNELS[dataset]
    k, kp = -(-720 // period), horizon // period
    total = k * d + d                                              # phase token embedding
    total += n * (m * d + (period * d if position else 0) + 16 * d * d + 17 * d)   # routers, positions, 2 attentions, MLP, norms
    total += (n - 1) * ((d * k + k) + (k * d + d + 2 * d))         # readback and re-embed between layers
    total += d * kp + kp                                           # phase decoder
    total += d * rank + rank                                       # pooling to the trend state
    total += (rank * horizon + horizon) if head == 'S1' else (rank * 3 + 3)      # trend decoder
    total += 2 * (1 if shared else c)                              # level and shape gates
    if affine:
        total += 2 * c
    return total


def normalize(c):
    gm = 0.0 if c.gm <= 0 else min(GM_HI, max(GM_LO, r2.g4(c.gm)))
    struct = tuple(c.struct)
    heads = int(c.heads) if int(c.heads) > 0 and struct[0] % int(c.heads) == 0 else 1
    rank = int(c.rank) if c.head == 'S1' else DEFAULT_RANK
    return Cell(struct, min(GATE_HI, max(GATE_LO, round(float(c.gate), 5))), gm,
                min(RM_HI, max(RM_LO, r2.g4(c.rm))), min(BB_HI, max(BB_LO, r2.g4(c.bb))), c.head, int(c.shared),
                min(DELTAS, key=lambda x: abs(x - float(c.delta))),
                min(ATTN_DROPOUTS, key=lambda x: abs(x - float(c.attn_drop))),
                min(POS_DROPOUTS, key=lambda x: abs(x - float(c.pos_drop))),
                min(PRED_DROPOUTS, key=lambda x: abs(x - float(c.pred_drop))), heads, int(c.affine), rank)


def cell_name(dataset, horizon, c):
    d, m, n = c.struct
    return (f'{dataset}_H{horizon}_S{SEED}__e1r3__d{d}_M{m}_N{n}_g{c.gate:g}_gm{c.gm:g}_rm{c.rm:g}_bb{c.bb:g}'
            f'_{c.head}_{"sh" if c.shared else "pc"}_dl{c.delta:g}_ad{c.attn_drop:g}_pd{c.pos_drop:g}'
            f'_qd{c.pred_drop:g}_h{c.heads}_{"af" if c.affine else "na"}_r{c.rank}')


def derive(base_job, dataset, horizon, c, rnd):
    job = copy.deepcopy(base_job)
    d, m, n = c.struct
    src = job['source']['config']
    hp = src['hyperparams']
    hp.update(latent_dim=d, phase_num_routers=m, layers=n, phase_attn_heads=c.heads)
    if c.delta == 1.0:
        hp.pop('huber_delta', None)
    else:
        hp['huber_delta'] = c.delta
    if c.attn_drop != DEFAULT_ATTN_DROPOUT:
        hp['phase_attn_dropout'] = c.attn_drop
    if c.pos_drop:
        hp['phase_pos_dropout'] = c.pos_drop
    if c.pred_drop:
        hp['predictor_dropout'] = c.pred_drop
    if c.affine:
        hp['revin_affine'] = True
    job['head'] = c.head
    job['shared_gate'] = bool(c.shared)
    job['variant'] = dict(job['variant'], gate_init=c.gate, extension_lr_mult=c.rm)
    job['extension_lr_mult'] = float(c.rm)
    job['backbone_lr_mult'] = float(c.bb)
    job['lr_split'] = {'gate_mult': float(c.gm), 'readout_mult': float(c.rm)}
    job['tune'] = dict(job.get('tune', {}), gate=c.gate, gm=c.gm, rm=c.rm, bb=c.bb, head=c.head,
                       shared=c.shared, delta=c.delta)
    if c.head == 'S1' and c.rank != DEFAULT_RANK:
        job['rank_override'] = c.rank
    job['structure'] = {'latent_dim': d, 'routers': m, 'layers': n}
    job['cell'] = dict(c._asdict(), struct=list(c.struct))
    job['round'] = rnd
    job.update(r2.frozen())
    name = cell_name(dataset, horizon, c)
    job['scratch'] = str(SCRATCH)
    job['stage'] = STAGE
    job['output'] = str(SCRATCH / STAGE / 'runs' / name)
    return job, name


def cell_of(job):
    c = job['cell']
    return normalize(Cell(tuple(c['struct']), c['gate'], c['gm'], c['rm'], c['bb'], c['head'], c['shared'],
                          c['delta'], c['attn_drop'], c['pos_drop'], c['pred_drop'], c['heads'], c['affine'],
                          c['rank']))


def from_r2(c, base_heads):
    return normalize(Cell(c.struct, c.gate, c.gm, c.rm, c.bb, c.head, c.shared, c.delta, DEFAULT_ATTN_DROPOUT, 0.0,
                          0.0, base_heads, 0, DEFAULT_RANK))


# ------------------------------------------------------------------ one setting
class Setting:
    def __init__(self, dataset, horizon, needs_r2=True):
        self.dataset, self.horizon, self.key = dataset, horizon, (dataset, horizon)
        self.needs_r2 = needs_r2
        self.plan = run_plan(dataset)
        self.base_job = json.loads(r0.base_path(dataset, horizon).read_text())
        self.record = r2.record_values()[self.key]
        self.params_cap, self.structures, seed_row = r2.Setting.read_r0(self)
        self.params_cap = int(PARAM_FACTOR * self.params_cap)
        hp = self.base_job['source']['config']['hyperparams']
        self.base_heads = hp.get('phase_attn_heads', 1)
        self.period = self.base_job['source']['config']['period']
        base = r2.Setting.base_cell_of(self.base_job, self.structures[0])
        self.base_head, self.base_shared = base.head, base.shared
        self.seed_cell = from_r2(base, self.base_heads)
        self.seed_result = (float(seed_row['test_mse']), float(seed_row['test_mae']))
        self.position = self.position_flag(base)
        self.grid = self.feasible_grid()
        self.cells, self.pending, self.running, self.rounds = {}, [], set(), {}
        self.population = []
        self.active, self.done, self.reason = False, False, ''
        self.loaded = False

    def position_flag(self, base):
        """Whether the routing layers carry position embeddings: the flag that reproduces the R0 base count."""
        rows = [r for r in csv.DictReader((SCRATCH / 'r0_plan.csv').open())
                if (r['dataset'], int(r['horizon'])) == self.key and r['run'] == 'base']
        target = int(rows[0]['params'])
        struct = r2.parse_structure(rows[0]['structure'])
        for flag in (True, False):
            if estimate_params(self.dataset, self.horizon, self.period, struct, base.head, base.shared,
                               DEFAULT_RANK, 0, flag) == target:
                return flag
        raise ValueError(f'{self.key}: the parameter estimate does not reproduce the R0 base count {target}')

    def feasible_grid(self):
        found = {tuple(s) for s in self.structures}
        for d in D_VALUES:
            for m in M_VALUES:
                for n in N_VALUES:
                    if self.estimate(Cell((d, m, n), 0.5, 1, 1, 1, self.base_head, self.base_shared, 1.0,
                                          0.1, 0, 0, 1, 0, DEFAULT_RANK)) <= self.params_cap:
                        found.add((d, m, n))
        return sorted(found)

    def estimate(self, c):
        return estimate_params(self.dataset, self.horizon, self.period, c.struct, c.head, c.shared, c.rank, c.affine,
                               self.position)

    # --- results
    def score(self, mse, mae):
        return max(mse / self.record[0], mae / self.record[1])

    @staticmethod
    def result_of(out):
        path = Path(out) / 'result.json'
        if not path.is_file():
            return None
        best = json.loads(path.read_text())['best']
        mse, mae = best['test_mse'], best['test_mae']
        return (mse, mae) if math.isfinite(mse) and math.isfinite(mae) else (float('inf'), float('inf'))

    def load(self):
        """Starting population: the R0 winner with the recorded rates and every finished R2 run."""
        self.population = [(self.seed_cell, -1, self.seed_result)]
        for path in sorted((SCRATCH / 'r2' / 'jobs').glob(f'{self.dataset}_H{self.horizon}_S{SEED}__e1r2__*.json')):
            job = json.loads(path.read_text())
            res = self.result_of(job['output'])
            if res is None or not math.isfinite(res[0]) or job['cell']['period'] != self.period:
                continue
            self.population.append((from_r2(r2.cell_of(job), self.base_heads), -1, res))
        for path in sorted((SCRATCH / STAGE / 'jobs').glob(f'{self.dataset}_H{self.horizon}_S{SEED}__e1r3__*.json')):
            job = json.loads(path.read_text())
            name = path.stem
            res = self.result_of(job['output'])
            self.cells[name] = dict(cell=cell_of(job), round=job['round'], path=path, out=job['output'], result=res)
            self.rounds.setdefault(job['round'], []).append(name)
            if res is None:
                self.pending.append(name)
        self.loaded = True

    def evaluated(self):
        if not self.loaded:
            return [(self.seed_cell, -1, self.seed_result)]
        return self.population + [(i['cell'], i['round'], i['result']) for i in self.cells.values() if i['result']]

    def best(self):
        return min(self.evaluated(), key=lambda t: self.score(*t[2]))

    def history(self):
        last = max(self.rounds, default=-1)
        return [min(self.score(*res) for _, rnd, res in self.evaluated() if rnd <= r) for r in range(last + 1)]

    def r2_finished(self):
        if not self.needs_r2:
            return True
        status = SCRATCH / 'r2_status.txt'
        if not status.is_file():
            return False
        return any(line.startswith(f'{self.dataset}-{self.horizon} ') and 'DONE' in line
                   for line in status.read_text().splitlines())

    # --- proposals
    def params_ok(self, c):
        return self.estimate(c) <= self.params_cap

    def valid_heads(self, d):
        return [h for h in (1, 2, 4) if d % h == 0]

    def sample(self, rng, center):
        local = rng.random() < 0.5

        def lr(value, lo, hi):
            if local:
                return value * 10 ** rng.gauss(0, 0.6)
            return 10 ** rng.uniform(math.log10(lo), math.log10(hi))

        struct = (rng.choices(self.structures, weights=(0.6, 0.2, 0.2)[:len(self.structures)])[0]
                  if rng.random() < 0.4 else rng.choice(self.grid))
        gate = (expit(logit(center.gate) + rng.gauss(0, 1.5)) if local
                else expit(rng.uniform(logit(GATE_LO), logit(GATE_HI))))
        head = ('S2' if center.head == 'S1' else 'S1') if rng.random() < 0.3 else center.head
        c = Cell(struct, gate, 0.0 if rng.random() < 0.05 else lr(center.gm or 1.0, GM_LO, GM_HI),
                 lr(center.rm, RM_LO, RM_HI), lr(center.bb, BB_LO, BB_HI), head,
                 1 - center.shared if rng.random() < 0.3 else center.shared,
                 rng.choice(DELTAS) if rng.random() < 0.4 else center.delta,
                 rng.choice(ATTN_DROPOUTS), rng.choice(POS_DROPOUTS), rng.choice(PRED_DROPOUTS),
                 rng.choice(self.valid_heads(struct[0])), int(rng.random() < 0.3), rng.choice(RANKS))
        return c

    def neighbor(self, struct, rng):
        axis = rng.randrange(3)
        values = (D_VALUES, M_VALUES, N_VALUES)[axis]
        current = struct[axis]
        index = values.index(current) if current in values else min(range(len(values)), key=lambda i: abs(values[i] - current))
        index = min(len(values) - 1, max(0, index + rng.choice((-1, 1))))
        new = list(struct)
        new[axis] = values[index]
        return tuple(new)

    def perturb(self, parent, rng, only_new=False):
        c = parent._asdict()
        if not only_new:
            for field, sigma in (('rm', 0.4), ('bb', 0.4)):
                if rng.random() < 0.5:
                    c[field] *= math.exp(rng.gauss(0, sigma))
            if rng.random() < 0.5:
                c['gate'] = expit(logit(c['gate']) + rng.gauss(0, 0.8))
            if rng.random() < 0.5:
                if c['gm'] == 0 or rng.random() < 0.1:
                    c['gm'] = 0.0 if rng.random() < 0.3 else 10 ** rng.uniform(-1.5, 1.5)
                else:
                    c['gm'] *= math.exp(rng.gauss(0, 0.4))
            if rng.random() < 0.12:
                c['head'] = 'S2' if c['head'] == 'S1' else 'S1'
            if rng.random() < 0.12:
                c['shared'] = 1 - c['shared']
            if rng.random() < 0.12:
                c['delta'] = rng.choice(DELTAS)
            if rng.random() < 0.2:
                c['struct'] = self.neighbor(c['struct'], rng)
        for field, values, prob in (('attn_drop', ATTN_DROPOUTS, 0.4), ('pos_drop', POS_DROPOUTS, 0.3),
                                    ('pred_drop', PRED_DROPOUTS, 0.3), ('rank', RANKS, 0.35)):
            if rng.random() < (prob if not only_new else 0.6):
                c[field] = rng.choice(values)
        if rng.random() < (0.15 if not only_new else 0.4):
            c['affine'] = 1 - c['affine']
        if rng.random() < 0.15:
            c['heads'] = rng.choice(self.valid_heads(c['struct'][0]))
        return Cell(**c)

    def propose(self, rnd):
        rng = random.Random(f'r3-{self.dataset}-{self.horizon}-{rnd}')
        want = self.plan['first'] if rnd == 0 else self.plan['batch']
        ranked = sorted(self.evaluated(), key=lambda t: self.score(*t[2]))
        parents = [t[0] for t in ranked[:8]]
        known = {cell_name(self.dataset, self.horizon, t[0]) for t in self.evaluated()}
        new, seen, tries = [], set(), 0
        while len(new) < want and tries < want * 40:
            tries += 1
            roll = rng.random()
            if roll < 0.2:
                c = self.sample(rng, parents[0])
            elif roll < 0.6:
                c = self.perturb(rng.choice(parents), rng)
            else:
                c = self.perturb(parents[0] if rng.random() < 0.5 else rng.choice(parents), rng, only_new=True)
            c = normalize(c)
            name = cell_name(self.dataset, self.horizon, c)
            if name in self.cells or name in seen or name in known or not self.params_ok(c):
                continue
            seen.add(name)
            new.append(c)
        return new

    def step(self):
        """Called with the controller lock held."""
        if self.done:
            return
        if not self.active:
            if not self.r2_finished():
                return
            self.load()
            self.active = True
            if self.score(*self.best()[2]) <= TARGET:
                self.done, self.reason = True, 'already met'
                return
        if self.pending or self.running:
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


def logit(x):
    return math.log(x / (1 - x))


def expit(x):
    return 1 / (1 + math.exp(-x))


# ------------------------------------------------------------------ controller
class Controller(r2.Controller):

    def launch(self, gpu, s, name, info, out):
        import shutil
        if (out / 'checkpoints').exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        with (out / 'job.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(SCRIPT), '--job', str(info['path'])], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
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
                info['result'] = (float('inf'), float('inf'))
            print('oom, requeued' if retry else 'FAILED', name, flush=True)
        if (out / 'checkpoints').exists() and not (out / 'result.json').exists():
            shutil.rmtree(out, ignore_errors=True)

    def status(self):
        lines = [time.strftime('%Y-%m-%d %H:%M:%S')]
        for s in self.settings:
            if not s.active:
                lines.append(f'{s.dataset}-{s.horizon:<4} waiting for R2' if not s.done else
                             f'{s.dataset}-{s.horizon:<4} DONE: {s.reason}')
                continue
            mse, mae = s.best()[2]
            lines.append(f'{s.dataset}-{s.horizon:<4} cells {len(s.cells):>4}/{s.plan["budget"]} round {max(s.rounds, default=-1)} '
                         f'running {len(s.running)} pending {len(s.pending)} best score {s.score(mse, mae):.4f} '
                         f'(mse {mse:.4f} mae {mae:.4f} target mse<={s.record[0] * TARGET:.4f}) '
                         f'{"DONE: " + s.reason if s.done else ""}')
        return '\n'.join(lines)

    def run(self):
        import threading
        threads = [threading.Thread(target=self.slot, args=(g, i), daemon=True)
                   for g in self.gpus for i in range(r2.MAX_PER_GPU)]
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


# ------------------------------------------------------------------ report, checks
def build_settings(only):
    out = []
    for tier in r2.TIERS:
        for dataset, horizon in tier:
            if only is None or f'{dataset}-{horizon}' in only:
                out.append(Setting(dataset, horizon))
    for (dataset, horizon), needs in EXTRA_SETTINGS:
        if only is None or f'{dataset}-{horizon}' in only:
            out.append(Setting(dataset, horizon, needs_r2=needs))
    return out


def report(settings):
    gold, lines, rows = r2.gold_values(), [], []
    for s in settings:
        if not s.loaded and (s.done or s.r2_finished()):
            s.load()
        evaluated = [t for t in s.evaluated() if math.isfinite(t[2][0])]
        c, _, (mse, mae) = min(evaluated, key=lambda t: s.score(*t[2]))
        bm, ba = s.record
        gm, ga = gold[s.key]
        state = ('DONE: ' + s.reason) if s.done else ('running' if s.active else 'waiting for R2')
        lines.append(f'{s.dataset}-{s.horizon:<4} r3 runs {len(s.cells):>4}  mse {mse:.4f} mae {mae:.4f}  '
                     f'vs B0H+ {100 * (bm - mse) / bm:+.2f}% {100 * (ba - mae) / ba:+.2f}%  '
                     f'vs Golden {100 * (gm - mse) / gm:+.2f}% {100 * (ga - mae) / ga:+.2f}%  '
                     f'{"met" if s.score(mse, mae) <= TARGET else "NOT met"}  {state}')
        lines.append(f'      best cell: {dict(c._asdict())}')
        rows.append(dict(dataset=s.dataset, horizon=s.horizon, runs=len(s.cells), mse=mse, mae=mae,
                         vs_b0hp_mse=100 * (bm - mse) / bm, vs_b0hp_mae=100 * (ba - mae) / ba,
                         vs_gold_mse=100 * (gm - mse) / gm, vs_gold_mae=100 * (ga - mae) / ga,
                         met=s.score(mse, mae) <= TARGET, **dict(c._asdict(), struct=list(c.struct))))
    text = '\n'.join(lines)
    print(text)
    (SCRATCH / f'{STAGE}_summary.txt').write_text(text)
    with (SCRATCH / f'{STAGE}_best.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (SCRATCH / f'{STAGE}_runs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'horizon', 'round', 'mse', 'mae'] + list(Cell._fields))
        for s in settings:
            for c, rnd, res in s.evaluated():
                writer.writerow([s.dataset, s.horizon, rnd, res[0], res[1]] + ['x'.join(map(str, c.struct))] +
                                [getattr(c, f) for f in Cell._fields[1:]])


def check():
    """The parameter estimate equals the counted parameters of every R0 plan row."""
    rows = list(csv.DictReader((SCRATCH / 'r0_plan.csv').open()))
    bad, total = 0, 0
    for dataset, horizon in {(r['dataset'], int(r['horizon'])) for r in rows}:
        job = json.loads(r0.base_path(dataset, horizon).read_text())
        base = r2.Setting.base_cell_of(job, (4, 8, 3))
        period = job['source']['config']['period']
        position = Setting(dataset, horizon).position
        for r in rows:
            if (r['dataset'], int(r['horizon'])) != (dataset, horizon):
                continue
            d, m, n = r2.parse_structure(r['structure'])
            est = estimate_params(dataset, horizon, period, (d, m, n), base.head, base.shared, DEFAULT_RANK, 0, position)
            total += 1
            if est != int(r['params']):
                bad += 1
                if bad <= 5:
                    print('mismatch', dataset, horizon, r['structure'], est, r['params'])
    print(f'estimate matches {total - bad}/{total} plan rows')
    return bad == 0


def smoke(gpu):
    """Cells with the new options (rank, affine, heads, dropouts) train for a few batches."""
    import shutil
    scratch = ROOT / 'research_runs/e1_tune_smoke3'
    if scratch.exists():
        shutil.rmtree(scratch)
    s = Setting('ETTh1', 192)
    b = s.seed_cell._replace(head='S1')
    cells = [normalize(b._replace(rank=6, attn_drop=0.3)),
             normalize(b._replace(rank=2, affine=1, pos_drop=0.2, pred_drop=0.2)),
             normalize(b._replace(struct=(8, 4, 2), heads=2, rank=4, attn_drop=0.0))]
    for c in cells:
        job, name = derive(s.base_job, s.dataset, s.horizon, c, 0)
        job['smoke'] = True
        job['output'] = str(scratch / 'runs' / name)
        path = scratch / 'jobs' / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, indent=2))
        Path(job['output']).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
        code = subprocess.call([sys.executable, str(SCRIPT), '--job', str(path)], cwd=ROOT, env=env,
                               stdout=open(Path(job['output']) / 'job.log', 'w'), stderr=subprocess.STDOUT)
        best = json.loads((Path(job['output']) / 'result.json').read_text())['best'] if code == 0 else None
        print(name, 'exit', code, best and (best['test_mse'], best['test_mae']), 'est params', s.estimate(c), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', default='', help='comma list such as ETTh1-96,Weather-336')
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    if args.smoke:
        smoke(int(args.gpus.split(',')[0]))
        return
    settings = build_settings(sa.parse_filter(args.only, str))
    if args.report:
        report(settings)
        return
    if args.run:
        r2.MAX_PER_GPU, r2.HEAVY_SLOTS = 6, 5
        Controller(settings, [int(g) for g in args.gpus.split(',')]).run()
        report(settings)


if __name__ == '__main__':
    main()
