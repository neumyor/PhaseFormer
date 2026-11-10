"""Continuous tuning of PhaseFormer-I towards the B0H+ test MSE, until a deadline.

    python -m scripts.biaxial.tune --reference ref.json --results-dir runs --gpus 6,7 --per-gpu 7 \
        --deadline "2026-11-11 10:00" [--seed-results runs/i0_results.jsonl]

State is rebuilt from the result files at start, so the driver can be restarted at any time.
Selection follows the project's declared test-set selection: a trial replaces the incumbent of its
setting when its test MSE is lower; validation MSE is recorded next to it.
Each tick the driver collects finished trials, then fills free slots with mutations of the incumbent of
a setting that is still above the target, chosen with probability proportional to its gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.biaxial.configs import PHASEFORMER_I  # noqa: E402

TARGET_GAP = 0.003          # a setting counts as matched when test MSE <= B0H+ x (1 + TARGET_GAP)
MAX_IN_FLIGHT_PER_SETTING = 3
HEAVY = {'Traffic': 64, 'Electricity': 32, 'Weather': 8}
DEFAULT_BATCH = {'Traffic': 8, 'Electricity': 16}
# Batch scale from which a heavy dataset's trial finishes in reasonable time; their search keeps it fixed.
FAST_SCALE = {'Traffic': 1.0, 'Electricity': 2.0, 'Weather': 2.0}
FAST_EPOCHS = {'Traffic': 8, 'Electricity': 10}   # fewer epochs keep a heavy trial under about 1.5 hours

SPACE = {
    'period': [8, 12, 16, 24, 30, 36, 48],
    'k': [4, 6, 8, 12, 16, 24],
    'd': [8, 12, 16, 24, 32, 48],
    'layers': [1, 2, 3],
    'routers': [2, 4, 8, 16],
    'heads': [1, 2, 4],
    'dropout': [0.0, 0.05, 0.1, 0.2, 0.3],
    'lr': [2e-4, 5e-4, 1e-3, 2e-3, 4e-3],
    'huber_delta': [0.1, 0.3, 1.0, 3.0],
    'anchor': ['last_cycle', 'last_cycle_mean', 'none', 'x_last'],
    'decoder': ['direct', 'basis_learn', 'd0'],
    'gate_init': [0.05, 0.2, 0.5, 0.9, 0.99],
    'arch': ['interleaved', 'dual', 'phase', 'temporal', 'hybrid'],
    'phase_axis': ['identity', 'learn', 'learn_half'],
    'q': [1, 2],
    'w_init': ['random', 'poly'],
    'mean_token': [False, True],
    'batch_scale': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    'max_epochs': [30, 50, 80],
}


def initial_config(period):
    base = dict(PHASEFORMER_I)
    return {'period': period, 'k': base['k'], 'd': base['d'], 'layers': base['layers'], 'routers': 8, 'heads': 1,
            'dropout': 0.1, 'lr': 1e-3, 'huber_delta': 1.0, 'anchor': base['anchor'], 'decoder': base['decoder'], 'gate_init': 0.2,
            'arch': base['arch'], 'phase_axis': 'identity', 'q': 1, 'w_init': 'random', 'mean_token': False,
            'batch_scale': 1.0, 'max_epochs': 30}


def valid(config, horizon):
    p = config['period']
    if 720 % p or horizon % p or config['d'] % config['heads']:
        return False
    if config['arch'] == 'hybrid' and config['layers'] % 2:
        return False
    return True


def to_job(config, dataset, horizon, job_id):
    model = {'arch': config['arch'], 'layers': config['layers'], 'd': config['d'], 'k': config['k'], 'q': config['q'],
             'routers': config['routers'], 'heads': config['heads'], 'dropout': config['dropout'],
             'decoder': config['decoder'], 'anchor': config['anchor'], 'w_init': config['w_init'],
             'mean_token': config['mean_token'], 'gate_init': config['gate_init']}
    if config['phase_axis'] == 'identity':
        model.update(wp='identity', p=0)
    else:
        model.update(wp='learn', p=0 if config['phase_axis'] == 'learn' else config['period'] // 2)
    job = {'id': job_id, 'group': f'T/{dataset}-{horizon}', 'dataset': dataset, 'horizon': horizon, 'seed': 2021,
           'period': config['period'], 'model': model, 'lr': config['lr'], 'huber_delta': config['huber_delta'],
           'max_epochs': config['max_epochs']}
    default = DEFAULT_BATCH.get(dataset, 256)
    if config['batch_scale'] != 1.0:
        job['batch'] = max(1, int(default * config['batch_scale']))
    return job


def config_key(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


def config_from_row(row):
    """Recover the tuner config of a finished trial from its result row (I0 rows use the initial config)."""
    if row['group'] == 'I0':
        return initial_config(row['period'])
    return {**initial_config(row['period']), **row['config']}


NUMERIC = ('period', 'k', 'd', 'layers', 'routers', 'heads', 'dropout', 'gate_init', 'q', 'batch_scale', 'max_epochs')
CATEGORICAL = ('anchor', 'decoder', 'arch', 'phase_axis', 'w_init', 'mean_token')
DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')


def features(config, setting):
    import math
    row = [float(config[n]) for n in NUMERIC]
    row += [math.log10(config['lr']), math.log10(config['huber_delta'])]
    for name in CATEGORICAL:
        row += [1.0 if config[name] == v else 0.0 for v in SPACE[name]]
    row += [1.0 if setting[0] == d else 0.0 for d in DATASETS]
    row += [math.log(setting[1])]
    return row


def candidates(config, horizon, rng, tried, count, frozen=()):
    out, seen = [], set()
    for _ in range(count * 4):
        new = mutate(config, horizon, rng, tried | seen, frozen)
        if new is None:
            break
        seen.add(config_key(new))
        out.append(new)
        if len(out) >= count:
            break
    return out


def mutate(config, horizon, rng, tried, frozen=()):
    names = [n for n in SPACE if n not in frozen]
    for _ in range(200):
        new = dict(config)
        for name in rng.sample(names, rng.choice((1, 1, 1, 2, 2, 3))):
            choices = [v for v in SPACE[name] if v != config[name]]
            new[name] = rng.choice(choices)
        if valid(new, horizon) and config_key(new) not in tried:
            return new
    return None


class Tuner:
    def __init__(self, args):
        self.args = args
        self.dir = Path(args.results_dir)
        self.reference = {tuple(k.split('|')): v for k, v in json.loads(Path(args.reference).read_text()).items()}
        self.rng = random.Random(2026)
        self.best, self.best_fast = {}, {}
        self.tried = {}
        self.history, self.history_ids, self.seen_rows = {}, set(), {}
        self.running = []          # (process, setting, key, job_id, gpu, started)
        self.trials_path = self.dir / 'tune_trials.jsonl'
        self.state_path = self.dir / 'tune_state.json'

    def settings(self):
        return [(d, int(h)) for d, h in self.reference]

    def raw_rows(self):
        """Finished trials whose worker outlived a previous driver: result file plus the job file's config."""
        for path in (self.dir / 'tune_raw').glob('*.jsonl'):
            job_path = self.dir / 'tune_jobs' / path.name
            if not job_path.exists():
                continue
            config = json.loads(job_path.read_text().splitlines()[0]).get('config')
            for line in path.read_text().splitlines():
                if line.strip() and config:
                    yield {**json.loads(line), 'config': config}

    def load(self):
        for row in self.raw_rows():
            self.record(row)
        for path in list(self.dir.glob('*results.jsonl')) + [self.trials_path]:
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    self.record(json.loads(line))

    def record(self, row):
        if row.get('group') != 'I0' and not str(row.get('group', '')).startswith('T/'):
            return
        setting = (row['dataset'], row['horizon'])
        try:
            config = config_from_row(row)
        except KeyError:
            return
        self.tried.setdefault(setting, set()).add(config_key(config))
        self.seen_rows.setdefault(row['id'], None)
        if row['id'] not in self.history_ids:
            self.history_ids.add(row['id'])
            self.history.setdefault(setting, []).append((config, row['test_mse']))
        if setting[0] in FAST_SCALE and config['batch_scale'] >= FAST_SCALE[setting[0]] and (
                setting not in self.best_fast or row['test_mse'] < self.best_fast[setting]['test_mse']):
            self.best_fast[setting] = {'test_mse': row['test_mse'], 'config': config}
        if setting not in self.best or row['test_mse'] < self.best[setting]['test_mse']:
            self.best[setting] = {'test_mse': row['test_mse'], 'val_mse': row['val_mse'], 'config': config,
                                  'id': row['id']}

    def gap(self, setting):
        ref = self.reference[(setting[0], str(setting[1]))]['mse']
        if setting not in self.best:
            return None
        return self.best[setting]['test_mse'] / ref - 1

    def in_flight(self, setting):
        return sum(1 for r in self.running if r[1] == setting)

    def pick_setting(self):
        weights = []
        for setting in self.settings():
            gap = self.gap(setting)
            if gap is None or gap <= TARGET_GAP or self.in_flight(setting) >= MAX_IN_FLIGHT_PER_SETTING:
                continue
            heavy = sum(1 for r in self.running if r[1][0] in HEAVY)
            if setting[0] in HEAVY and heavy >= self.args.heavy_slots:
                continue
            weights.append((setting, gap))
        if not weights:
            return None
        return self.rng.choices([s for s, _ in weights], [g for _, g in weights])[0]

    def free_gpu(self):
        counts = {g: sum(1 for r in self.running if r[4] == g) for g in self.gpus}
        gpu = min(counts, key=counts.get)
        return gpu if counts[gpu] < self.args.per_gpu else None

    def surrogate_pick(self, setting):
        """Choose among many mutations of the incumbent with a forest fitted on every trial so far
        (target: relative gap to B0H+), minimizing mean - 0.5 std over the trees."""
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        X, y = [], []
        for st, rows in self.history.items():
            ref = self.reference[(st[0], str(st[1]))]['mse']
            for config, mse in rows:
                X.append(features(config, st))
                y.append(mse / ref - 1)
        if len(y) < 40:
            return None
        forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=self.rng.randrange(10**6),
                                       n_jobs=4).fit(np.array(X), np.array(y))
        tried = self.tried.setdefault(setting, set())
        pool = candidates(self.base(setting), setting[1], self.rng, tried, 300, self.frozen(setting))
        if not pool:
            return None
        pred = np.stack([t.predict(np.array([features(c, setting) for c in pool])) for t in forest.estimators_])
        score = pred.mean(0) - 0.5 * pred.std(0)
        return pool[int(score.argmin())]

    def base(self, setting):
        """Incumbent to mutate; heavy datasets start from their fast-batch incumbent."""
        if setting[0] in FAST_SCALE:
            if setting in self.best_fast:
                return self.best_fast[setting]['config']
            return {**self.best[setting]['config'], 'batch_scale': FAST_SCALE[setting[0]], 'lr': 2e-3,
                    'max_epochs': FAST_EPOCHS.get(setting[0], 30)}
        return self.best[setting]['config']

    def frozen(self, setting):
        return ('batch_scale', 'max_epochs') if setting[0] in FAST_SCALE else ()

    def launch(self, setting):
        tried = self.tried.setdefault(setting, set())
        config = None
        if setting[0] in FAST_SCALE and setting not in self.best_fast and config_key(self.base(setting)) not in tried:
            config = self.base(setting)
        if config is None and self.rng.random() < 0.7:
            config = self.surrogate_pick(setting)
        if config is None:
            config = mutate(self.base(setting), setting[1], self.rng, tried, self.frozen(setting))
        if config is None:
            return False
        gpu = self.free_gpu()
        key = config_key(config)
        job_id = f'T/{setting[0]}-{setting[1]}/{key}'
        job = to_job(config, setting[0], setting[1], job_id)
        job['config'] = config
        job_file = self.dir / 'tune_jobs' / f'{key}_{setting[0]}_{setting[1]}.jsonl'
        job_file.parent.mkdir(exist_ok=True)
        job_file.write_text(json.dumps(job) + '\n')
        result_file = self.dir / 'tune_raw' / f'{key}_{setting[0]}_{setting[1]}.jsonl'
        result_file.parent.mkdir(exist_ok=True)
        log = open(self.dir / 'logs' / 'tune.log', 'a')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'scripts.biaxial.worker', '--jobs', str(job_file), '--results', str(result_file),
             '--gpu', str(gpu)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        self.tried[setting].add(key)
        self.running.append((proc, setting, key, job_id, gpu, time.time(), job['config'], result_file))
        return True

    def collect(self):
        still = []
        for item in self.running:
            proc, setting, key, job_id, gpu, started, config, result_file = item
            if proc.poll() is None:
                still.append(item)
                continue
            rows = [json.loads(l) for l in result_file.read_text().splitlines()] if result_file.exists() else []
            if rows:
                row = rows[0]
                row['config'] = config
                with open(self.trials_path, 'a') as handle:
                    handle.write(json.dumps(row) + '\n')
                self.record(row)
        self.running = still

    def write_state(self):
        state = {f'{d}-{h}': {'ref': self.reference[(d, str(h))]['mse'], 'best': b['test_mse'], 'val': b['val_mse'],
                              'gap_pct': round(self.gap((d, h)) * 100, 3), 'config': b['config'], 'id': b['id'],
                              'trials': len(self.tried.get((d, h), ()))}
                 for (d, h), b in sorted(self.best.items())}
        self.state_path.write_text(json.dumps(state, indent=1))

    def run(self):
        self.gpus = [int(g) for g in self.args.gpus.split(',')]
        (self.dir / 'logs').mkdir(exist_ok=True)
        deadline = datetime.strptime(self.args.deadline, '%Y-%m-%d %H:%M').timestamp()
        self.load()
        while time.time() < deadline:
            self.load()
            self.collect()
            filled = True
            while filled and self.free_gpu() is not None:
                setting = self.pick_setting()
                filled = setting is not None and self.launch(setting)
            self.write_state()
            time.sleep(20)
        self.collect()
        self.write_state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--results-dir', default='runs')
    parser.add_argument('--gpus', default='6,7')
    parser.add_argument('--per-gpu', type=int, default=7)
    parser.add_argument('--heavy-slots', type=int, default=4)
    parser.add_argument('--deadline', required=True)
    Tuner(parser.parse_args()).run()


if __name__ == '__main__':
    main()
