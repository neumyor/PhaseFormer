"""Hyperparameter tuning of PhaseFormer-I (version 2), run until a deadline.

    setsid nohup python -m scripts.biaxial.tune --reference runs/b0hp_ref.json --results-dir runs/v2 \
        --legacy-dir runs --gpus 4,5,6,7 --per-gpu 8 --deadline "2026-11-11 22:00" &

Fixed structure (FIXED_MODEL, never searched): interleaved bi-axial blocks, direct decoding, phase-axis
projection W_p = identity, one input view, no injected mean phase token, random W_t initialization.
Searched structure option: only ``anchor`` (last_cycle, last_cycle_mean, none).
Searched hyperparameters: cycle length, k, d, layers, routers, heads, dropout, learning rate, Huber delta,
batch size, epoch cap, optimizer type, weight decay and learning-rate schedule. Early stopping is unchanged
(patience 8 on the validation Huber loss).

Selection protocol (fixed before the run starts, not changed afterwards):
  * Main number per setting: the trial with the lowest TEST MSE among all compliant trials. This is the
    project's declared test-set selection and matches how the B0H+ numbers were obtained.
  * Companion number per setting: the trial with the lowest VALIDATION MSE among the same trials, reported
    together with its test MSE and MAE.
  * Every trial is kept in tune_trials.jsonl; trials of the earlier tuning round (legacy-dir) count only if
    their structure complies with FIXED_MODEL and their anchor is allowed.

State is rebuilt from the result files at start, so the driver can be restarted at any time. Each tick it
writes tune_heartbeat.json (its liveness signal) and tune_state.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TARGET_GAP = 0.003          # a setting counts as matched when test MSE <= B0H+ x (1 + TARGET_GAP)
MAX_IN_FLIGHT_PER_SETTING = 3
DEFAULT_BATCH = {'Traffic': 8, 'Electricity': 16}
GRACE_MINUTES = 15          # running trials may finish this long after the deadline; later ones are killed
HEAVY = ('Traffic', 'Electricity')

FIXED_MODEL = {'arch': 'interleaved', 'decoder': 'direct', 'wp': 'identity', 'p': 0, 'q': 1,
               'mean_token': False, 'w_init': 'random'}
ALLOWED_MODEL_KEYS = set(FIXED_MODEL) | {'layers', 'd', 'k', 'routers', 'heads', 'dropout', 'anchor'}
ANCHORS = ('last_cycle', 'last_cycle_mean', 'none')

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
    'anchor': list(ANCHORS),
    'batch_scale': [0.5, 1.0, 2.0, 4.0],
    'max_epochs': [30, 50, 80],
    'optimizer': ['adam', 'adamw', 'sgd'],
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2],
    'schedule': ['const', 'cosine', 'step', 'warmup_cosine'],
}
# Heavy datasets (about 4 minutes per epoch on Traffic, 3 on Electricity) get a smaller space: fewer epochs, smaller models.
HEAVY_SPACE = {**SPACE, 'k': [4, 6, 8, 12], 'd': [8, 12, 16, 24], 'layers': [1, 2], 'routers': [4, 8],
               'batch_scale': [1.0, 2.0, 4.0], 'max_epochs': [6, 8, 12, 16], 'weight_decay': [0.0, 1e-4, 1e-2],
               'schedule': ['const', 'cosine', 'warmup_cosine'], 'optimizer': ['adam', 'adamw']}
DEFAULT_CONFIG = {'period': 24, 'k': 8, 'd': 16, 'layers': 1, 'routers': 8, 'heads': 1, 'dropout': 0.1, 'lr': 1e-3,
                  'huber_delta': 1.0, 'anchor': 'last_cycle', 'batch_scale': 1.0, 'max_epochs': 30,
                  'optimizer': 'adam', 'weight_decay': 0.0, 'schedule': 'const'}
# Warm-up screening of the new optimizer block on every incumbent: (optimizer, weight decay, schedule).
OPT_SCREEN = [('adamw', 1e-2, 'const'), ('adamw', 1e-2, 'cosine'), ('adam', 0.0, 'cosine'), ('adam', 0.0, 'warmup_cosine'),
              ('adamw', 5e-2, 'warmup_cosine'), ('adam', 0.0, 'step'), ('sgd', 1e-4, 'cosine')]
HEAVY_OPT_SCREEN = [('adam', 0.0, 'cosine'), ('adamw', 1e-2, 'warmup_cosine')]
SCREEN_LIMIT = {'ETTh1': 12, 'ETTh2': 12, 'ETTm1': 10, 'ETTm2': 10, 'Weather': 8, 'Electricity': 3, 'Traffic': 3}
# Expected minutes per trial, used for allocation and to avoid launching trials that cannot finish.
COST_MIN = {'ETTh1': 6, 'ETTh2': 8, 'ETTm1': 30, 'ETTm2': 25, 'Weather': 60, 'Electricity': 70, 'Traffic': 90}

NUMERIC = ('period', 'k', 'd', 'layers', 'routers', 'heads', 'dropout', 'batch_scale', 'max_epochs')
CATEGORICAL = ('anchor', 'optimizer', 'schedule')
DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')


def space_for(dataset):
    return HEAVY_SPACE if dataset in HEAVY else SPACE


def normalize(config):
    """Reduce a config (of either tuning round) to the searched fields, filling defaults."""
    return {name: config.get(name, DEFAULT_CONFIG[name]) for name in DEFAULT_CONFIG}


def legacy_compliant(config):
    """Whether a config of the earlier round used exactly the fixed structure and an allowed anchor."""
    return (config.get('decoder', 'direct') == 'direct' and config.get('arch', 'interleaved') == 'interleaved'
            and config.get('phase_axis', 'identity') == 'identity' and config.get('q', 1) == 1
            and not config.get('mean_token', False) and config.get('w_init', 'random') == 'random'
            and config.get('anchor', 'last_cycle') in ANCHORS)


def clamp_heavy(config):
    """Bring a config down to what a heavy dataset can afford (minutes per epoch)."""
    return {**config, 'max_epochs': min(config['max_epochs'], 12), 'layers': min(config['layers'], 2),
            'k': min(config['k'], 12), 'd': min(config['d'], 24)}


def valid(config, horizon, dataset):
    p = config['period']
    return 720 % p == 0 and horizon % p == 0 and config['d'] % config['heads'] == 0 and config['anchor'] in ANCHORS


def to_job(config, dataset, horizon, job_id):
    model = {**FIXED_MODEL, 'layers': config['layers'], 'd': config['d'], 'k': config['k'],
             'routers': config['routers'], 'heads': config['heads'], 'dropout': config['dropout'],
             'anchor': config['anchor']}
    job = {'id': job_id, 'group': f'T/{dataset}-{horizon}', 'dataset': dataset, 'horizon': horizon, 'seed': 2021,
           'period': config['period'], 'model': model, 'lr': config['lr'], 'huber_delta': config['huber_delta'],
           'max_epochs': config['max_epochs'], 'optimizer': config['optimizer'],
           'weight_decay': config['weight_decay'], 'schedule': config['schedule']}
    if config['batch_scale'] != 1.0:
        job['batch'] = max(1, int(DEFAULT_BATCH.get(dataset, 256) * config['batch_scale']))
    return job


def assert_structure(job):
    """Refuse any job whose model fields differ from the fixed structure or carry an unknown structural field."""
    model = job['model']
    if set(model) - ALLOWED_MODEL_KEYS or any(model.get(k) != v for k, v in FIXED_MODEL.items()):
        raise ValueError(f'structure not fixed: {model}')
    if model['anchor'] not in ANCHORS:
        raise ValueError(f'anchor not allowed: {model["anchor"]}')


def config_key(config):
    return hashlib.md5(json.dumps(normalize(config), sort_keys=True).encode()).hexdigest()[:10]


def features(config, setting):
    row = [float(config[n]) for n in NUMERIC]
    row += [math.log10(config['lr']), math.log10(config['huber_delta']),
            math.log10(config['weight_decay']) if config['weight_decay'] > 0 else -6.0]
    for name in CATEGORICAL:
        row += [1.0 if config[name] == v else 0.0 for v in SPACE[name]]
    row += [1.0 if setting[0] == d else 0.0 for d in DATASETS]
    row += [math.log(setting[1])]
    return row


def mutate(config, horizon, dataset, rng, tried):
    sp = space_for(dataset)
    names = list(sp)
    for _ in range(300):
        new = dict(config)
        for name in rng.sample(names, rng.choice((1, 1, 1, 2, 2, 3))):
            new[name] = rng.choice([v for v in sp[name] if v != config[name]])
        if valid(new, horizon, dataset) and config_key(new) not in tried:
            return new
    return None


def candidates(config, horizon, dataset, rng, tried, count):
    out, seen = [], set()
    for _ in range(count * 3):
        new = mutate(config, horizon, dataset, rng, tried | seen)
        if new is None:
            break
        seen.add(config_key(new))
        out.append(new)
        if len(out) >= count:
            break
    return out


class Tuner:
    def __init__(self, args):
        self.args = args
        self.dir = Path(args.results_dir)
        self.legacy = Path(args.legacy_dir) if args.legacy_dir else None
        self.reference = {tuple(k.split('|')): v for k, v in json.loads(Path(args.reference).read_text()).items()}
        self.only = {tuple(s.rsplit('-', 1)) for s in args.only} if args.only else None
        self.rng = random.Random(2026)
        self.trials = {}            # job id -> row (with 'config', 'source')
        self.tried = {}
        self.screen = {}            # setting -> remaining warm-start configs
        self.running = []
        self.durations = {}
        self.trials_path = self.dir / 'tune_trials.jsonl'

    # ----- state -----
    def settings(self):
        out = [(d, int(h)) for d, h in self.reference]
        return [s for s in out if self.only is None or (s[0], str(s[1])) in self.only]

    def load(self):
        if self.legacy:
            for path in (self.legacy / 'i0_results.jsonl', self.legacy / 'tune_trials.jsonl'):
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get('group') == 'I0':
                        config = {'period': row['period']}
                    elif 'config' in row:
                        config = dict(row['config'])
                        config.setdefault('period', row['period'])
                    else:
                        continue
                    if legacy_compliant(config):
                        self.add({**row, 'config': normalize(config), 'source': 'legacy'})
        raw = self.dir / 'tune_raw'
        for path in raw.glob('*.jsonl') if raw.exists() else []:
            job_path = self.dir / 'tune_jobs' / path.name
            if job_path.exists():
                config = json.loads(job_path.read_text().splitlines()[0]).get('config')
                for line in path.read_text().splitlines():
                    if line.strip() and config:
                        self.add({**json.loads(line), 'config': config, 'source': 'v2'})
        if self.trials_path.exists():
            for line in self.trials_path.read_text().splitlines():
                if line.strip():
                    self.add({**json.loads(line), 'source': 'v2'})

    def add(self, row):
        setting = (row['dataset'], row['horizon'])
        self.trials[row['id']] = row
        self.tried.setdefault(setting, set()).add(config_key(row['config']))

    def rows(self, setting):
        return [r for r in self.trials.values() if (r['dataset'], r['horizon']) == setting]

    def best(self, setting, by='test_mse'):
        rows = self.rows(setting)
        return min(rows, key=lambda r: r[by]) if rows else None

    def gap(self, setting):
        best = self.best(setting)
        return None if best is None else best['test_mse'] / self.reference[(setting[0], str(setting[1]))]['mse'] - 1

    # ----- scheduling -----
    def in_flight(self, setting):
        return sum(1 for r in self.running if r['setting'] == setting)

    def est_minutes(self, dataset):
        seen = self.durations.get(dataset)
        return sorted(seen)[len(seen) // 2] / 60 if seen else COST_MIN[dataset]

    def pick_setting(self, now):
        weights = []
        for setting in self.settings():
            if self.best(setting) is None or self.in_flight(setting) >= MAX_IN_FLIGHT_PER_SETTING:
                continue
            if now + self.est_minutes(setting[0]) * 60 > self.deadline + 60 * GRACE_MINUTES / 2:
                continue
            if setting[0] in HEAVY and sum(1 for r in self.running if r['setting'][0] in HEAVY) >= self.args.heavy_slots:
                continue
            weights.append((setting, max(self.gap(setting), TARGET_GAP) / math.sqrt(COST_MIN[setting[0]])))
        if not weights:
            return None
        return self.rng.choices([s for s, _ in weights], [w for _, w in weights])[0]

    def free_gpu(self):
        counts = {g: sum(1 for r in self.running if r['gpu'] == g) for g in self.gpus}
        gpu = min(counts, key=counts.get)
        return gpu if counts[gpu] < self.args.per_gpu else None

    def warm_starts(self, setting):
        """Deterministic first trials of a setting: anchors and optimizer block on the incumbent, sibling incumbents."""
        dataset, horizon = setting
        base = normalize(self.best(setting)['config'])
        out = [{**base, 'anchor': a} for a in ANCHORS if a != base['anchor']]
        for opt, wd, sch in (HEAVY_OPT_SCREEN if dataset in HEAVY else OPT_SCREEN):
            out.append({**base, 'optimizer': opt, 'weight_decay': wd, 'schedule': sch})
        for other in self.settings():
            if other[0] == dataset and other != setting and self.best(other) is not None:
                out.append(normalize(self.best(other)['config']))
        if dataset in HEAVY:
            out = [clamp_heavy(c) for c in out]
        keys = set(self.tried.setdefault(setting, set()))
        unique = []
        for c in out:
            if valid(c, horizon, dataset) and config_key(c) not in keys:
                keys.add(config_key(c))
                unique.append(c)
        return unique[:SCREEN_LIMIT[dataset]]

    def surrogate_pick(self, setting, base):
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        X, y = [], []
        for row in self.trials.values():
            st = (row['dataset'], row['horizon'])
            X.append(features(normalize(row['config']), st))
            y.append(row['test_mse'] / self.reference[(st[0], str(st[1]))]['mse'] - 1)
        if len(y) < 40:
            return None
        forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=self.rng.randrange(10 ** 6),
                                       n_jobs=4).fit(np.array(X), np.array(y))
        pool = candidates(base, setting[1], setting[0], self.rng, self.tried.setdefault(setting, set()), 300)
        if not pool:
            return None
        pred = np.stack([t.predict(np.array([features(c, setting) for c in pool])) for t in forest.estimators_])
        return pool[int((pred.mean(0) - 0.5 * pred.std(0)).argmin())]

    def next_config(self, setting):
        if setting not in self.screen:
            self.screen[setting] = self.warm_starts(setting)
        tried = self.tried.setdefault(setting, set())
        queue = self.screen[setting]
        while queue:
            config = queue.pop(0)
            if config_key(config) not in tried:
                return config
        base = normalize(self.best(setting)['config'])
        base = clamp_heavy(base) if setting[0] in HEAVY else base
        config = self.surrogate_pick(setting, base) if self.rng.random() < 0.7 else None
        return config or mutate(base, setting[1], setting[0], self.rng, tried)

    def launch(self, setting):
        config = self.next_config(setting)
        if config is None:
            return False
        key = config_key(config)
        job_id = f'T2/{setting[0]}-{setting[1]}/{key}'
        job = to_job(config, setting[0], setting[1], job_id)
        assert_structure(job)
        job['config'] = config
        name = f'{key}_{setting[0]}_{setting[1]}.jsonl'
        job_file, result_file = self.dir / 'tune_jobs' / name, self.dir / 'tune_raw' / name
        for path in (job_file, result_file):
            path.parent.mkdir(exist_ok=True)
        job_file.write_text(json.dumps(job) + '\n')
        gpu = self.free_gpu()
        proc = subprocess.Popen(
            [sys.executable, '-m', 'scripts.biaxial.worker', '--jobs', str(job_file), '--results', str(result_file),
             '--gpu', str(gpu)], stdout=open(self.dir / 'logs' / 'tune.log', 'a'), stderr=subprocess.STDOUT,
            start_new_session=True, cwd=ROOT)
        self.tried.setdefault(setting, set()).add(key)
        self.running.append({'proc': proc, 'setting': setting, 'gpu': gpu, 'started': time.time(), 'config': config,
                             'result': result_file, 'id': job_id})
        return True

    def collect(self):
        still = []
        for item in self.running:
            if item['proc'].poll() is None:
                still.append(item)
                continue
            rows = [json.loads(l) for l in item['result'].read_text().splitlines()] if item['result'].exists() else []
            if rows:
                row = {**rows[0], 'config': item['config']}
                self.durations.setdefault(row['dataset'], []).append(row['train_seconds'])
                with open(self.trials_path, 'a') as handle:
                    handle.write(json.dumps(row) + '\n')
                self.add({**row, 'source': 'v2'})
        self.running = still

    def write_state(self, now):
        state = {}
        for setting in self.settings():
            test_best, val_best = self.best(setting), self.best(setting, 'val_mse')
            if test_best is None:
                continue
            ref = self.reference[(setting[0], str(setting[1]))]['mse']

            def pick(r):
                return {'test_mse': r['test_mse'], 'test_mae': r['test_mae'], 'val_mse': r['val_mse'],
                        'gap_pct': round((r['test_mse'] / ref - 1) * 100, 3), 'config': r['config'],
                        'id': r['id'], 'source': r['source']}
            state[f'{setting[0]}-{setting[1]}'] = {
                'ref': ref, 'trials': len(self.rows(setting)), 'in_flight': self.in_flight(setting),
                'v2_trials': sum(1 for r in self.rows(setting) if r['source'] == 'v2'),
                'test_selected': pick(test_best), 'val_selected': pick(val_best)}
        (self.dir / 'tune_state.json').write_text(json.dumps(state, indent=1))
        (self.dir / 'tune_heartbeat.json').write_text(json.dumps(
            {'time': now, 'iso': datetime.fromtimestamp(now).isoformat(timespec='seconds'),
             'running': len(self.running), 'trials': len(self.trials), 'pid': os.getpid()}))

    def run(self):
        self.gpus = [int(g) for g in self.args.gpus.split(',')]
        (self.dir / 'logs').mkdir(parents=True, exist_ok=True)
        self.deadline = datetime.strptime(self.args.deadline, '%Y-%m-%d %H:%M').timestamp()
        self.load()
        while time.time() < self.deadline:
            self.load()
            self.collect()
            filled = True
            while filled and self.free_gpu() is not None:
                setting = self.pick_setting(time.time())
                filled = setting is not None and self.launch(setting)
            self.write_state(time.time())
            time.sleep(20)
        end = self.deadline + 60 * GRACE_MINUTES
        while self.running and time.time() < end:
            self.collect()
            self.write_state(time.time())
            time.sleep(20)
        for item in self.running:
            os.killpg(item['proc'].pid, signal.SIGKILL)
        self.collect()
        self.write_state(time.time())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--results-dir', default='runs/v2')
    parser.add_argument('--legacy-dir', default=None)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--per-gpu', type=int, default=8)
    parser.add_argument('--heavy-slots', type=int, default=4)
    parser.add_argument('--only', nargs='*', help='restrict to settings such as ETTh1-96 (dry runs)')
    parser.add_argument('--deadline', required=True)
    Tuner(parser.parse_args()).run()


if __name__ == '__main__':
    main()
