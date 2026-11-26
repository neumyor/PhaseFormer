#!/usr/bin/env python3
"""Ablation, trend branches and trend/periodic errors of PhaseFormer+ (E1) on Solar, as done on the seven datasets.

Configurations (--prepare writes final_configs_solar.csv):
  PhaseFormer+   per Solar setting the run of research_runs/e1_newdata_v1 (scripts/e1_newdata.py, seed 2021) with the lowest
                 score of scripts/e1_beat.py, the Solar column of the main table
  PhaseFormer    per Solar setting the arm-pf run of ~/niuyiming/PhaseFormer_solarpm_20261109/research_runs/solarpm_20261109
                 with the lowest validation MSE, the PhaseFormer column of the main table

Experiments, each the procedure of the seven datasets:
  ablation   scripts/e1_paper/ablation.py: the single-gate variant (single_gate) and the linear variant (linear_mk, anchor =
             mean of the last cycle) on the four Solar settings with the configuration of the setting, seed 2021, not
             retuned; the configuration itself once more on Solar-96 (reproduction). The model is built as by the worker
             that trained the configuration (scripts/e1_beat.py: E1 patches of scripts/phaseformer_e12.py, state dimension r).
  branches   scripts/trend_branch/run.py and tune.py on Solar-96 and Solar-192: PhaseFormer (pf, retrained; reproduces the
             main table) and the N-BEATS, ETSformer and BasisFormer branches added to it, the default cell (branch lr x1) and
             the six cells of tune.py; per branch the candidate with the lowest validation trend MSE (cycle-level, P = 24).
  baselines  on Solar-96/192, seed 2021, as in the paper: DLinear and CycleNet by scripts/component_mse/train_baseline.py
             (shared protocol; Solar is 10-minute data, CycleNet cycle 144 as in CycleNet's Solar script, phase of a step =
             minute of the day // 10); SparseTSF by its official code and its official Solar configuration
             (~/niuyiming/exp3_baselines/SparseTSF, scripts/SparseTSF/linear/solar.sh: period 4, batch 256, lr 0.02,
             patience 5, MSE loss) on our Solar.csv with the 7:1:2 split (--data custom), as run_exp3.py did on the
             seven datasets.
  trend      cycle-level MSE_trend, MSE_periodic and MSE_total (P_eval = 24, scripts/component_mse/trend_definitions.py) on
             the test predictions of every model above and of PhaseFormer+ (its checkpoint, not retrained).

Everything lives in research_runs/e1_solar_extra_v1 (predictions in predictions/, in the layout of scripts/component_mse/dump.py).

    python3 scripts/e1_solar_extra.py --prepare
    python3 scripts/e1_solar_extra.py --run --gpus 0,1,2,3,4,5,6,7 --slots 3
    CUDA_VISIBLE_DEVICES=0 python3 scripts/e1_solar_extra.py --export
    CUDA_VISIBLE_DEVICES= python3 scripts/e1_solar_extra.py --efficiency
    python3 scripts/e1_solar_extra.py --report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.e1_newdata as nd  # noqa: E402,F401  (targets of the Solar settings in e1_beat)
import scripts.e1_beat as eb  # noqa: E402

OUT = ROOT / 'research_runs/e1_solar_extra_v1'
NEWDATA = ROOT / 'research_runs/e1_newdata_v1'
PF_RUNS = Path.home() / 'niuyiming/PhaseFormer_solarpm_20261109/research_runs/solarpm_20261109'
DUMP = OUT / 'predictions'
DATASET, SEED, PERIOD = 'Solar', 2021, 24
HS = (96, 192, 336, 720)
TREND_HS = (96, 192)
ABLATION_ARMS = ('single_gate', 'linear_mk')
CHECK = (96,)
BASELINES = ('DLinear', 'CycleNet')
EXP3 = Path.home() / 'niuyiming/exp3_baselines'
PY = sys.executable
BRANCH_CELLS = (('default', 'nb', {}, 1.0), ('default', 'ets', {}, 1.0), ('default', 'basis', {}, 1.0),
                ('nb_d2_lr01', 'nb', {'degree': 2}, 0.1), ('nb_d1_lr03', 'nb', {'degree': 1}, 0.3),
                ('ets_lr03', 'ets', {}, 0.3), ('basis_lr03', 'basis', {}, 0.3),
                ('nb_d2_lr003', 'nb', {'degree': 2}, 0.03), ('nb_d1_lr01', 'nb', {'degree': 1}, 0.1))
BRANCH_NAMES = {'nb': '+N-BEATS', 'ets': '+ETSformer', 'basis': '+BasisFormer'}
NEED_MIB = 8000


# ------------------------------------------------------------------ configurations
def pfplus_runs():
    """Per horizon: (job, run dir, result) of the PhaseFormer+ run of the main table."""
    out = {}
    for h in HS:
        best = None
        for result_path in (NEWDATA / 'search' / 'runs').glob(f'{DATASET}_H{h}_S{SEED}__beat__*/result.json'):
            b = json.loads(result_path.read_text())['best']
            s = eb.score((DATASET, h), b['test_mse'], b['test_mae'])
            if best is None or s < best[0]:
                best = (s, result_path.parent)
        run = best[1]
        choice = OUT / 'solar96_choice.json'
        if h == 96 and choice.is_file():                  # the configuration of the second Solar-96 search (scripts/e1_solar96.py)
            c = json.loads(choice.read_text())
            run = Path(c['run_dir'])
            out[h] = (Path(c['job']), run, json.loads((run / 'result.json').read_text()))
            continue
        out[h] = (NEWDATA / 'search' / 'jobs' / f'{run.name}.json', run, json.loads((run / 'result.json').read_text()))
    return out


def pf_runs():
    """Per horizon: (job, run dir, result) of the PhaseFormer run of the main table (lowest validation MSE)."""
    out = {}
    for h in HS:
        best = None
        for stage in ('grid', 'refine'):
            for result_path in (PF_RUNS / stage / 'runs').glob(f'{DATASET}_H{h}_S{SEED}__pf__*/result.json'):
                record = json.loads(result_path.read_text())
                if best is None or record['best']['val_mse'] < best[2]['best']['val_mse']:
                    best = (PF_RUNS / stage / 'jobs' / f'{result_path.parent.name}.json', result_path.parent, record)
        out[h] = best
    return out


def write_configs():
    rows = []
    for model, runs in (('PhaseFormer+', pfplus_runs()), ('PhaseFormer', pf_runs())):
        for h, (job, run, record) in runs.items():
            rows.append(dict(model=model, dataset=DATASET, horizon=h, job=str(job), run_dir=str(run),
                             test_mse=record['best']['test_mse'], test_mae=record['best']['test_mae'],
                             val_mse=record['best']['val_mse'], params=record.get('parameters', {}).get('total', ''),
                             epochs=record.get('epochs', '')))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'final_configs_solar.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f'{r["model"]:<13} {DATASET}-{r["horizon"]:<4} {r["test_mse"]:.4f}/{r["test_mae"]:.4f}  {Path(r["run_dir"]).name}')


# ------------------------------------------------------------------ jobs
def write(path, job):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(job, indent=1))
    return path


def ablation_jobs():
    paths = []
    for h, (job_path, _, _) in pfplus_runs().items():
        for arm in ABLATION_ARMS + (('baseline',) if h in CHECK else ()):
            job = copy.deepcopy(json.loads(job_path.read_text()))
            name = f'{DATASET}_H{h}_S{SEED}__{arm}'
            job.update(e1_variant=arm, stage=f'ablation_{arm}', scratch=str(OUT), extra_kind='ablation',
                       output=str(OUT / 'ablation' / 'runs' / arm / name))
            paths.append(write(OUT / 'ablation' / 'jobs' / f'{name}.json', job))
    return paths


def branch_jobs():
    frozen = eb.frozen()
    paths = []
    pf = pf_runs()
    for h in TREND_HS:
        source = json.loads(pf[h][0].read_text())['source']
        for tag, kind, options, lr_mult in (('pf', 'pf', {}, 1.0),) + BRANCH_CELLS:
            job = dict(scratch=str(OUT), stage='v1', smoke=False, dataset=DATASET, horizon=h, seed=SEED, arm='pf',
                       protocol='e14', init='scratch', head='S2', shared_gate=False, variant_id=None, variant=None,
                       extension_lr_mult=1.0, backbone_lr_mult=1.0, warm_source=None, epoch_test=False,
                       source=copy.deepcopy(source), trend_branch=kind, extra_kind='branch', **frozen)
            name = f'{DATASET}_H{h}_S{SEED}__{kind}' if tag in ('pf', 'default') else f'{DATASET}_H{h}_S{SEED}__{tag}'
            if tag not in ('pf', 'default'):
                job.update(branch_options=options, branch_lr_mult=lr_mult, tune_tag=tag)
            job['output'] = str(OUT / 'branches' / 'runs' / name)
            paths.append(write(OUT / 'branches' / 'jobs' / f'{name}.json', job))
    return paths


def baseline_jobs():
    paths = []
    for h in TREND_HS:
        for model in BASELINES:
            name = f'{model}__{DATASET}_H{h}_S{SEED}'
            job = dict(extra_kind='baseline', model=model, dataset=DATASET, horizon=h, seed=SEED,
                       output=str(OUT / 'baselines' / 'runs' / name))
            paths.append(write(OUT / 'baselines' / 'jobs' / f'{name}.json', job))
        name = f'SparseTSF__{DATASET}_H{h}_S{SEED}'
        job = dict(extra_kind='sparsetsf', model='SparseTSF', dataset=DATASET, horizon=h, seed=SEED,
                   output=str(OUT / 'baselines' / 'runs' / name))
        paths.append(write(OUT / 'baselines' / 'jobs' / f'{name}.json', job))
    return paths


def prepare():
    write_configs()
    counts = {kind: len(f()) for kind, f in (('ablation', ablation_jobs), ('branches', branch_jobs),
                                             ('baselines', baseline_jobs))}
    print(counts)


# ------------------------------------------------------------------ workers
def ablation_worker(job_path):
    """scripts/e1_paper/ablation.py worker with the model built as by scripts/e1_beat.py."""
    import scripts.b0hp_ablation as abl
    import scripts.b0hp_structure_ablation as sa
    import scripts.phaseformer_e12 as pe
    from scripts.e1_paper.ablation import apply_variant
    from scripts.vldbj_m1 import proto
    sa.register_heads()
    job = json.loads(Path(job_path).read_text())
    linear = job['e1_variant'].startswith('linear')
    attach = abl.make_attach(dict(job, ablation_arm='linear_trend') if linear else job)
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        pe.apply_structure(model, j['ablation_arm'], j)
        rank = int(j.get('rank_override', 3))
        if rank != 3:
            eb.set_state_dimension(model, rank)
        apply_variant(model, j)
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
    proto.run_job(job_path)


def patch_solar_baselines():
    """Solar entries of scripts/component_mse/models.py (10-minute data with a daily cycle of 144 steps)."""
    import numpy as np
    from scripts.component_mse import models
    models.CYCLE_LENGTH[DATASET] = 144
    original = models.cycle_phase

    def cycle_phase(timestamps, dataset):
        if dataset != DATASET:
            return original(timestamps, dataset)
        stamps = np.array([str(s)[:19] for s in timestamps]).astype('datetime64[m]')
        minutes = (stamps - stamps.astype('datetime64[D]')).astype('int64')
        return (minutes // 10).astype('int64') % models.CYCLE_LENGTH[DATASET]

    models.cycle_phase = cycle_phase


def baseline_worker(job_path):
    from scripts.component_mse import dump, train_baseline
    job = json.loads(Path(job_path).read_text())
    patch_solar_baselines()
    sys.argv = ['train_baseline.py', '--model', job['model'], '--dataset', job['dataset'], '--horizon', str(job['horizon']),
                '--seed', str(job['seed']), '--dump-root', str(DUMP)]
    train_baseline.main()
    record = dump.result_path(DUMP, job['model'], job['dataset'], job['horizon'], job['seed'])
    shutil.copyfile(record, Path(job['output']) / 'result.json')


def sparsetsf_worker(job_path):
    """Official SparseTSF, official Solar configuration; pred.npy / true.npy of its test split to output/prediction.npz."""
    import numpy as np
    job = json.loads(Path(job_path).read_text())
    workdir, h, out = EXP3 / 'SparseTSF', job['horizon'], Path(job['output'])
    data = workdir / 'dataset' / 'Solar.csv'
    if not data.exists():
        shutil.copyfile(ROOT / 'resources/all_datasets/solar/Solar.csv', data)
    tag = f'exp3_SparseTSF_{DATASET}_{h}'
    command = [PY, '-u', 'run_longExp.py', '--is_training', '1', '--root_path', './dataset/', '--data_path', 'Solar.csv',
               '--model_id', tag, '--model', 'SparseTSF', '--data', 'custom', '--features', 'M', '--seq_len', '720',
               '--pred_len', str(h), '--period_len', '4', '--model_type', 'linear', '--enc_in', '137', '--train_epochs', '30',
               '--patience', '5', '--itr', '1', '--batch_size', '256', '--learning_rate', '0.02']
    subprocess.check_call(command, cwd=workdir)
    folder = sorted((workdir / 'results').glob(f'{tag}_*'))[-1]
    pred, true = np.load(folder / 'pred.npy'), np.load(folder / 'true.npy')
    np.savez(out / 'prediction.npz', prediction=pred.astype('float32'), truth=true.astype('float32'))
    mse, mae = float(np.mean((pred - true) ** 2)), float(np.mean(np.abs(pred - true)))
    (out / 'result.json').write_text(json.dumps(dict(model='SparseTSF', dataset=DATASET, horizon=h, seed=SEED, test_mse=mse,
                                                     test_mae=mae, command=command, results=str(folder),
                                                     implementation='official code and Solar configuration'), indent=1))


def launch(path, gpu):
    job = json.loads(Path(path).read_text())
    out = Path(job['output'])
    if (out / 'result.json').is_file():
        return 0
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    kind = job['extra_kind']
    if kind == 'branch':
        command = [sys.executable, str(ROOT / 'scripts/trend_branch/run.py'), '--job', str(path)]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), f'--{kind}-job', str(path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (out / 'job.log').open('a') as log:
        return subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def run(gpus, slots):
    # longest first: PhaseFormer+ variants of the long horizons, then the branch runs, then the linear baselines
    queue = sorted(ablation_jobs(), key=lambda p: -int(p.name.split('_H')[1].split('_')[0])) + branch_jobs() + baseline_jobs()
    queue = [p for p in queue if not (Path(json.loads(p.read_text())['output']) / 'result.json').is_file()]
    lock = threading.Lock()
    failed = []

    def next_task():
        with lock:
            if not queue:
                return None
            path = queue.pop(0)

        def done(code, path=path):
            print(('FAILED ' if code else 'done ') + path.stem, flush=True)
            if code:
                failed.append(path.stem)
        return path, done, NEED_MIB

    print(f'{len(queue)} jobs on GPUs {gpus} x {slots}', flush=True)
    eb.launch = launch
    eb.SCRATCH = OUT                       # STOP file of the pool: research_runs/e1_solar_extra_v1/STOP
    eb.run_pool(gpus, slots, next_task)
    print('failed:', failed, flush=True)


# ------------------------------------------------------------------ predictions
def predict(model, args, split):
    import numpy as np
    import torch
    from src.dataset.data_factory import data_provider
    device = torch.device('cuda:0')
    model.to(device).eval()
    dataset, loader = data_provider(args.dataset_args, split)
    predictions, truths = [], []
    with torch.inference_mode():
        for batch in loader:
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
            predictions.append(pred.cpu().numpy())
            truths.append(by[:, -model.pred_len:].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(truths), dataset


def load(model, record):
    import torch
    state = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state)
    return model


def checked_save(name, h, prediction, truth, dataset, record, extra):
    from scripts.component_mse import dump
    mse, mae = dump.test_arrays(prediction, truth)
    recorded = record['best']['test_mse']
    if abs(mse - recorded) > max(1e-5, 1e-4 * recorded):
        raise AssertionError(f'{name} {DATASET}-{h}: exported MSE {mse} != recorded {recorded}')
    dump.save_truth(DUMP, DATASET, h, truth, dataset.scaler)
    dump.save_prediction(DUMP, name, DATASET, h, SEED, prediction)
    dump.save_result(DUMP, name, DATASET, h, SEED, dict(model=name, dataset=DATASET, horizon=h, seed=SEED, test_mse=mse,
                                                       test_mae=mae, val_mse=record['best']['val_mse'], **extra))
    print(json.dumps({'model': name, 'horizon': h, 'test_mse': mse}), flush=True)


def export():
    """Test predictions of PhaseFormer+, PhaseFormer and the selected branch runs; the branch selection on validation."""
    from scripts.component_mse import dump, trend_definitions as td
    from scripts.trend_branch.run import patched_build_model
    import scripts.b0hp_structure_ablation as sa
    import scripts.phaseformer_e12 as pe
    from scripts.vldbj_m1 import proto
    sa.register_heads()
    original = proto.build_model
    for h in TREND_HS:
        job_path, run, record = pfplus_runs()[h]
        job = dict(json.loads(job_path.read_text()), smoke=True)
        job.pop('source_hashes', None)
        job.pop('code_commit', None)
        model, args, _, _ = original(job)
        pe.apply_structure(model, job['ablation_arm'], job)
        if int(job.get('rank_override', 3)) != 3:
            eb.set_state_dimension(model, int(job['rank_override']))
        prediction, truth, dataset = predict(load(model, record), args, 'test')
        checked_save('PhaseFormer+', h, prediction, truth, dataset, record, dict(run_dir=str(run)))

    for h in TREND_HS:
        import numpy as np
        out = OUT / 'baselines' / 'runs' / f'SparseTSF__{DATASET}_H{h}_S{SEED}'
        with np.load(out / 'prediction.npz') as handle:
            prediction, truth = handle['prediction'], handle['truth']
        reference = dump.load_truth(DUMP, DATASET, h)
        if reference.shape != truth.shape or not np.allclose(reference, truth, atol=1e-4):
            raise AssertionError(f'SparseTSF {DATASET}-{h}: its test windows differ from the shared truth array')
        record = json.loads((out / 'result.json').read_text())
        dump.save_prediction(DUMP, 'SparseTSF', DATASET, h, SEED, prediction)
        dump.save_result(DUMP, 'SparseTSF', DATASET, h, SEED, record)

    build = patched_build_model()
    candidates = []
    for job_path in sorted((OUT / 'branches' / 'jobs').glob('*.json')):
        job = json.loads(job_path.read_text())
        record = json.loads((Path(job['output']) / 'result.json').read_text())
        model, args, _, _ = build(job)
        model = load(model, record)
        val_prediction, val_truth, _ = predict(model, args, 'val')
        val_trend = td.score_file(val_prediction, val_truth, PERIOD, ('cycle-level',))['cycle-level']['trend']
        candidates.append(dict(horizon=job['horizon'], branch=job['trend_branch'], cell=job.get('tune_tag', 'default'),
                               polynomial_degree=job.get('branch_options', {}).get('degree', ''),
                               branch_lr_mult=job.get('branch_lr_mult', 1.0), val_mse=record['best']['val_mse'],
                               val_trend_mse_cycle_level=val_trend, test_mse=record['best']['test_mse'],
                               test_mae=record['best']['test_mae'], job=str(job_path)))
        print(json.dumps({k: candidates[-1][k] for k in ('horizon', 'branch', 'cell', 'val_trend_mse_cycle_level')}), flush=True)
    for c in candidates:
        group = [x for x in candidates if (x['horizon'], x['branch']) == (c['horizon'], c['branch'])]
        c['selected'] = int(c is min(group, key=lambda x: x['val_trend_mse_cycle_level']))
    with (OUT / 'branches' / 'branch_candidates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    for c in (c for c in candidates if c['selected']):
        job = json.loads(Path(c['job']).read_text())
        record = json.loads((Path(job['output']) / 'result.json').read_text())
        model, args, _, _ = build(job)
        prediction, truth, dataset = predict(load(model, record), args, 'test')
        name = 'PhaseFormer' if c['branch'] == 'pf' else BRANCH_NAMES[c['branch']]
        checked_save(name, c['horizon'], prediction, truth, dataset, record, dict(cell=c['cell'], run_dir=job['output']))


# ------------------------------------------------------------------ efficiency
def efficiency():
    """Parameters and FLOPs of the four PhaseFormer+ runs, measured by scripts/e1_paper/efficiency.py (CPU, one window)."""
    import types
    import scripts.e1_paper.efficiency as ef
    import scripts.phaseformer_e12 as pe

    def structure(model, arm, job):
        pe.apply_structure(model, arm, job)
        if int(job.get('rank_override', 3)) != 3:
            eb.set_state_dimension(model, int(job['rank_override']))

    ef.pe13 = types.SimpleNamespace(apply_structure=structure)       # built as by the worker that trained the runs
    ef.sa.register_heads()
    rows = []
    for h, (job, run, record) in pfplus_runs().items():
        m = ef.measure(job)
        if m['params_total'] != record['parameters']['total']:
            raise AssertionError(f'{DATASET}-{h}: {m["params_total"]} parameters != recorded {record["parameters"]["total"]}')
        rows.append(dict(dataset=DATASET, horizon=h, run_dir=str(run), **m))
        print(rows[-1], flush=True)
    with (OUT / 'efficiency_solar.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ report
def report():
    from scripts.component_mse import dump, trend_definitions as td
    rows = []
    for h in TREND_HS:
        truth = dump.load_truth(DUMP, DATASET, h)
        for name in ('DLinear', 'SparseTSF', 'CycleNet', 'PhaseFormer', '+N-BEATS', '+ETSformer', '+BasisFormer',
                     'PhaseFormer+'):
            try:
                prediction = dump.load_prediction(DUMP, name, DATASET, h, SEED)
            except FileNotFoundError:
                continue
            s = td.score_file(prediction, truth, PERIOD, ('cycle-level',))['cycle-level']
            record = json.loads(dump.result_path(DUMP, name, DATASET, h, SEED).read_text())
            rows.append(dict(setting=f'{DATASET}-{h}', model=name, mse_trend=s['trend'], mse_periodic=s['periodic'],
                             mse_total=s['total'], test_mse=record['test_mse'], test_mae=record['test_mae'], seed=SEED))
    if rows:
        with (OUT / 'trend_errors_solar.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for r in rows:
            print(f'{r["setting"]:<10} {r["model"]:<13} trend {r["mse_trend"]:.4f} periodic {r["mse_periodic"]:.4f} '
                  f'total {r["mse_total"]:.4f}')
    pfplus = pfplus_runs()
    pf = pf_runs()
    rows = []
    for h in HS:
        rows.append(dict(dataset=DATASET, horizon=h, model='PhaseFormer', mse=pf[h][2]['best']['test_mse'],
                         mae=pf[h][2]['best']['test_mae'], source=str(pf[h][1])))
        for arm in ABLATION_ARMS + ('baseline',):
            path = OUT / 'ablation' / 'runs' / arm / f'{DATASET}_H{h}_S{SEED}__{arm}' / 'result.json'
            if h == 96 and (OUT / 'solar96_choice.json').is_file():
                path = Path(pfplus[96][1]).parents[2] / 'variants' / 'runs' / f'{pfplus[96][1].name}__{arm}' / 'result.json'
            if path.is_file():
                b = json.loads(path.read_text())['best']
                rows.append(dict(dataset=DATASET, horizon=h, model=arm, mse=b['test_mse'], mae=b['test_mae'],
                                 source=str(path.parent)))
        rows.append(dict(dataset=DATASET, horizon=h, model='PhaseFormer+', mse=pfplus[h][2]['best']['test_mse'],
                         mae=pfplus[h][2]['best']['test_mae'], source=str(pfplus[h][1])))
    with (OUT / 'ablation_solar.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f'{r["dataset"]}-{r["horizon"]:<4} {r["model"]:<13} {r["mse"]:.4f}/{r["mae"]:.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--efficiency', action='store_true')
    parser.add_argument('--ablation-job', type=Path)
    parser.add_argument('--baseline-job', type=Path)
    parser.add_argument('--sparsetsf-job', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots', type=int, default=3)
    args = parser.parse_args()
    if args.ablation_job:
        ablation_worker(args.ablation_job)
    elif args.baseline_job:
        baseline_worker(args.baseline_job)
    elif args.sparsetsf_job:
        sparsetsf_worker(args.sparsetsf_job)
    elif args.prepare:
        prepare()
    elif args.run:
        run([int(g) for g in args.gpus.split(',')], args.slots)
    elif args.export:
        export()
    elif args.efficiency:
        efficiency()
    else:
        report()


if __name__ == '__main__':
    main()
