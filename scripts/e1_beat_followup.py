#!/usr/bin/env python3
"""The paper's analyses on the configurations that scripts/e1_beat.py selected for six settings.

The six new configurations (NEW below, runs of research_runs/e1_beat_v1/search) replace the selected configurations of
ETTh1-336, ETTh2-192, ETTh2-336, ETTh2-720, Weather-336 and Weather-720. Every analysis of the paper that is built on the
selected configuration of one of these settings is rerun on the new one, with the variants of the original scripts:

  ablation     single gate and linear trend layer anchored on the last cycle mean (scripts/e1_paper/ablation.py), six settings
  hp           routers M in {1, 2, 4, 8, 16}, interaction (full attention, linear mixing, none), input length L in
               {96, 192, 336} (scripts/e1_paper/hyperparameter_study.py), Weather-336
  trend_hp     other form of the basis, learned basis with r in {1, 2, 3, 4, 6}, g0 in {0.01, 0.1, 0.5, 0.9} for both
               gates, other sharing of the gates (tab:hyper_trend), ETTh2-336 and Weather-336
  analyze      trend and periodic errors of ETTh2-192 (cycle level, P_eval = 24, scripts/e1_paper/trend_errors.py),
               parameters and FLOPs of the six configurations and of the Weather-336 variants of hp (CPU)

    python3 scripts/e1_beat_followup.py --run --gpus 2 --slots 12
    python3 scripts/e1_beat_followup.py --analyze
    python3 scripts/e1_beat_followup.py --report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.e1_beat as eb  # noqa: E402

OUT = eb.SCRATCH / 'followup'
NEW = {
    ('ETTh1', 336): 'ETTh1_H336_S2021__beat__d8_M2_N1_h2_p24_S2_r3_sh_g0.98493_gm1449_rm2.734_bb3.053_dl0.3_ad0.05_pd0_qd0.2_na',
    ('ETTh2', 192): 'ETTh2_H192_S2021__beat__d8_M8_N1_h1_p24_S1_r1_pc_g0.0114_gm2.525_rm0.1215_bb0.5576_dl0.3_ad0.1_pd0_qd0_af',
    ('ETTh2', 336): 'ETTh2_H336_S2021__beat__d8_M2_N1_h1_p24_S2_r4_pc_g0.82071_gm32.89_rm0.6919_bb1.221_dl1_ad0_pd0_qd0_na',
    ('ETTh2', 720): 'ETTh2_H720_S2021__beat__d4_M2_N1_h1_p24_S1_r6_pc_g0.9988_gm214_rm1.536_bb0.9258_dl0.3_ad0.2_pd0.2_qd0_na',
    ('Weather', 336): 'Weather_H336_S2021__beat__d8_M8_N2_h1_p24_S2_r3_pc_g0.00042_gm253.3_rm0.3576_bb1.403_dl1_ad0.1_pd0_qd0_af',
    ('Weather', 720): 'Weather_H720_S2021__beat__d8_M8_N1_h1_p36_S2_r3_pc_g0.03382_gm826.5_rm627.3_bb0.9125_dl0.5_ad0.1_pd0_qd0_na',
}
HP_SETTINGS = (('Weather', 336),)
TREND_HP_SETTINGS = (('ETTh2', 336), ('Weather', 336))
M_LIST, VARIANTS, L_LIST = (1, 2, 4, 8, 16), ('full_attn', 'linear_mix', 'none'), (96, 192, 336)
RANKS, GATES = (1, 2, 3, 4, 6), (0.01, 0.1, 0.5, 0.9)
FLIP = {'S1': 'S2', 'S2': 'S1'}


def new_job(key):
    return json.loads((eb.SCRATCH / 'search' / 'jobs' / f'{NEW[key]}.json').read_text())


def new_run(key):
    return eb.SCRATCH / 'search' / 'runs' / NEW[key]


# ------------------------------------------------------------------ jobs
def variants():
    """(analysis, key, level, change) for every run; change edits a copy of the job of the new configuration."""
    out = []
    for key in NEW:
        out.append(('ablation', key, 'single_gate', lambda j: j.__setitem__('e1_variant', 'single_gate')))
        out.append(('ablation', key, 'linear_mk', lambda j: j.__setitem__('e1_variant', 'linear_mk')))
    for key in HP_SETTINGS:
        selected = new_job(key)['source']['config']['hyperparams']['phase_num_routers']
        for m in M_LIST:
            if m != selected:
                out.append(('hp', key, f'M_{m}', lambda j, m=m: j['source']['config']['hyperparams'].__setitem__('phase_num_routers', m)))
        for v in VARIANTS:
            out.append(('hp', key, f'interaction_{v}', lambda j, v=v: j.__setitem__('routing_variant', v)))
        for length in L_LIST:
            out.append(('hp', key, f'L_{length}', lambda j, length=length: j['source']['config'].__setitem__('lookback', length)))
    for key in TREND_HP_SETTINGS:
        def flip(j):
            j['head'] = FLIP[j['head']]
        out.append(('trend_hp', key, 'basis_flip', flip))
        for r in RANKS:
            def rank(j, r=r):
                j['head'] = 'S1'
                j['rank_override'] = r
                j['force_rank'] = True
            out.append(('trend_hp', key, f'r_{r}', rank))
        for g in GATES:
            def gate(j, g=g):
                j['variant'] = dict(j['variant'], gate_init=g)
            out.append(('trend_hp', key, f'g0_{g:g}', gate))
        out.append(('trend_hp', key, 'gate_flip', lambda j: j.__setitem__('shared_gate', not j['shared_gate'])))
    return out


def write_jobs():
    paths = []
    for analysis, key, level, change in variants():
        job = copy.deepcopy(new_job(key))
        job.pop('cell', None)
        change(job)
        job['stage'] = f'followup_{analysis}'
        name = f'{key[0]}_H{key[1]}_S{eb.SEED}__{analysis}__{level}'
        job['output'] = str(OUT / 'runs' / analysis / name)
        path = OUT / 'jobs' / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(job, indent=1))
        paths.append(path)
    return paths


# ------------------------------------------------------------------ worker
def build_e1(model, job):
    """The E1 model of the job (the patches of scripts/e1_beat.py); force_rank rebuilds the state even at r = 3."""
    import scripts.phaseformer_e12 as pe
    pe.apply_structure(model, 'e1', job)
    rank = int(job.get('rank_override', 3))
    if rank != 3 or job.get('force_rank'):
        eb.set_state_dimension(model, rank)


def worker(job_path):
    import scripts.b0hp_ablation as abl
    import scripts.b0hp_structure_ablation as sa
    import scripts.e1_paper.ablation as ab
    import scripts.e1_paper.hyperparameter_study as hs
    from scripts.vldbj_m1 import proto
    sa.register_heads()
    job = json.loads(Path(job_path).read_text())
    linear = job.get('e1_variant', '').startswith('linear')
    attach = abl.make_attach(dict(job, ablation_arm='linear_trend') if linear else job)
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        build_e1(model, j)
        ab.apply_variant(model, j)
        if j.get('routing_variant'):
            hs.patch_routing(model, j['routing_variant'])
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
    proto.run_job(job_path)


def launch(path, gpu):
    import os
    import shutil
    import subprocess
    out = Path(json.loads(Path(path).read_text())['output'])
    if (out / 'result.json').is_file():
        return 0
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (out / 'job.log').open('a') as log:
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', str(path)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def run(gpus, slots):
    # Weather first: those runs take 10 to 25 minutes
    queue = sorted(write_jobs(), key=lambda p: (not p.name.startswith('Weather'), p.name))
    lock = threading.Lock()
    failed = []

    def next_task():
        with lock:
            if not queue:
                return None
            path = queue.pop(0)
        need = eb.NEED_MIB.get(path.name.split('_')[0], eb.DEFAULT_NEED_MIB)
        return path, lambda code, path=path: (failed.append(path.name) if code else print('done', path.stem, flush=True)), need

    eb.launch = launch                 # the pool starts this script, not scripts/e1_beat.py, as the worker
    eb.run_pool(gpus, slots, next_task)
    print('failed:', failed)


# ------------------------------------------------------------------ analyses on the trained runs
def load_trained(job, run_dir):
    import torch
    from scripts.vldbj_m1.proto import build_model
    import scripts.b0hp_structure_ablation as sa
    sa.register_heads()
    job = dict(job, smoke=True)
    job.pop('source_hashes', None)
    job.pop('code_commit', None)
    model, args, _, _ = build_model(job)
    build_e1(model, job)
    if run_dir is not None:
        record = json.loads((Path(run_dir) / 'result.json').read_text())
        state = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(state)
        return model, args, record
    return model, args, None


def trend_errors(key):
    import numpy as np
    import torch
    from scripts.component_mse import dump
    from scripts.component_mse import trend_definitions as td
    from src.dataset.data_factory import data_provider
    model, args, record = load_trained(new_job(key), new_run(key))
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, loader = data_provider(args.dataset_args, 'test')
    predictions, truths = [], []
    with torch.inference_mode():
        for batch in loader:
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            predictions.append(model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:].cpu().numpy())
            truths.append(by[:, -model.pred_len:].cpu().numpy())
    prediction, truth = np.concatenate(predictions), np.concatenate(truths)
    mse, mae = dump.test_arrays(prediction, truth)
    recorded = record['best']['test_mse']
    if abs(mse - recorded) > max(1e-5, 1e-4 * recorded):
        raise AssertionError(f'{key}: exported MSE {mse} != recorded {recorded}')
    s = td.score_file(prediction, truth, 24, ('cycle-level',))['cycle-level']
    return dict(dataset=key[0], horizon=key[1], mse_trend=s['trend'], mse_periodic=s['periodic'], mse_total=s['total'],
                test_mse_exported=mse, test_mse_recorded=recorded, test_mae_exported=mae, windows=int(prediction.shape[0]),
                run_dir=str(new_run(key)))


def efficiency(job):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    import scripts.e1_paper.hyperparameter_study as hs
    from scripts.run_lowrank_interaction_stage2 import forecast
    from scripts.vldbj_m1.model import parameter_breakdown
    from src.dataset.data_factory import data_provider
    model, args, _ = load_trained(job, None)
    if job.get('routing_variant'):
        hs.patch_routing(model, job['routing_variant'])
    model.eval()
    dataset, _ = data_provider(args.dataset_args, 'val')
    batch = [torch.as_tensor(v).float().unsqueeze(0) for v in dataset[0]]
    with FlopCounterMode(display=False) as counter:
        forecast(model, batch, 'cpu')
    total = counter.get_total_flops() / 1e6
    trend = {n: sum(o.values()) / 1e6 for n, o in counter.get_flop_counts().items()}.get(
        'PhaseFormer.lowrank_interaction_branch', 0.0)
    parts = parameter_breakdown(model)
    hp = job['source']['config']['hyperparams']
    return dict(channels=int(batch[0].shape[-1]), period=hp['period_len'], layers=hp['layers'], latent_dim=hp['latent_dim'],
                routers=hp['phase_num_routers'], heads=hp['phase_attn_heads'], params_total=parts['total'],
                params_phase_pathway=parts['backbone'],
                params_trend_pathway_and_gates=parts['level_branch'] + parts['fusion'] + parts['modulation'],
                mflops_total=round(total, 4), mflops_trend_pathway=round(trend, 4), mflops_phase_pathway=round(total - trend, 4))


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('wrote', path, len(rows))


def analyze():
    import torch
    torch.set_num_threads(4)
    save(OUT / 'trend_errors.csv', [trend_errors(('ETTh2', 192))])
    save(OUT / 'efficiency_new_configs.csv',
         [dict(dataset=k[0], horizon=k[1], run_dir=str(new_run(k)), **efficiency(new_job(k))) for k in NEW])
    rows = []
    for path in sorted((OUT / 'jobs').glob('*__hp__*.json')):
        setting, _, level = path.stem.split('__')
        factor, value = level.split('_', 1)
        if factor in ('M', 'interaction'):
            e = efficiency(json.loads(path.read_text()))
            rows.append(dict(setting=setting, factor=factor, level=value, params=e['params_total'], mflops=e['mflops_total']))
    for key in HP_SETTINGS:
        e = efficiency(new_job(key))
        rows.append(dict(setting=f'{key[0]}_H{key[1]}_S{eb.SEED}', factor='selected', level='',
                         params=e['params_total'], mflops=e['mflops_total']))
    save(OUT / 'efficiency_hp_variants.csv', rows)


def report():
    rows = []
    for path in sorted((OUT / 'jobs').glob('*.json')):
        setting, analysis, level = path.stem.split('__')
        job = json.loads(path.read_text())
        result = Path(job['output']) / 'result.json'
        row = dict(analysis=analysis, dataset=job['dataset'], horizon=job['horizon'], level=level)
        if result.is_file():
            r = json.loads(result.read_text())
            row.update(test_mse=r['best']['test_mse'], test_mae=r['best']['test_mae'], val_mse=r['best']['val_mse'],
                       epochs=r['epochs'], params=r['parameters']['total'], run_dir=str(result.parent))
        rows.append(row)
    for key in NEW:
        r = json.loads((new_run(key) / 'result.json').read_text())
        rows.append(dict(analysis='selected', dataset=key[0], horizon=key[1], level='selected',
                         test_mse=r['best']['test_mse'], test_mae=r['best']['test_mae'], val_mse=r['best']['val_mse'],
                         epochs=r['epochs'], params=r['parameters']['total'], run_dir=str(new_run(key))))
    fields = ['analysis', 'dataset', 'horizon', 'level', 'test_mse', 'test_mae', 'val_mse', 'epochs', 'params', 'run_dir']
    path = OUT / 'results.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    done = sum('test_mse' in r for r in rows)
    print(f'{done}/{len(rows)} runs with results ->', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--gpus', default='2')
    parser.add_argument('--slots', type=int, default=12)
    args = parser.parse_args()
    if args.job:
        worker(args.job)
    elif args.run:
        run([int(g) for g in args.gpus.split(',')], args.slots)
        report()
    elif args.analyze:
        analyze()
    else:
        report()


if __name__ == '__main__':
    main()
