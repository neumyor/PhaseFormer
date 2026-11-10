#!/usr/bin/env python3
"""PhaseFormer and PhaseFormer-LA runs for the component-MSE experiment.

The two models keep their registered protocols: PhaseFormer is the conference
arm (``pf``), PhaseFormer-LA is the M1 arm under the frozen vldbj_m1 worker with
the best per-setting configuration found so far. Both are trained by the frozen
worker ``scripts/vldbj_m1/proto.py`` and their test predictions are exported
afterwards from the saved best-validation checkpoint.

Run one task with ``--job <job.json>``; build the job list with ``--prepare``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.vldbj_m1 import protocol  # noqa: E402
from scripts.vldbj_m1.proto import provenance  # noqa: E402
from scripts.vldbj_m1.run import commit, e14_sources, write_immutable  # noqa: E402

DATASETS = ('Weather', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
HORIZONS = (96, 192)
SEEDS = (2021, 2022, 2023)
HEAD = protocol.FINAL_HEAD
LOOKBACK = 720

# Best known PhaseFormer-LA configuration per setting: the wide-v1 search for
# Weather and Traffic-96, the aggressive round for Traffic-192. Each was chosen
# on the test split at seed 2021 (test-set selection, user decision) and is kept
# fixed for the three seeds so the comparison is a seed study, not a new search.
LA_VARIANTS = {
    ('Weather', 96): {'gate_init': 0.999, 'extension_lr_mult': 30.0, 'communication_width': 2,
                      'backbone_lr_mult': 1.0},
    ('Weather', 192): {'gate_init': 0.6, 'extension_lr_mult': 3.3333, 'communication_width': 2,
                       'backbone_lr_mult': 1.0},
    ('Traffic', 96): {'gate_init': 0.5, 'extension_lr_mult': 30.0, 'communication_width': 2,
                      'backbone_lr_mult': 1.0},
    ('Traffic', 192): {'gate_init': 0.25, 'extension_lr_mult': 0.3333, 'communication_width': 3,
                       'backbone_lr_mult': 2.0},
    # ETT: the wide-v1 per-setting winners, except ETTm1 where the later period
    # search found better configurations at period 48.
    ('ETTh1', 96): {'gate_init': 0.1, 'extension_lr_mult': 0.1, 'communication_width': 3,
                    'backbone_lr_mult': 1.0},
    ('ETTh1', 192): {'gate_init': 0.1, 'extension_lr_mult': 100.0, 'communication_width': 2,
                     'backbone_lr_mult': 1.0},
    ('ETTh2', 96): {'gate_init': 0.999, 'extension_lr_mult': 3.3333, 'communication_width': 2,
                    'backbone_lr_mult': 1.0},
    ('ETTh2', 192): {'gate_init': 0.1, 'extension_lr_mult': 0.1, 'communication_width': 3,
                     'backbone_lr_mult': 0.3},
    ('ETTm1', 96): {'gate_init': 0.1, 'extension_lr_mult': 3.0, 'communication_width': 2,
                    'backbone_lr_mult': 1.0, 'period': 48},
    ('ETTm1', 192): {'gate_init': 0.1, 'extension_lr_mult': 10.0, 'communication_width': 4,
                     'backbone_lr_mult': 2.0, 'period': 48},
    # ETTm2 is the one place the wide search preferred a different level head and
    # per-channel gates (S3/perC): its best runs were 0.1607 (H96) and 0.2150
    # (H192), against the registered S2/shared variants we ran first.
    ('ETTm2', 96): {'gate_init': 0.99, 'extension_lr_mult': 30.0, 'communication_width': 2,
                    'backbone_lr_mult': 1.0, 'head': 'S3', 'shared': False},
    ('ETTm2', 192): {'gate_init': 0.2, 'extension_lr_mult': 1.0, 'communication_width': 2,
                     'backbone_lr_mult': 1.0, 'head': 'S3', 'shared': False},
}
# Where each chosen configuration was measured, for the record.
LA_SOURCES = {
    ('Weather', 96): 'vldbj_wide_v1 Weather_H96_S2021__m1__g0.999_lr30_w2_bb1 (0.143889)',
    ('Weather', 192): 'vldbj_wide_v1 Weather_H192_S2021__m1__g0.6_lr3.3333_w2_bb1 (0.184392)',
    ('Traffic', 96): 'vldbj_wide_v1 Traffic_H96_S2021__m1__g0.5_lr30_w2_bb1 (0.358660)',
    ('Traffic', 192): 'aggressive_ettm1_traffic_v2 stage1 Traffic_H192_S2021__m1__g0.25_lr0.3333_w3_bb2 (0.375950)',
    ('ETTh1', 96): 'vldbj_wide_v1 ETTh1_H96 wide best (0.357556)',
    ('ETTh1', 192): 'vldbj_wide_v1 ETTh1_H192 wide best (0.389171)',
    ('ETTh2', 96): 'vldbj_wide_v1 ETTh2_H96 wide best (0.273386)',
    ('ETTh2', 192): 'vldbj_wide_v1 ETTh2_H192 wide best (0.334671)',
    ('ETTm1', 96): 'period search ETTm1_H96 p48 g0.1 lr3 w2 (0.285722)',
    ('ETTm1', 192): 'period search ETTm1_H192 p48 g0.1 lr10 w4 bb2 (0.322985)',
    ('ETTm2', 96): 'vldbj_wide_v1 ETTm2_H96 wide best (0.160718)',
    ('ETTm2', 192): 'vldbj_wide_v1 ETTm2_H192 wide best (0.215036)',
}


def job_name(dataset, horizon, seed, arm):
    if arm == 'pf':
        return f'{dataset}_H{horizon}_S{seed}__pf__conference'
    variant = LA_VARIANTS[(dataset, horizon)]
    name = (f"{dataset}_H{horizon}_S{seed}__m1__g{variant['gate_init']:g}"
            f"_lr{variant['extension_lr_mult']:g}_w{variant['communication_width']:g}"
            f"_bb{variant['backbone_lr_mult']:g}")
    if variant.get('period', 24) != 24:
        name += f"__p{variant['period']}"
    head = variant.get('head', HEAD)
    shared = variant.get('shared', protocol.FINAL_SHARED_GATE[dataset])
    if head != HEAD or bool(shared) != bool(protocol.FINAL_SHARED_GATE[dataset]):
        name += f"__{head}__{'shared' if shared else 'perC'}"
    return name


def build_job(base, sources, dataset, horizon, seed, arm):
    if arm == 'pf':
        job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm='pf',
                   protocol='conference', init='scratch', head=HEAD,
                   shared_gate=protocol.FINAL_SHARED_GATE[dataset], variant_id=None,
                   variant=None, extension_lr_mult=1.0, backbone_lr_mult=1.0,
                   warm_source=None, epoch_test=False, source=sources[(dataset, horizon, seed)])
        protocol.conference_source(job['source']['config'])
        return job
    variant = LA_VARIANTS[(dataset, horizon)]
    source = sources[(dataset, horizon, seed)]
    if variant.get('period', 24) != 24:
        import copy as _copy
        source = _copy.deepcopy(source)
        source['config']['period'] = variant['period']
        source['config']['hyperparams']['period_len'] = variant['period']
    job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm='m1', protocol='e14',
               init='scratch', head=variant.get('head', HEAD),
               shared_gate=variant.get('shared', protocol.FINAL_SHARED_GATE[dataset]),
               variant_id=None,
               variant={'gate_init': variant['gate_init'],
                        'extension_lr_mult': variant['extension_lr_mult'],
                        'communication_width': variant['communication_width']},
               extension_lr_mult=float(variant['extension_lr_mult']),
               backbone_lr_mult=float(variant['backbone_lr_mult']),
               warm_source=None, epoch_test=False, source=source)
    return job


def prepare(scratch, stage):
    sources = e14_sources()
    base = {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': False, 'stage': stage}
    jobs = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                for arm in ('pf', 'm1'):
                    job = build_job(base, sources, dataset, horizon, seed, arm)
                    job['output'] = str(scratch / stage / 'runs' / job_name(dataset, horizon, seed, arm))
                    path = scratch / stage / 'jobs' / f"{Path(job['output']).name}.json"
                    write_immutable(path, job)
                    jobs.append(str(path))
    seed_order = {seed: i for i, seed in enumerate(SEEDS)}
    jobs.sort(key=lambda p: (seed_order[int(Path(p).name.split('_S')[1].split('__')[0])],
                             0 if 'Weather' in Path(p).name else 1, p))
    return jobs


def export(job_path, dump_root, which='best'):
    import numpy as np
    import torch

    from scripts.component_mse import dump
    from scripts.vldbj_m1.proto import build_model
    from src.dataset.data_factory import data_provider

    job = json.loads(Path(job_path).read_text())
    run_dir = Path(job['output'])
    record = json.loads((run_dir / 'result.json').read_text())
    checkpoint = record[which + '_checkpoint'] if which == 'last' else record['checkpoint']
    model, args, _, _ = build_model(job)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state)
    device = torch.device('cuda:0')
    model.to(device).eval()
    test_set, test_loader = data_provider(args.dataset_args, 'test')
    branch = getattr(model, 'lowrank_interaction_branch', None)
    if branch is not None and hasattr(branch, 'intervention'):
        branch.intervention = 'native'
    predictions, truths = [], []
    with torch.inference_mode():
        for batch in test_loader:
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
            predictions.append(pred.cpu().numpy())
            truths.append(by[:, -model.pred_len:].cpu().numpy())
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    mse, mae = dump.test_arrays(prediction, truth)
    recorded = record[which]['test_mse']
    if abs(mse - recorded) > max(1e-5, 1e-4 * recorded):
        raise AssertionError(f'exported MSE {mse} differs from recorded {recorded}')
    model_key = 'PhaseFormer' if job['arm'] == 'pf' else 'PhaseFormerLA'
    scaler = test_set.scaler
    dump.save_truth(dump_root, job['dataset'], job['horizon'], truth, scaler)
    dump.save_prediction(dump_root, model_key, job['dataset'], job['horizon'], job['seed'], prediction)
    variant = job.get('variant') or {}
    dump.save_result(dump_root, model_key, job['dataset'], job['horizon'], job['seed'], {
        'model': model_key, 'dataset': job['dataset'], 'horizon': job['horizon'],
        'seed': job['seed'], 'lookback': LOOKBACK, 'arm': job['arm'],
        'head': job['head'], 'shared_gate': job['shared_gate'],
        'variant': variant, 'backbone_lr_mult': job['backbone_lr_mult'],
        'test_mse': mse, 'test_mae': mae, 'test_size': int(prediction.shape[0]),
        'channels': int(prediction.shape[2]), 'checkpoint': checkpoint,
        'run_dir': str(run_dir), 'la_source': LA_SOURCES.get((job['dataset'], job['horizon']))
        if job['arm'] == 'm1' else None,
        'space': 'standardized (train-fitted StandardScaler)',
    })
    print(json.dumps({'model': model_key, 'dataset': job['dataset'], 'horizon': job['horizon'],
                      'seed': job['seed'], 'test_mse': mse}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--stage', default='component_mse')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--job')
    parser.add_argument('--dump-root', type=Path, default=None)
    parser.add_argument('--which', default='best', choices=('best', 'last'))
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()

    if args.prepare:
        jobs = prepare(args.scratch_dir, args.stage)
        print(json.dumps(jobs, indent=1))
        return
    if args.export_only:
        export(args.job, args.dump_root, args.which)
        return
    from scripts.vldbj_m1.proto import run_job
    run_job(args.job)
    export(args.job, args.dump_root, args.which)


if __name__ == '__main__':
    main()
