#!/usr/bin/env python3
"""Full-test inference only; never train or alter registered configurations.

Run separately in each original source checkout. Select only ETTh1/ETTm2/Weather,
H96/H720, seeds 2021/2022, including each family's no-interaction reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ATTENTION = (
    'phaseformer_l_i0_late_fusion', 'phaseformer_l_i1_phase_to_trend',
    'phaseformer_l_i2_trend_to_phase', 'phaseformer_l_i3_parallel_bidir',
    'phaseformer_l_i6_alternating', 'phaseformer_l_i7_joint',
)
LIGHTWEIGHT = ('b0', 'pt_add', 'pt_correct', 'tp_add', 'tp_mod')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--family', choices=('attention', 'lightweight'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    cli = parser.parse_args()
    if not 0 <= cli.shard < cli.shards:
        parser.error('Invalid shard')
    repo = cli.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import torch
    import pytorch_lightning as pl
    from scripts.run_lowrank_interaction_stage2 import construct
    from scripts.search_phaseformer import evaluate
    from src.dataset.data_factory import data_provider
    if cli.family == 'lightweight':
        from scripts.lowrank_token_interaction.model import load_model
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    arms = ATTENTION if cli.family == 'attention' else LIGHTWEIGHT
    jobs = {}
    for path in sorted(cli.runs.glob('*/config.json')):
        config = json.loads(path.read_text())
        key = (config['dataset'], config['horizon'], config['seed'])
        if key[0] not in ('ETTh1', 'ETTm2', 'Weather') or key[1] not in (96, 720) or key[2] not in (2021, 2022):
            continue
        arm = config['mechanism'] if cli.family == 'attention' else config['token_communication']['arm']
        if arm not in arms:
            continue
        identity = (*key, arm)
        if identity in jobs:
            raise ValueError(f'Ambiguous checkpoint: {identity}')
        if config.get('init_checkpoint') or config['input_hypothesis'] != 'none':
            raise ValueError('Unexpected initialization or input intervention')
        hp = config['hyperparams']
        if not hp['use_lowrank_interaction'] or hp.get('use_trend_token', False):
            raise ValueError('Expected original low-rank linear branch')
        checkpoint = config.get('checkpoint', '')
        if cli.family == 'attention':
            with (path.parent / 'metrics.csv').open() as stream:
                metrics = list(csv.DictReader(stream))[-1]
            checkpoint = metrics['checkpoint']
        checkpoint = Path(checkpoint)
        checkpoint = checkpoint if checkpoint.is_absolute() else repo / checkpoint
        jobs[identity] = (path, config, checkpoint)
    expected = {(d, h, s, a) for d in ('ETTh1', 'ETTm2', 'Weather')
                for h in (96, 720) for s in (2021, 2022) for a in arms}
    if set(jobs) != expected:
        raise ValueError(f'Incomplete matrix: {expected - set(jobs)}')
    cli.output.mkdir(parents=True, exist_ok=True)
    result_path = cli.output / f'{cli.family}_{cli.shard}.jsonl'
    completed = {}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            row = json.loads(line)
            completed[row['setting'] + '__' + row['arm']] = row
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for index, (identity, (path, config, checkpoint)) in enumerate(sorted(jobs.items())):
        if index % cli.shards != cli.shard:
            continue
        dataset_name, horizon, seed, arm = identity
        setting = f'{dataset_name}_H{horizon}_S{seed}_test'
        checkpoint_hash = sha256(checkpoint)
        config_hash = sha256(path)
        previous = completed.get(setting + '__' + arm)
        if previous:
            if previous['checkpoint_sha256'] != checkpoint_hash or previous['config_sha256'] != config_hash:
                raise ValueError('Completed result provenance changed')
            continue
        pl.seed_everything(seed, workers=True)
        started = time.monotonic()
        if cli.family == 'lightweight':
            model, args = load_model(path.parent, 'cuda')
        else:
            model, args = construct(config, workers=0)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
            model.load_state_dict(state, strict=True)
            model = model.to('cuda').eval()
        data, loader = data_provider(args.dataset_args, 'test')
        if loader.drop_last or len(data) == 0:
            raise ValueError('Expected complete nonempty test split')
        scores = evaluate(model, loader, data, 'test', horizon, cli.output, bad_case_limit=0)
        if not all(math.isfinite(scores[k]) for k in ('test_mse', 'test_mae')):
            raise ValueError('Nonfinite test metrics')
        data_file = Path(args.dataset_args.root_path) / args.dataset_args.data_path
        row = dict(setting=setting, dataset=dataset_name, horizon=horizon, seed=seed,
                   arm=arm, family=cli.family, **scores, test_samples=len(data),
                   checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                   config=str(path), config_sha256=config_hash, source_commit=source_commit,
                   data_file=str(data_file), data_sha256=sha256(data_file),
                   lookback=config['lookback'], rank=config['hyperparams']['weak_period_residual_rank'],
                   batch_size=args.dataset_args.batch_size, max_epochs=config['max_epochs'],
                   fusion_mode=config['hyperparams'].get('lowrank_fusion_mode', 'scalar'),
                   torch_version=torch.__version__, lightning_version=pl.__version__,
                   gpu=torch.cuda.get_device_name(), elapsed_seconds=time.monotonic() - started)
        with result_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({k: row[k] for k in ('setting', 'arm', 'test_mse', 'test_mae', 'elapsed_seconds')}), flush=True)
        del model, data, loader
        torch.cuda.empty_cache()
    print(f'COMPLETE {cli.family} shard {cli.shard}', flush=True)


if __name__ == '__main__':
    main()
