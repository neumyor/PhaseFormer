"""Complete the fixed interaction matrix, then evaluate every test split once."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NORM_ARMS = ('i0', 'pt_post', 'pt_pre', 'pt_scaled', 'pt_null',
             'tp_post', 'tp_pre', 'tp_scaled', 'tp_selective')
FAMILIES = {
    'normalization': ('PhaseFormer_lowrank_fix', 'lowrank_six_20261007', NORM_ARMS),
    'context': ('PhaseFormer_phase_context_20261009', 'phase_context_pilot_20261009',
                ('b0', 'context_full', 'phase_only')),
    'centered': ('PhaseFormer_centered_mod_20261009', 'centered_mod_pilot_20261009',
                 ('b0', 'tp_mod_centered')),
}
SETTINGS = [(d, h, s) for d in ('ETTh1', 'ETTm2', 'Weather')
            for h in (96, 720) for s in (2021, 2022)]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def module_for(family):
    directory = 'phase_context_interaction' if family == 'context' else 'centered_mod_interaction'
    return importlib.import_module('scripts.'+directory+'.run')


def training_commit(run, result):
    if result.get('code_commit'):
        return result['code_commit']
    manifest = json.loads((Path(run).parents[1]/'manifest.json').read_text())
    return manifest['code_commit']


def inventory(storage, scratch, smoke=False):
    original = storage/'PhaseFormer_token_interaction_20261008/research_runs/lowrank_token_interaction_20261008_scratch'
    manifest_path = original/'manifest.json'
    sources = {s['setting']: s for s in json.loads(manifest_path.read_text())['sources']}
    entries, missing = [], []
    for family, (repo, experiment, arms) in FAMILIES.items():
        for dataset, horizon, seed in SETTINGS:
            setting = f'{dataset}_H{horizon}_S{seed}_val'
            for arm in arms:
                run = storage/repo/'research_runs'/f'{experiment}_scratch/runs/{setting}__{arm}'
                reused_baseline = False
                if family != 'normalization' and arm == 'b0' and not (run/'result.json').exists():
                    run = original/'runs'/f'{setting}__b0'
                    reused_baseline = True
                key = family+'__'+setting+'__'+arm
                existing = (run/'result.json').exists()
                if not existing:
                    if family == 'normalization' or arm == 'b0':
                        raise ValueError(f'Required existing checkpoint missing: {run}')
                    run = scratch/'new_runs'/key
                    missing.append(key)
                entry = {'key': key, 'family': family, 'setting': setting.replace('_val', '_test'),
                         'dataset': dataset, 'horizon': horizon, 'seed': seed, 'arm': arm,
                         'baseline_arm': 'i0' if family == 'normalization' else 'b0',
                         'run': str(run), 'existing': existing, 'reused_baseline': reused_baseline,
                         'source': copy.deepcopy(sources[setting]), 'new_training': not existing}
                if existing:
                    result = json.loads((run/'result.json').read_text())
                    config = json.loads((run/'config.json').read_text())
                    if result['arm'] != arm or result.get('pretrained_checkpoint_loaded') or result.get('smoke'):
                        raise ValueError(f'Unexpected source result: {key}')
                    if config.get('init_checkpoint') or config.get('evaluate_test'):
                        raise ValueError(f'Unexpected source configuration: {key}')
                    checkpoint = Path(config['checkpoint'])
                    if digest(checkpoint) != result['checkpoint_sha256']:
                        raise ValueError(f'Checkpoint provenance mismatch: {key}')
                    entry.update(source_checkpoint_sha256=result['checkpoint_sha256'],
                                 source_config_sha256=digest(run/'config.json'),
                                 source_commit=training_commit(run, result))
                entries.append(entry)
    if len(entries) != 168 or len(missing) != 22:
        raise ValueError(f'Unexpected source matrix: {len(entries)} records, {len(missing)} missing')
    return {'entries': entries, 'new_training_keys': missing, 'source_manifest_sha256': digest(manifest_path),
            'smoke': smoke, 'training_before_test': True,
            'protocol': 'Original fixed hyperparameters; validation-best checkpoint; no test tuning',
            'script_sha256': digest(__file__),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}


def worker(path, mode):
    import torch
    import pytorch_lightning as pl
    item = json.loads(path.read_text())
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if mode == 'train':
        module_for(item['family']).train_job(item['job_path'])
        return
    run = Path(item['run'])
    result = json.loads((run/'result.json').read_text())
    config = json.loads((run/'config.json').read_text())
    checkpoint = Path(config['checkpoint'])
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != result['checkpoint_sha256'] or result.get('pretrained_checkpoint_loaded'):
        raise ValueError('Checkpoint provenance mismatch')
    if config.get('init_checkpoint') or config.get('evaluate_test') or result.get('smoke'):
        raise ValueError('Expected from-scratch validation-selected source')
    output = Path(item['evaluation_output'])
    if output.exists():
        prior = json.loads(output.read_text())
        if prior['checkpoint_sha256'] != checkpoint_hash or prior['config_sha256'] != digest(run/'config.json'):
            raise ValueError('Completed evaluation source changed')
        return
    if item['family'] == 'normalization':
        from scripts.run_lowrank_interaction_stage2 import construct
        model, args = construct(config, workers=0)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(state, strict=True)
        model = model.to('cuda').eval()
    else:
        directory = 'phase_context_interaction' if item['family'] == 'context' else 'centered_mod_interaction'
        # The shared b0 source has a different metadata envelope but identical model state.
        if item['reused_baseline']:
            from scripts.lowrank_token_interaction.model import load_model
        else:
            load_model = importlib.import_module('scripts.'+directory+'.model').load_model
        model, args = load_model(run, 'cuda')
    from scripts.lowrank_token_interaction.model import state_hash
    from scripts.search_phaseformer import evaluate
    from src.dataset.data_factory import data_provider
    pl.seed_everything(item['seed'], workers=True)
    before = state_hash(model.state_dict())
    data, loader = data_provider(args.dataset_args, 'test')
    assert not loader.drop_last and not isinstance(loader.sampler, torch.utils.data.RandomSampler)
    started = time.monotonic()
    scores = evaluate(model, loader, data, 'test', item['horizon'], output.parent, bad_case_limit=0)
    assert before == state_hash(model.state_dict())
    if not all(torch.isfinite(torch.tensor(scores[k])) for k in ('test_mse', 'test_mae')):
        raise ValueError('Nonfinite test metric')
    row = {k: item[k] for k in ('key', 'family', 'setting', 'dataset', 'horizon', 'seed',
                               'arm', 'baseline_arm', 'existing', 'reused_baseline', 'new_training')}
    row.update(**scores, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        config_sha256=digest(run/'config.json'), source_commit=training_commit(run, result),
        dataset_sha256=digest(Path(args.dataset_args.root_path)/args.dataset_args.data_path),
        test_samples=len(data), lookback=config['lookback'], batch_size=args.dataset_args.batch_size,
        epoch_budget=result['epoch_budget'], best_val_loss=result['best_val_loss'],
        state_unchanged=True, elapsed_seconds=time.monotonic()-started,
        environment={'torch': str(torch.__version__), 'lightning': pl.__version__,
                     'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
        precision='medium', evaluator_sha256=digest(__file__))
    save(output, row)
    print(json.dumps({k: row[k] for k in ('key', 'test_mse', 'test_mae')}), flush=True)


def run_tasks(items, mode, gpus, scratch):
    def bucket(index):
        for item in items[index::len(gpus)]:
            path = scratch/'entries'/f"{item['key']}.json"
            save(path, item)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(path), '--mode', mode]
            with (scratch/'logs'/f"{mode}__{item['key']}.log").open('w') as log:
                code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus[index], 'OPENBLAS_NUM_THREADS': '1',
                         'MKL_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
            if code:
                raise RuntimeError(f'{mode} failed: {item["key"]}; exit={code}')
            print(mode, item['key'], 'complete', flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(bucket, i) for i in range(len(gpus))]
        for future in futures:
            future.result()


def main(args):
    if args.worker:
        worker(args.worker, args.mode)
        return
    args.scratch.mkdir(parents=True, exist_ok=True)
    for directory in ('entries', 'logs', 'jobs', 'evaluations'):
        (args.scratch/directory).mkdir(exist_ok=True)
    manifest_path = args.scratch/'manifest.json'
    manifest = inventory(args.storage, args.scratch.resolve(), args.smoke)
    if args.inventory:
        print(json.dumps({'records': len(manifest['entries']), 'new_training': len(manifest['new_training_keys']),
                          'missing': manifest['new_training_keys']}, indent=2))
        return
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise ValueError('Refuse changed matrix or source provenance')
    else:
        save(manifest_path, manifest)
    new = [entry for entry in manifest['entries'] if entry['new_training']]
    if args.smoke:
        new = [next(e for e in new if e['family'] == family and e['horizon'] == 96) for family in ('context', 'centered')]
    for entry in new:
        module = module_for(entry['family'])
        job = {'epochs': 1 if args.smoke else 30, 'smoke': args.smoke,
               'code_commit': module.commit(), 'source_hashes': module.provenance(),
               'source': entry['source'], 'arm': entry['arm'], 'output': entry['run']}
        path = args.scratch/'jobs'/f"{entry['key']}.json"
        if path.exists() and json.loads(path.read_text()) != job:
            raise ValueError('Training job changed')
        save(path, job)
        entry['job_path'] = str(path.resolve())
    gpus = args.gpus.split(',')
    run_tasks(new, 'train', gpus, args.scratch)
    if args.smoke:
        save(args.scratch/'smoke_complete.json', {'jobs': len(new), 'test_inference_runs': 0})
        print('SMOKE COMPLETE; no test inference', flush=True)
        return
    save(args.scratch/'training_complete_before_test.json', {'jobs': len(new), 'completed_at': time.time()})
    for entry in manifest['entries']:
        entry['evaluation_output'] = str((args.scratch/'evaluations'/f"{entry['key']}.json").resolve())
    run_tasks(manifest['entries'], 'evaluate', gpus, args.scratch)
    save(args.scratch/'complete.json', {'training_jobs': len(new), 'test_inference_runs': len(manifest['entries'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage', type=Path)
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--mode', choices=('train', 'evaluate'))
    main(parser.parse_args())
