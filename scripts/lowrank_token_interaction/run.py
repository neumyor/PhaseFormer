#!/usr/bin/env python3
"""Private training controller for the five static-gate token-interaction arms."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lowrank_token_interaction.model import ARMS, random_models, validate_protocol
from scripts.run_lowrank_interaction_stage2 import digest, forecast, save_json

DEFAULT_SETTINGS = ','.join(f'{d}:{h}:{s}' for d in ('ETTh1', 'ETTm2', 'Weather')
                            for h in (96, 720) for s in (2021, 2022))
CHILDREN = set()
CHILDREN_LOCK = threading.Lock()
STOP = threading.Event()


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def provenance():
    paths = sorted((ROOT / 'scripts/lowrank_token_interaction').glob('*.py'))
    paths += [ROOT / p for p in ('src/models/PhaseFormer.py', 'src/models/lowrank_interaction.py',
        'src/models/lowrank_fusion.py', 'src/models/phase_adapters.py', 'src/models/phaseformer_presets.py',
        'src/dataset/data_factory.py', 'src/dataset/data_loader.py', 'src/training/runner.py',
        'config/base_config.py', 'src/dataset/data_info.py', 'src/utils/metrics.py',
        'scripts/run_lowrank_interaction_stage2.py', 'scripts/analyze_lowrank_six.py',
        'scripts/analyze_lowrank_stage2.py', 'scripts/search_phaseformer.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def sources_from_manifest(path, settings):
    original = json.loads(Path(path).read_text())
    if original.get('study') != 'level' or original.get('smoke'):
        raise ValueError('Use the completed from-scratch level study as config source')
    requested = {tuple(s.split(':')) for s in settings.split(',')}
    sources = []
    found = set()
    for item in original['sources']:
        source = copy.deepcopy(item)
        cfg = source['config']
        key = tuple(str(cfg[k]) for k in ('dataset', 'horizon', 'seed'))
        if key not in requested:
            continue
        if key in found:
            raise ValueError(f'Duplicate source: {key}')
        found.add(key)
        if source['setting'] != f'{key[0]}_H{key[1]}_S{key[2]}_val':
            raise ValueError('Mislabeled source setting')
        validate_protocol(cfg)
        hp = cfg['hyperparams']
        assert not cfg.get('init_checkpoint') and not cfg['evaluate_test']
        assert cfg['lookback'] == 720 and cfg['period'] == 24 and cfg['percent'] == 100
        assert cfg['loss'] == 'huber' and cfg.get('input_hypothesis', 'none') == 'none'
        assert hp['weak_period_residual_rank'] == cfg['horizon'] // 8
        assert hp['weak_period_residual_pool_factor'] == 1
        assert hp['weak_period_residual_smooth_ratio'] == 0
        assert hp['trend_interaction_mode'] == 'none' and not hp.get('use_trend_token', False)
        cfg.update(checkpoint='', init_checkpoint='')
        sources.append(source)
    if found != requested:
        raise ValueError('Requested settings are missing or ambiguous')
    return sorted(sources, key=lambda s: s['setting'])


def write_immutable(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Refuse changed protocol: {path}')
    else:
        save_json(path, value)


def train_job(path):
    job = json.loads(Path(path).read_text())
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'result.json').exists():
            result = json.loads((out / 'result.json').read_text())
            if result['code_commit'] != job['code_commit'] or digest(result['checkpoint']) != result['checkpoint_sha256']:
                raise ValueError('Existing success has changed provenance')
            return
        if (out / 'checkpoints').exists():
            raise RuntimeError(f'Incomplete training requires a new attempt directory: {out}')
        _train_job(path)


def _train_job(path):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import pytorch_lightning as pl
    from src.dataset.data_factory import data_provider
    from src.training.runner import build_logger, build_trainer, restore_best_checkpoint
    from scripts.search_phaseformer import evaluate

    job = json.loads(Path(path).read_text())
    if job['code_commit'] != commit() or job['source_hashes'] != provenance():
        raise ValueError('Worker source differs from frozen protocol')
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    original_load = torch.load
    def forbid_load(*args, **kwargs):
        raise AssertionError('No pretrained weight loading during initialization')
    torch.load = forbid_load
    try:
        model, reference, args, cfg, common_hash = random_models(job['source']['config'], job['arm'], job['epochs'])
    finally:
        torch.load = original_load
    _, train_loader = data_provider(args.dataset_args, 'train')
    val_set, val_loader = data_provider(args.dataset_args, 'val')
    batch = [v[:2] for v in next(iter(val_loader))]
    device = torch.device('cuda:0')
    model.to(device).eval()
    reference.to(device).eval()
    with torch.inference_mode():
        prediction, _ = forecast(model, batch, device)
        expected, _ = forecast(reference, batch, device)
        torch.testing.assert_close(prediction, expected, atol=0, rtol=0)
        altered = list(batch)
        altered[1] = torch.randn_like(batch[1])
        altered[2] = torch.randn_like(batch[2])
        altered[3] = torch.randn_like(batch[3])
        torch.testing.assert_close(prediction, forecast(model, altered, device)[0], atol=0, rtol=0)
    assert all(p.requires_grad for p in model.parameters())
    save_json(out / 'initialization.json', {
        'common_random_state_sha256': common_hash, 'pretrained_checkpoint_loaded': False,
        'initial_max_abs_vs_b0': 0.0, 'all_parameters_trainable': True,
        'label_and_time_marks_unreachable': True})
    del reference
    gradient_records = []
    class TrainingAudit(pl.Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            assert all(m.training for m in module.modules())
            assert all(p.requires_grad for p in module.parameters())

        def on_after_backward(self, trainer, module):
            if trainer.global_step < 8:
                names = ('lowrank_interaction_branch.', 'lowrank_dynamic_fusion.',
                         'lowrank_interaction_gate', 'embedding.', 'predictor.')
                record = {'step': int(trainer.global_step), 'gradient_l1': {}}
                for name, parameter in module.named_parameters():
                    if name.startswith(names):
                        if parameter.grad is not None:
                            assert torch.isfinite(parameter.grad).all(), name
                            record['gradient_l1'][name] = float(parameter.grad.abs().sum())
                gradient_records.append(record)

    trainer, checkpoint = build_trainer(max_epochs=job['epochs'],
        logger=build_logger(str(out / 'lightning'), version='train'), patience=8,
        checkpoint_dir=str(out / 'checkpoints'), progress=False)
    trainer.callbacks.append(TrainingAudit())
    if job['smoke']:
        trainer.limit_train_batches = 4
        trainer.limit_val_batches = 2
    pl.seed_everything(cfg['seed'], workers=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.monotonic() - started
    save_json(out / 'gradient_startup.json', gradient_records)
    prefix = 'lowrank_interaction_branch.communication.'
    for name, _ in model.named_parameters():
        if name.startswith(prefix):
            if not any(r['gradient_l1'].get(name, 0) > 0 for r in gradient_records):
                raise AssertionError(f'Communication gradient failed to start: {name}')
    restore_best_checkpoint(model, checkpoint)
    model.to(device).eval()
    metrics = evaluate(model, val_loader, val_set, 'val', cfg['horizon'], out)
    cfg['checkpoint'] = checkpoint.best_model_path
    save_json(out / 'config.json', cfg)
    model_count = sum(p.numel() for p in model.parameters())
    communication_count = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))
    save_json(out / 'result.json', {
        'setting': job['source']['setting'], 'arm': job['arm'], **metrics,
        'code_commit': commit(), 'elapsed_sec': elapsed, 'epochs': trainer.current_epoch,
        'epoch_budget': job['epochs'], 'effective_patience': 8,
        'parameter_count': model_count, 'communication_parameters': communication_count,
        'trainable_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'communication_width': min(2, model.lowrank_interaction_branch.rank, model.latent_dim),
        'common_random_state_sha256': common_hash, 'pretrained_checkpoint_loaded': False,
        'initial_max_abs_vs_b0': 0.0, 'checkpoint': checkpoint.best_model_path,
        'checkpoint_sha256': digest(checkpoint.best_model_path),
        'best_val_loss': float(checkpoint.best_model_score),
        'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'smoke': job['smoke'],
        'environment': {'torch': torch.__version__, 'lightning': pl.__version__,
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}})


def launch(path, gpu):
    job = json.loads(Path(path).read_text())
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        train_job(path)
        return
    if STOP.is_set():
        return
    command = [sys.executable, str(Path(__file__).resolve()), '--job', str(path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    save_json(out / 'status.json', {'status': 'running', 'gpu': gpu, 'command': command})
    with (out / 'process.log').open('a') as log:
        code = run_child(command, env, log)
    save_json(out / 'status.json', {'status': 'stopped' if STOP.is_set() else 'completed' if code == 0 else 'failed',
                                  'gpu': gpu, 'command': command, 'returncode': code})
    if code and not STOP.is_set():
        stop_children()
        raise RuntimeError(f'Worker failed: {path}')


def run_child(command, env=None, log=None):
    with CHILDREN_LOCK:
        if STOP.is_set():
            return -signal.SIGTERM
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT if log else None, start_new_session=True)
        CHILDREN.add(process)
    try:
        while True:
            try:
                return process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                if STOP.is_set():
                    terminate_child(process)
                    return process.returncode
    finally:
        with CHILDREN_LOCK:
            CHILDREN.discard(process)


def request_stop(*args):
    # A signal handler must not acquire a lock held by its interrupted thread.
    STOP.set()


def terminate_child(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def stop_children(*args):
    STOP.set()
    with CHILDREN_LOCK:
        processes = list(CHILDREN)
    for process in processes:
        terminate_child(process)


def prepare(args):
    import yaml
    import torch
    import pytorch_lightning as pl
    scratch, audit = args.scratch_dir.resolve(), args.audit_dir.resolve()
    if scratch == audit or scratch in audit.parents or audit in scratch.parents:
        raise ValueError('Scratch and canonical audit must be disjoint')
    sources = sources_from_manifest(args.source_manifest.resolve(), args.settings)
    manifest = {'experiment_id': audit.name, 'sources': sources, 'arms': list(ARMS),
        'epochs': args.epochs, 'smoke': args.smoke, 'code_commit': commit(),
        'source_hashes': provenance(), 'source_manifest_sha256': digest(args.source_manifest),
        'baseline': 'b0', 'communication_width': 2, 'evaluation_split': 'val'}
    write_immutable(scratch / 'manifest.json', manifest)
    jobs = []
    for source in sources:
        for arm in ARMS:
            name = f"{source['setting']}__{arm}"
            path = scratch / 'jobs' / f'{name}.json'
            job = {k: manifest[k] for k in ('epochs', 'smoke', 'code_commit', 'source_hashes')}
            job.update(source=source, arm=arm, output=str(scratch / 'runs' / name))
            write_immutable(path, job)
            jobs.append(path)
    settings = [{'setting': s['setting'], 'dataset': s['config']['dataset'],
                 'horizon': s['config']['horizon'], 'seed': s['config']['seed'],
                 'lookback': 720, 'split': 'val'} for s in sources]
    run = {'experiment_id': audit.name, 'code': {'commit': manifest['code_commit'],
            'source_hashes': manifest['source_hashes'], 'isolated_entry': str(Path(__file__).relative_to(ROOT))},
        'mechanism': {'description': 'Non-attention latent interaction with static level/shape output fusion',
                      'arms': list(ARMS), 'communication_width': 2, 'zero_initialized_exit': True},
        'experiment': {'baseline': 'b0', 'candidate': list(ARMS[1:]), 'settings': settings,
            'training': {'epochs': args.epochs, 'patience': 8, 'initialization': 'from scratch',
                         'all_parameters_trainable': True, 'loss': 'Huber', 'checkpoint': 'lowest validation Huber'},
            'metrics': ['MSE', 'MAE'], 'smoke': args.smoke},
        'execution': {'environment': {'python': sys.version, 'torch': torch.__version__,
            'lightning': pl.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'},
            'settings': [{'setting': s['setting'], 'source_config': s['config'],
            'commands': [[sys.executable, str(Path(__file__).resolve()), '--job', str(p)]
                         for p in jobs if p.name.startswith(s['setting'] + '__')]} for s in sources]},
        'selection': {'source': 'validation', 'test_set_selection': False,
            'selected_configs': [{'setting': s['setting'], 'config_id': list(ARMS),
             'search_notes': 'All predeclared arms retained for analysis; no test selection'} for s in sources]},
        'analysis': {'ranking_metric': 'mae', 'top_k': 3,
                     'dedup_rule': 'within category/channel, start separation >=H; at most9 per candidate/setting'},
        'validation': {'status': 'incomplete'}}
    audit.mkdir(parents=True, exist_ok=True)
    if (audit / 'run.yaml').exists():
        existing = yaml.safe_load((audit / 'run.yaml').read_text())
        for key in ('experiment_id', 'code', 'mechanism', 'experiment', 'execution', 'selection'):
            if existing[key] != run[key]:
                raise ValueError(f'Audit protocol conflict: {key}')
    else:
        if any(audit.iterdir()):
            raise ValueError('Refuse an audit directory without registered metadata')
        (audit / 'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    return manifest, jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--settings', default=DEFAULT_SETTINGS)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        train_job(args.job)
        return
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if any(v is None for v in (args.source_manifest, args.scratch_dir, args.audit_dir)):
        parser.error('Source manifest, scratch directory and audit directory are required')
    if args.epochs < 1 or args.smoke and args.epochs != 1:
        parser.error('Smoke requires one epoch; formal epoch count must be positive')
    gpus = args.gpus.split(',')
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        parser.error('GPU ids must be unique integers')
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    with (args.scratch_dir / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, jobs = prepare(args)
        print(json.dumps({'jobs': len(jobs), 'arms': list(ARMS), 'code': manifest['code_commit']}), flush=True)
        if args.prepare_only:
            return
        tasks = queue.Queue()
        for path in jobs:
            tasks.put(path)
        def bucket(gpu):
            while not STOP.is_set():
                try:
                    path = tasks.get_nowait()
                except queue.Empty:
                    return
                launch(path, gpu)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(bucket, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        if STOP.is_set():
            raise RuntimeError('Controller stopped before audit')
        code = run_child([sys.executable, str(ROOT / 'scripts/lowrank_token_interaction/analyze.py'),
            '--scratch-dir', str(args.scratch_dir.resolve()), '--audit-dir', str(args.audit_dir.resolve()),
            '--gpus', args.gpus])
        if code:
            raise RuntimeError('Analysis failed')


if __name__ == '__main__':
    main()
