#!/usr/bin/env python3
"""Controller for the VLDBJ compressed-M1 pilot, 28-setting training and test.

Stages (each refuses to run out of order):
  pilot  -- validation-only head/gate pilot, seed 2021, one setting per dataset
  select -- apply the pre-registered selection rule to the pilot
  main   -- b0 and m1 with the selected spec, 28 settings x 3 seeds
  test   -- one test read per finished main checkpoint, plus modulation-off
            and cycle-mean invariance checks
  tune   -- test-set tuning round (user decision 2026-10-10): six extension
            variants over five datasets, one seed; each job trains and reads
            its test split once, recording the variant with its test number
  tune-select -- pick the per-setting winner against the main-stage
            configuration on seed-paired test reads (test-set selection)
  confirm -- rerun the chosen variants on the seeds the tune round skipped
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_lowrank_interaction_stage2 import digest, save_json  # noqa: E402
from scripts.vldbj_m1.model import ARMS, HEADS, paired_models, parameter_breakdown  # noqa: E402

DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
# One registered pilot setting per dataset, covering short, middle and long horizons.
PILOT = (('ETTh1', 96), ('ETTh2', 720), ('ETTm1', 336), ('ETTm2', 96),
         ('Weather', 192), ('Electricity', 96), ('Traffic', 96))
PILOT_SEED = 2021
GATES = (False, True)
# Fewer parameters first; used to break near-ties.
HEAD_PREFERENCE = ('S2', 'S1', 'S3')
HEAD_TIE = 0.005
GATE_TIE = 0.001
# Test-set tuning round (user decision 2026-10-10): one seed over the high-coupling
# and lagging datasets, six extension variants; selection happens on test reads.
TUNE_DATASETS = ('Traffic', 'Electricity', 'Weather', 'ETTm1', 'ETTm2')
TUNE_SEED = 2021
TUNE_GRID = (
    {'communication_width': 2, 'extension_lr_mult': 10.0, 'gate_init': 0.2},
    {'communication_width': 2, 'extension_lr_mult': 1.0, 'gate_init': 0.5},
    {'communication_width': 4, 'extension_lr_mult': 1.0, 'gate_init': 0.2},
    {'communication_width': 4, 'extension_lr_mult': 10.0, 'gate_init': 0.2},
    {'communication_width': 2, 'extension_lr_mult': 10.0, 'gate_init': 0.5},
    {'communication_width': 4, 'extension_lr_mult': 10.0, 'gate_init': 0.5},
)
# The confirm stage reruns the per-setting winners on the seeds the tune round
# did not cover; seed TUNE_SEED of every variant is already measured there.
CONFIRM_SEEDS = tuple(s for s in SEEDS if s != TUNE_SEED)
E14 = 'research_runs/phaseformer_L_e14_main_v1/results.csv'
# E14 runs live in the original server checkout; experiments run in a frozen copy.
E14_ROOT = Path(os.environ.get('VLDBJ_E14_ROOT', ROOT))
STOP = False


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def provenance():
    paths = sorted((ROOT / 'scripts/vldbj_m1').glob('*.py'))
    paths += [ROOT / p for p in ('tests/test_vldbj_m1.py', 'scripts/centered_mod_interaction/model.py',
              'scripts/lowrank_token_interaction/model.py', 'scripts/run_lowrank_interaction_stage2.py')]
    for directory in ('src/models', 'src/dataset', 'src/utils', 'src/training', 'config'):
        paths += sorted((ROOT / directory).rglob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def e14_sources():
    """E14 phase-only configs: the ICLR backbone with the E14 training protocol."""
    sources = {}
    with (E14_ROOT / E14).open() as handle:
        for row in csv.DictReader(handle):
            if row['arm'] != 'phase_only':
                continue
            run_dir = Path(row['run_dir'])
            run_dir = run_dir if run_dir.is_absolute() else E14_ROOT / run_dir
            cfg = json.loads((run_dir / 'config.json').read_text())
            key = (cfg['dataset'], int(cfg['horizon']), int(cfg['seed']))
            if key in sources:
                raise ValueError(f'Ambiguous E14 source {key}')
            if cfg['mechanism'] != 'no_residual' or cfg.get('init_checkpoint'):
                raise ValueError(f'Unexpected E14 source {key}')
            cfg.pop('checkpoint', None)
            sources[key] = {'config': cfg, 'source_run': str(run_dir.relative_to(E14_ROOT)),
                            'source_config_sha256': digest(run_dir / 'config.json')}
    return sources


def name(job):
    base = (f"{job['dataset']}_H{job['horizon']}_S{job['seed']}__{job['arm']}"
            f"__{job['head']}__{'shared' if job['shared_gate'] else 'perC'}")
    if job.get('variant_id') is not None:
        return f"{base}__t{job['variant_id']}"
    return base


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
            return
        if (out / 'checkpoints').exists():
            raise RuntimeError(f'Incomplete run requires a new directory: {out}')
        _train_job(job, out)


def _train_job(job, out):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import pytorch_lightning as pl
    from src.dataset.data_factory import data_provider
    from src.training.runner import build_logger, build_trainer, restore_best_checkpoint
    from scripts.search_phaseformer import evaluate

    if job['code_commit'] != commit() or job['source_hashes'] != provenance():
        raise ValueError('Worker source differs from frozen protocol')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('medium')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    original_load = torch.load
    def forbid_load(*args, **kwargs):
        raise AssertionError('No pretrained weight loading during initialization')
    torch.load = forbid_load
    try:
        model, reference, args, cfg, common_hash = paired_models(
            job['source']['config'], job['arm'], job['head'], job['shared_gate'])
    finally:
        torch.load = original_load
    _, train_loader = data_provider(args.dataset_args, 'train')
    val_set, val_loader = data_provider(args.dataset_args, 'val')
    device = torch.device('cuda:0')
    x = next(iter(val_loader))
    model.to(device).eval()
    reference.to(device).eval()
    with torch.inference_mode():
        from scripts.run_lowrank_interaction_stage2 import forecast
        torch.testing.assert_close(forecast(model, x, device)[0], forecast(reference, x, device)[0],
                                   atol=0, rtol=0)
    del reference
    epoch_times = []

    class EpochTimer(pl.Callback):
        def on_train_epoch_start(self, trainer, module):
            torch.cuda.synchronize()
            self.start = time.monotonic()

        def on_train_epoch_end(self, trainer, module):
            torch.cuda.synchronize()
            epoch_times.append(time.monotonic() - self.start)

    hp = cfg['hyperparams']
    trainer, checkpoint = build_trainer(max_epochs=1 if job['smoke'] else cfg['max_epochs'],
        logger=build_logger(str(out / 'lightning'), version='train'),
        patience=int(hp.get('patience', 8)), checkpoint_dir=str(out / 'checkpoints'), progress=False)
    trainer.callbacks.append(EpochTimer())
    if job['smoke']:
        trainer.limit_train_batches = 4
        trainer.limit_val_batches = 2
    pl.seed_everything(cfg['seed'], workers=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.monotonic() - started
    restore_best_checkpoint(model, checkpoint)
    model.to(device).eval()
    metrics = evaluate(model, val_loader, val_set, 'val', cfg['horizon'], out)
    cfg['checkpoint'] = checkpoint.best_model_path
    save_json(out / 'config.json', cfg)
    save_json(out / 'result.json', {
        'name': name(job), 'stage': job['stage'], **metrics,
        'parameters': parameter_breakdown(model), 'common_random_state_sha256': common_hash,
        'code_commit': commit(), 'elapsed_sec': elapsed, 'epochs': trainer.current_epoch,
        'epoch_train_median_seconds': sorted(epoch_times)[len(epoch_times) // 2] if epoch_times else None,
        'checkpoint': checkpoint.best_model_path, 'checkpoint_sha256': digest(checkpoint.best_model_path),
        'best_val_loss': float(checkpoint.best_model_score), 'smoke': job['smoke'],
        'peak_memory_bytes': torch.cuda.max_memory_allocated(),
        'dataset_sha256': digest(Path(args.dataset_args.root_path) / args.dataset_args.data_path),
        'environment': {'torch': torch.__version__, 'lightning': pl.__version__,
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}})


def test_job(path):
    """Single registered test read for a finished main checkpoint."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from src.dataset.data_factory import data_provider
    from scripts.vldbj_m1.model import construct

    job = json.loads(Path(path).read_text())
    out = Path(job['output'])
    if (out / 'test_result.json').exists():
        return
    if job['code_commit'] != commit():
        raise ValueError('Test reader source differs from frozen protocol')
    result = json.loads((out / 'result.json').read_text())
    cfg = json.loads((out / 'config.json').read_text())
    if digest(result['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after training')
    torch.set_float32_matmul_precision('medium')
    model, args = construct({**cfg, 'checkpoint': ''})
    state = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, loader = data_provider(args.dataset_args, 'test')
    branch = model.lowrank_interaction_branch
    modes = ('native', 'off') if job['arm'] == 'm1' else ('native',)
    sums = {m: [0.0, 0.0, 0] for m in modes}
    per_window = {m: [] for m in modes}
    with torch.inference_mode():
        for batch in loader:
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            true = by[:, -model.pred_len:]
            for mode in modes:
                if job['arm'] == 'm1':
                    branch.intervention = mode
                pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
                err = pred - true
                sums[mode][0] += err.square().sum().item()
                sums[mode][1] += err.abs().sum().item()
                sums[mode][2] += err.numel()
                per_window[mode].append(err.square().mean(1).cpu().numpy().astype('float32'))
        if job['arm'] == 'm1':
            branch.intervention = 'native'
    record = {'name': result['name'], 'split': 'test', 'precision': 'medium',
              'test_size': int(sum(a.shape[0] for a in per_window['native']))}
    for mode in modes:
        se, ae, n = sums[mode]
        suffix = '' if mode == 'native' else '_modulation_off'
        record[f'test_mse{suffix}'] = se / n
        record[f'test_mae{suffix}'] = ae / n
    np.savez_compressed(out / 'test_window_mse.npz',
                        **{m: np.concatenate(per_window[m]) for m in modes})
    if job['arm'] == 'm1':
        record['cycle_mean_check'] = cycle_mean_check(model, loader, device)
    record['code_commit'] = commit()
    save_json(out / 'test_result.json', record)


def cycle_mean_check(model, loader, device):
    """Highest-precision check that modulation leaves every future cycle mean."""
    import torch
    torch.set_float32_matmul_precision('highest')
    model = model.double()
    captured = {}
    hook = model.predictor.register_forward_hook(lambda m, i, o: captured.__setitem__('y', o.detach()))
    branch = model.lowrank_interaction_branch
    bx, by, bxm, bym = [v.double().to(device) for v in next(iter(loader))]
    bx, by, bxm, bym = bx[:8], by[:8], bxm[:8], bym[:8]
    outputs = {}
    with torch.inference_mode():
        for mode in ('native', 'off'):
            branch.intervention = mode
            model(bx, bxm, model._build_decoder_input(by), bym)
            outputs[mode] = captured['y']
    branch.intervention = 'native'
    hook.remove()
    native, off = outputs['native'], outputs['off']
    cycle_gap = (native.mean(2) - off.mean(2)).abs().max().item()
    return {'max_cycle_mean_difference': cycle_gap,
            'max_phase_output_change': (native - off).abs().max().item(),
            'tolerance': 1e-9 * max(1.0, native.abs().max().item()),
            'passed': cycle_gap <= 1e-9 * max(1.0, native.abs().max().item())}


EXTENSION_PREFIXES = ('lowrank_interaction_branch.', 'lowrank_interaction_gate', 'lowrank_dynamic_fusion.')


def _attach_extension_lr_multiplier(model, mult):
    """Split Adam into backbone and extension groups with a scaled extension lr."""
    import types
    import torch
    base_lr = float(model.args.training_args.learning_rate)

    def configure_optimizers(self):
        backbone, extension = [], []
        for pname, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (extension if pname.startswith(EXTENSION_PREFIXES) else backbone).append(p)
        optimizer = torch.optim.Adam([
            {'params': backbone, 'lr': base_lr},
            {'params': extension, 'lr': base_lr * mult}])
        if self.args.training_args.lr_schedule_config.type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.training_args.lr_schedule_config.tmax, eta_min=1e-8)
            return [optimizer], [scheduler]
        return optimizer

    model.configure_optimizers = types.MethodType(configure_optimizers, model)


def tune_job(path):
    """Train one m1 tuning variant and read its test split once.

    Tuning selects on the test split by user decision (2026-10-10, time-boxed);
    every variant, its configuration and its test number are recorded here, so
    the selection is auditable and must be labelled test-based in any report.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import pytorch_lightning as pl
    from src.dataset.data_factory import data_provider
    from src.training.runner import build_logger, build_trainer, restore_best_checkpoint
    from scripts.search_phaseformer import evaluate
    from scripts.vldbj_m1.model import parameter_breakdown, paired_models

    job = json.loads(Path(path).read_text())
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'tune_result.json').exists():
            return
        if (out / 'checkpoints').exists():
            raise RuntimeError(f'Incomplete run requires a new directory: {out}')
        if job['code_commit'] != commit() or job['source_hashes'] != provenance():
            raise ValueError('Worker source differs from frozen protocol')
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('medium')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        original_load = torch.load
        def forbid_load(*args, **kwargs):
            raise AssertionError('No pretrained weight loading during initialization')
        torch.load = forbid_load
        try:
            model, reference, args, cfg, common_hash = paired_models(
                job['source']['config'], job['arm'], job['head'], job['shared_gate'],
                job['variant'])
        finally:
            torch.load = original_load
        del reference
        lr_mult = float(job['variant']['extension_lr_mult'])
        if lr_mult != 1.0:
            _attach_extension_lr_multiplier(model, lr_mult)
        _, train_loader = data_provider(args.dataset_args, 'train')
        val_set, val_loader = data_provider(args.dataset_args, 'val')
        hp = cfg['hyperparams']
        trainer, checkpoint = build_trainer(max_epochs=1 if job['smoke'] else cfg['max_epochs'],
            logger=build_logger(str(out / 'lightning'), version='train'),
            patience=int(hp.get('patience', 8)), checkpoint_dir=str(out / 'checkpoints'), progress=False)
        if job['smoke']:
            trainer.limit_train_batches = 4
            trainer.limit_val_batches = 2
        pl.seed_everything(cfg['seed'], workers=True)
        model.train()
        started = time.monotonic()
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        elapsed = time.monotonic() - started
        restore_best_checkpoint(model, checkpoint)
        device = torch.device('cuda:0')
        model.to(device).eval()
        metrics = evaluate(model, val_loader, val_set, 'val', cfg['horizon'], out)
        _, test_loader = data_provider(args.dataset_args, 'test')
        branch = model.lowrank_interaction_branch
        sums = {m: [0.0, 0.0, 0] for m in ('native', 'off')}
        with torch.inference_mode():
            for batch in test_loader:
                bx, by, bxm, bym = [v.float().to(device) for v in batch]
                true = by[:, -model.pred_len:]
                for mode in ('native', 'off'):
                    branch.intervention = mode
                    pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
                    err = pred - true
                    sums[mode][0] += err.square().sum().item()
                    sums[mode][1] += err.abs().sum().item()
                    sums[mode][2] += err.numel()
        branch.intervention = 'native'
        save_json(out / 'tune_result.json', {
            'name': name(job), 'stage': 'tune', 'variant': job['variant'],
            **metrics,
            'test_mse': sums['native'][0] / sums['native'][2],
            'test_mae': sums['native'][1] / sums['native'][2],
            'test_mse_modulation_off': sums['off'][0] / sums['off'][2],
            'test_mae_modulation_off': sums['off'][1] / sums['off'][2],
            'best_val_loss': float(checkpoint.best_model_score),
            'parameters': parameter_breakdown(model), 'common_random_state_sha256': common_hash,
            'code_commit': commit(), 'elapsed_sec': elapsed, 'epochs': trainer.current_epoch,
            'smoke': job['smoke']})


def tune_select(scratch, smoke):
    """Select one extension variant per tuned setting against the main-stage
    configuration, on the test split (test-set selection, user decision
    2026-10-10). Every candidate number is recorded here so the selection is
    auditable; reports built on it must label it test-set selection.

    The comparison is seed-paired: the main-stage configuration is the variant
    ``None`` of prepare_config and every tune run shares its common random
    state, so the tune reads and the main seed-TUNE_SEED test reads differ only
    in the variant fields. A setting keeps the main-stage configuration unless
    some variant beats it; those variant winners are rerun on CONFIRM_SEEDS.
    """
    main = {}
    for path in (scratch / 'main' / 'runs').glob(f'*_S{TUNE_SEED}__m1__*/test_result.json'):
        r = json.loads(path.read_text())
        dataset, rest = r['name'].split('_H', 1)
        horizon = int(rest.split('_')[0])
        if dataset in TUNE_DATASETS:
            main[(dataset, horizon)] = r['test_mse']
    expected = {(d, h) for d in TUNE_DATASETS for h in HORIZONS}
    if set(main) != expected:
        raise ValueError(f'Missing main test reads: {sorted(expected - set(main))}')
    tune = {}
    for path in (scratch / 'tune' / 'runs').glob(f'*_S{TUNE_SEED}__*/tune_result.json'):
        r = json.loads(path.read_text())
        if r['smoke'] != smoke:
            raise ValueError('Smoke and formal results must not mix')
        dataset, rest = r['name'].split('_H', 1)
        horizon = int(rest.split('_')[0])
        variant_id = int(r['name'].rsplit('__t', 1)[1])
        tune[(dataset, horizon, variant_id)] = r['test_mse']
    missing = {(d, h, v) for d in TUNE_DATASETS for h in HORIZONS
               for v in range(len(TUNE_GRID))} - set(tune)
    if missing:
        raise ValueError(f'Tune round incomplete: missing {sorted(missing)}')
    settings = {}
    for key in sorted(expected):
        dataset, horizon = key
        best = min(range(len(TUNE_GRID)), key=lambda v: tune[(dataset, horizon, v)])
        best_mse = tune[(dataset, horizon, best)]
        keep_main = best_mse >= main[key]
        settings[f'{dataset}_H{horizon}'] = {
            'chosen': None if keep_main else best,
            'main_test_mse': main[key],
            'variant_test_mse': {f't{v}': tune[(dataset, horizon, v)] for v in range(len(TUNE_GRID))},
            'margin_vs_main': (main[key] - best_mse) / main[key],
        }
    record = {
        'selection_basis': ('test split, seed-paired single-seed reads; test-set '
                            'selection by user decision 2026-10-10; exploratory'),
        'rule': 'choose the variant with the lowest test MSE; keep the main-stage '
                'configuration when no variant beats it',
        'confirm_seeds': list(CONFIRM_SEEDS),
        'settings': settings, 'smoke': smoke,
    }
    write_immutable(scratch / 'tune_selection.json', record)
    return record


def plan_jobs(stage, scratch, sources, selection, smoke):
    base = {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': smoke, 'stage': stage}
    specs = []
    if stage == 'pilot':
        for dataset, horizon in PILOT:
            for head in HEADS:
                for gate in GATES:
                    specs.append((dataset, horizon, PILOT_SEED, 'm1', head, gate))
    elif stage == 'tune':
        for dataset in TUNE_DATASETS:
            for horizon in HORIZONS:
                for vid, variant in enumerate(TUNE_GRID):
                    specs.append((dataset, horizon, TUNE_SEED, 'm1', selection['head'],
                                  selection['shared_gate'][dataset], vid, dict(variant)))
    elif stage == 'confirm':
        chosen = json.loads((scratch / 'tune_selection.json').read_text())
        if chosen['smoke'] != smoke:
            raise ValueError('Smoke and formal selections must not mix')
        for setting, record in chosen['settings'].items():
            if record['chosen'] is None:
                continue
            dataset, horizon = setting.split('_H')
            for seed in CONFIRM_SEEDS:
                specs.append((dataset, int(horizon), seed, 'm1', selection['head'],
                              selection['shared_gate'][dataset], record['chosen'],
                              dict(TUNE_GRID[record['chosen']])))
    else:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    for arm in ARMS:
                        specs.append((dataset, horizon, seed, arm, selection['head'],
                                      selection['shared_gate'][dataset]))
    jobs = []
    for spec in specs:
        dataset, horizon, seed, arm, head, gate = spec[:6]
        job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm=arm, head=head,
                   shared_gate=gate, source=sources[(dataset, horizon, seed)])
        if len(spec) > 6:
            job.update(variant_id=spec[6], variant=spec[7])
        job['output'] = str(scratch / stage / 'runs' / name(job))
        path = scratch / stage / 'jobs' / f'{name(job)}.json'
        write_immutable(path, job)
        jobs.append(path)
    # Slow, many-channel settings start first so they do not form the tail.
    order = {'Traffic': 0, 'Electricity': 1, 'Weather': 2}
    return sorted(jobs, key=lambda p: (order.get(p.name.split('_')[0], 3), p.name))


def select(scratch, smoke):
    rows = {}
    for path in (scratch / 'pilot' / 'runs').glob('*/result.json'):
        r = json.loads(path.read_text())
        if r['smoke'] != smoke:
            raise ValueError('Smoke and formal pilot results must not mix')
        dataset, rest = r['name'].split('_H', 1)
        horizon = int(rest.split('_')[0])
        _, _, head, gate = r['name'].split('__')
        rows[(dataset, horizon, head, gate == 'shared')] = r['val_mse']
    expected = {(d, h, head, g) for d, h in PILOT for head in HEADS for g in GATES}
    if set(rows) != expected:
        raise ValueError(f'Pilot incomplete: missing {sorted(expected - set(rows))}')
    # Head: setting-normalized validation MSE averaged over settings and both gates.
    score = {}
    for head in HEADS:
        values = []
        for d, h in PILOT:
            best = min(rows[(d, h, k, g)] for k in HEADS for g in GATES)
            values += [rows[(d, h, head, g)] / best for g in GATES]
        score[head] = sum(values) / len(values)
    best = min(score.values())
    head = next(k for k in HEAD_PREFERENCE if score[k] <= best * (1 + HEAD_TIE))
    shared = {}
    for d, h in PILOT:
        shared[d] = rows[(d, h, head, True)] <= rows[(d, h, head, False)] * (1 + GATE_TIE)
    selection = {'head': head, 'shared_gate': shared, 'head_scores': score,
                 'rule': {'head': f'lowest mean setting-normalized val MSE; within {HEAD_TIE:.1%} prefer {HEAD_PREFERENCE}',
                          'gate': f'per dataset, shared if its val MSE <= per-variable x (1+{GATE_TIE})'},
                 'pilot_val_mse': {f'{d}_H{h}__{k}__{"shared" if g else "perC"}': v
                                   for (d, h, k, g), v in sorted(rows.items())},
                 'test_read': False, 'smoke': smoke}
    write_immutable(scratch / 'selection.json', selection)
    return selection


def run_jobs(jobs, gpus, worker_flag):
    tasks = queue.Queue()
    for path in jobs:
        tasks.put(path)
    failures = []

    def bucket(gpu):
        while True:
            try:
                path = tasks.get_nowait()
            except queue.Empty:
                return
            job = json.loads(Path(path).read_text())
            out = Path(job['output'])
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            with (out / f'{worker_flag[2:]}.log').open('a') as log:
                code = subprocess.call([sys.executable, str(Path(__file__).resolve()), worker_flag, str(path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(str(path))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:5]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('pilot', 'select', 'main', 'test', 'tune',
                                            'tune-select', 'confirm'))
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots-per-gpu', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--job', type=Path)
    parser.add_argument('--test-job', type=Path)
    parser.add_argument('--tune-job', type=Path)
    args = parser.parse_args()
    if args.job:
        return train_job(args.job)
    if args.test_job:
        return test_job(args.test_job)
    if args.tune_job:
        return tune_job(args.tune_job)
    if args.stage is None or args.scratch_dir is None:
        parser.error('--stage and --scratch-dir are required')
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    gpus = [g for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    with (scratch / f'{args.stage}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'select':
            print(json.dumps(select(scratch, args.smoke), indent=1))
            return
        if args.stage == 'tune-select':
            print(json.dumps(tune_select(scratch, args.smoke), indent=1))
            return
        sources = e14_sources()
        selection = None
        if args.stage in ('main', 'test', 'tune', 'confirm'):
            selection = json.loads((scratch / 'selection.json').read_text())
        jobs = plan_jobs('main' if args.stage == 'test' else args.stage, scratch, sources, selection, args.smoke)
        write_immutable(scratch / f'{args.stage}_manifest.json', {
            'stage': args.stage, 'jobs': [str(p) for p in jobs], 'code_commit': commit(),
            'e14_results_sha256': digest(E14_ROOT / E14), 'selection': selection, 'smoke': args.smoke})
        print(json.dumps({'stage': args.stage, 'jobs': len(jobs), 'code': commit()}), flush=True)
        if args.prepare_only:
            return
        if args.stage == 'test':
            missing = [p for p in jobs if not (Path(json.loads(p.read_text())['output']) / 'result.json').exists()]
            if missing:
                raise RuntimeError(f'Test read refused: {len(missing)} main runs unfinished')
            run_jobs(jobs, gpus, '--test-job')
        elif args.stage in ('tune', 'confirm'):
            run_jobs(jobs, gpus, '--tune-job')
        else:
            run_jobs(jobs, gpus, '--job')


if __name__ == '__main__':
    main()
