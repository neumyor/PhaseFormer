#!/usr/bin/env python3
"""Controller for experiments 1/2 of the VLDBJ extension.

Plan: docs/PhaseFormer_VLDBJ_LA_protocol_warmstart_plan.md. Every test-based
choice below (protocol adoption, checkpoint rule, warm-start backbone learning
rate) is test-set selection by user decision (2026-10-10) and is recorded with
all candidate numbers; reports must label it as such.

Stages (each refuses to run out of order):
  stage0      -- PhaseFormer under the conference protocol, seed 2021; records the
                 best-validation and last-epoch test reads and, for diagnosis only,
                 the per-epoch test trajectory of the conference scripts
  decide      -- pre-registered adoption rule against the Golden and E14
  main        -- conference protocol: PhaseFormer seeds 2022/2023, L and LA
                 seeds 2021-2023 (only if the conference protocol was adopted)
  warm-probe  -- LA initialized from the seed-2021 PhaseFormer checkpoint, two
                 backbone learning-rate multipliers
  warm-select -- per-setting multiplier by test MSE
  warm-main   -- the selected warm start on seeds 2022/2023
  collect     -- results.csv and summary.json over every finished run
  all         -- the stages above in order, with main and warm-probe in one queue
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_lowrank_interaction_stage2 import digest, save_json  # noqa: E402
from scripts.vldbj_m1 import protocol  # noqa: E402
from scripts.vldbj_m1.run import (  # noqa: E402
    DATASETS, E14, E14_ROOT, EXTENSION_PREFIXES, HORIZONS, SEEDS, TUNE_GRID, commit, e14_sources,
    provenance as m1_provenance, write_immutable)

PROTO_ARMS = ('pf', 'b0', 'm1')
STAGE0_SEED = 2021
WARM_SEED = 2021
WARM_GATE_INIT = 1e-3
WARM_INIT_TOLERANCE = 0.005
BACKBONE_LR_MULTS = (0.1, 1.0)
# Adoption rule (plan section 2, Stage 0).
ADOPT_MAX_MEAN_GAP = 0.01
ADOPT_MIN_DATASETS = 5
RULE_TIE = 0.002
SLOW_FIRST = {'Traffic': 0, 'Electricity': 1, 'Weather': 2, 'ETTm1': 3, 'ETTm2': 4}


def provenance():
    hashes = m1_provenance()
    for rel in ('tests/test_vldbj_proto.py', 'docs/PhaseFormer_gold_standard.md'):
        hashes[rel] = digest(ROOT / rel)
    return hashes


def name(job):
    base = f"{job['dataset']}_H{job['horizon']}_S{job['seed']}__{job['arm']}__{job['protocol']}"
    if job['init'] == 'warm':
        base += f"__warm_bb{job['backbone_lr_mult']:g}"
    if job.get('static_q'):
        base += '__staticq'
    return base


def e14_records():
    """E14 phase-only rows with their checkpoint, keyed by (dataset, horizon, seed)."""
    rows = {}
    with (E14_ROOT / E14).open() as handle:
        for row in csv.DictReader(handle):
            if row['arm'] != 'phase_only':
                continue
            run_dir = Path(row['run_dir'])
            run_dir = run_dir if run_dir.is_absolute() else E14_ROOT / run_dir
            # Rows reused from earlier rounds carry only the recorded validation MSE.
            ckpts = sorted(run_dir.glob('attempts/*/checkpoints/best.ckpt'))
            rows[(row['dataset'], int(row['horizon']), int(row['seed']))] = {
                'test_mse': float(row['test_mse']), 'test_mae': float(row['test_mae']),
                'val_mse': float(row['val_mse'] or row['recorded_val_mse']), 'checkpoint': str(ckpts[-1]) if ckpts else ''}
    return rows


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def source_config(job):
    cfg = job['source']['config']
    return protocol.conference_source(cfg) if job['protocol'] == 'conference' else cfg


def build_model(job):
    """Fresh model of the job's arm from the seed's common random state."""
    from scripts.lowrank_token_interaction.model import state_hash
    from scripts.run_lowrank_interaction_stage2 import construct as legacy_construct
    from scripts.vldbj_m1.model import paired_models
    import pytorch_lightning as pl
    src = source_config(job)
    if job['arm'] == 'pf':
        # PhaseFormer takes the backbone draw of the same-seed L/LA (the extension
        # is constructed first, so a plain PhaseFormer draw would differ): the
        # PF -> L -> LA ladder then shares one initialization.
        _, reference, _, _, common_hash = paired_models(src, 'b0', job['head'], job['shared_gate'])
        common = reference.state_dict()
        pl.seed_everything(src['seed'], workers=True)
        model, args = legacy_construct(src, workers=0)
        state = model.state_dict()
        for key, value in state.items():
            if common[key].shape != value.shape:
                raise AssertionError(key)
            value.copy_(common[key])
        return model, args, src, state_hash({k: common[k] for k in state})
    variant = dict(job['variant']) if job['variant'] else None
    if job['init'] == 'warm':
        variant = dict(variant or {}, gate_init=WARM_GATE_INIT)
    model, reference, args, cfg, common_hash = paired_models(
        src, job['arm'], job['head'], job['shared_gate'], variant, bool(job.get('static_q', False)))
    del reference
    return model, args, cfg, common_hash


def load_warm(model, warm):
    """Copy every PhaseFormer parameter into the backbone; the extension keeps its init."""
    import torch
    if digest(warm['checkpoint']) != warm['sha256']:
        raise ValueError('Warm-start checkpoint changed since planning')
    state = torch.load(warm['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    current = model.state_dict()
    missing, unexpected = set(current) - set(state), set(state) - set(current)
    if unexpected or any(not k.startswith(EXTENSION_PREFIXES) for k in missing):
        raise ValueError(f'Invalid warm start: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for key, value in state.items():
        if current[key].shape != value.shape:
            raise ValueError(f'Shape mismatch for {key}')
        current[key].copy_(value)
    return sorted(missing)


def attach_lr_groups(model, backbone_mult, extension_mult):
    """Adam with separate backbone and extension learning rates (constant, as the protocol)."""
    import types
    import torch
    base_lr = float(model.args.training_args.learning_rate)

    def configure_optimizers(self):
        backbone, extension = [], []
        for pname, p in self.named_parameters():
            if p.requires_grad:
                (extension if pname.startswith(EXTENSION_PREFIXES) else backbone).append(p)
        groups = [{'params': backbone, 'lr': base_lr * backbone_mult}]
        if extension:
            groups.append({'params': extension, 'lr': base_lr * extension_mult})
        optimizer = torch.optim.Adam(groups)
        if self.args.training_args.lr_schedule_config.type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.training_args.lr_schedule_config.tmax, eta_min=1e-8)
            return [optimizer], [scheduler]
        return optimizer

    model.configure_optimizers = types.MethodType(configure_optimizers, model)


def test_read(model, loader, arm, device):
    """Element-mean test MSE/MAE (and modulation off for LA) plus per-window MSE."""
    import numpy as np
    import torch
    modes = ('native', 'off') if arm == 'm1' else ('native',)
    branch = model.lowrank_interaction_branch if arm == 'm1' else None
    sums = {m: [0.0, 0.0, 0] for m in modes}
    windows = {m: [] for m in modes}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            true = by[:, -model.pred_len:]
            for mode in modes:
                if branch is not None:
                    branch.intervention = mode
                pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
                err = pred - true
                sums[mode][0] += err.square().sum().item()
                sums[mode][1] += err.abs().sum().item()
                sums[mode][2] += err.numel()
                windows[mode].append(err.square().mean(1).cpu().numpy().astype('float32'))
    if branch is not None:
        branch.intervention = 'native'
    record = {}
    for mode in modes:
        se, ae, n = sums[mode]
        suffix = '' if mode == 'native' else '_modulation_off'
        record[f'test_mse{suffix}'] = se / n
        record[f'test_mae{suffix}'] = ae / n
    record['test_size'] = int(sum(w.shape[0] for w in windows['native']))
    return record, {m: np.concatenate(w) for m, w in windows.items()}


def run_job(path):
    job = json.loads(Path(path).read_text())
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'result.json').exists():
            return
        if (out / 'checkpoints').exists():
            raise RuntimeError(f'Incomplete run requires a new directory: {out}')
        _run_job(job, out)


def _run_job(job, out):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import random
    import numpy as np
    import torch
    import pytorch_lightning as pl
    from src.dataset.data_factory import data_provider
    from src.training.runner import build_logger, build_trainer, restore_best_checkpoint
    from scripts.search_phaseformer import evaluate
    from scripts.vldbj_m1.model import parameter_breakdown
    from scripts.vldbj_m1.run import cycle_mean_check

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
        model, args, cfg, common_hash = build_model(job)
    finally:
        torch.load = original_load
    device = torch.device('cuda:0')
    _, train_loader = data_provider(args.dataset_args, 'train')
    val_set, val_loader = data_provider(args.dataset_args, 'val')
    _, test_loader = data_provider(args.dataset_args, 'test')
    warm_record = None
    if job['init'] == 'warm':
        missing = load_warm(model, job['warm_source'])
        model.to(device).eval()
        init_val = evaluate(model, val_loader, val_set, 'val', cfg['horizon'], out)['val_mse']
        reference = job['warm_source']['pf_val_mse']
        gap = abs(init_val - reference) / reference
        warm_record = {**job['warm_source'], 'extension_keys': missing, 'init_val_mse': init_val,
                       'init_val_relative_gap': gap}
        if gap > WARM_INIT_TOLERANCE and not job['smoke']:
            raise AssertionError(f'Warm start does not reproduce PhaseFormer: val gap {gap:.4%}')
    if job['backbone_lr_mult'] != 1.0 or job['extension_lr_mult'] != 1.0:
        attach_lr_groups(model, job['backbone_lr_mult'], job['extension_lr_mult'])
    epoch_times, epoch_tests = [], []

    class EpochTimer(pl.Callback):
        def on_train_epoch_start(self, trainer, module):
            torch.cuda.synchronize()
            self.start = time.monotonic()

        def on_train_epoch_end(self, trainer, module):
            torch.cuda.synchronize()
            epoch_times.append(time.monotonic() - self.start)

    class EpochTest(pl.Callback):
        """Diagnostic per-epoch test read of the conference scripts; never used for
        selection. RNG states are restored so training is unaffected."""

        def on_validation_epoch_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            states = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(),
                      random.getstate())
            was_training = module.training
            record, _ = test_read(module, test_loader, job['arm'], device)
            module.train(was_training)
            torch.set_rng_state(states[0])
            torch.cuda.set_rng_state_all(states[1])
            np.random.set_state(states[2])
            random.setstate(states[3])
            val_loss = trainer.callback_metrics.get('val_loss')
            epoch_tests.append({'epoch': trainer.current_epoch,
                                'val_loss': None if val_loss is None else float(val_loss),
                                **{k: v for k, v in record.items() if k.startswith('test_m')}})

    trainer, checkpoint = build_trainer(max_epochs=1 if job['smoke'] else cfg['max_epochs'],
        logger=build_logger(str(out / 'lightning'), version='train'),
        patience=int(cfg['hyperparams'].get('patience', 8)), checkpoint_dir=str(out / 'checkpoints'),
        progress=False)
    trainer.callbacks.append(EpochTimer())
    if job['epoch_test']:
        trainer.callbacks.append(EpochTest())
    if job['smoke']:
        trainer.limit_train_batches = 4
        trainer.limit_val_batches = 2
    pl.seed_everything(cfg['seed'], workers=True)
    model.to(device).train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.monotonic() - started
    # Conference rule: the weights at the end of fit (last epoch after early stopping).
    last_path = out / 'checkpoints' / 'last.pt'
    torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, last_path)
    model.to(device).eval()
    val_last = evaluate(model, val_loader, val_set, 'val', cfg['horizon'], out)
    test_last, windows_last = test_read(model, test_loader, job['arm'], device)
    restore_best_checkpoint(model, checkpoint)
    model.to(device).eval()
    val_best = evaluate(model, val_loader, val_set, 'val', cfg['horizon'], out)
    test_best, windows_best = test_read(model, test_loader, job['arm'], device)
    np.savez_compressed(out / 'test_window_mse.npz',
                        **{f'best_{m}': w for m, w in windows_best.items()},
                        **{f'last_{m}': w for m, w in windows_last.items()})
    cycle = cycle_mean_check(model, test_loader, device) if job['arm'] == 'm1' else None
    cfg['checkpoint'] = checkpoint.best_model_path
    save_json(out / 'config.json', cfg)
    save_json(out / 'result.json', {
        'name': name(job), 'stage': job['stage'], 'arm': job['arm'], 'protocol': job['protocol'],
        'init': job['init'], 'dataset': job['dataset'], 'horizon': job['horizon'], 'seed': job['seed'],
        'variant_id': job['variant_id'], 'backbone_lr_mult': job['backbone_lr_mult'],
        'extension_lr_mult': job['extension_lr_mult'],
        'best': {'val_mse': val_best['val_mse'], 'val_mae': val_best['val_mae'], **test_best},
        'last': {'val_mse': val_last['val_mse'], 'val_mae': val_last['val_mae'], **test_last},
        'epoch_test_diagnostic': epoch_tests or None, 'cycle_mean_check': cycle, 'warm_start': warm_record,
        'parameters': parameter_breakdown(model), 'common_random_state_sha256': common_hash,
        'code_commit': commit(), 'elapsed_sec': elapsed, 'epochs': trainer.current_epoch,
        'epoch_train_median_seconds': sorted(epoch_times)[len(epoch_times) // 2] if epoch_times else None,
        'checkpoint': checkpoint.best_model_path, 'checkpoint_sha256': digest(checkpoint.best_model_path),
        'last_checkpoint': str(last_path), 'last_checkpoint_sha256': digest(last_path),
        'best_val_loss': float(checkpoint.best_model_score), 'smoke': job['smoke'],
        'peak_memory_bytes': torch.cuda.max_memory_allocated(),
        'dataset_sha256': digest(Path(args.dataset_args.root_path) / args.dataset_args.data_path),
        'environment': {'torch': torch.__version__, 'lightning': pl.__version__,
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}})


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def settings_filter(raw, smoke):
    if not raw:
        return [(d, h) for d in DATASETS for h in HORIZONS]
    if not smoke:
        raise ValueError('--settings is only allowed with --smoke')
    out = []
    for item in raw.split(','):
        dataset, horizon = item.split('-')
        out.append((dataset, int(horizon)))
    return out


def make_job(base, sources, *, dataset, horizon, seed, arm, proto, init='scratch', backbone_lr_mult=1.0,
             warm_source=None, epoch_test=False):
    variant_id = protocol.FINAL_VARIANTS.get((dataset, horizon)) if arm == 'm1' else None
    variant = dict(TUNE_GRID[variant_id]) if variant_id is not None else None
    job = dict(base, dataset=dataset, horizon=horizon, seed=seed, arm=arm, protocol=proto, init=init,
               head=protocol.FINAL_HEAD, shared_gate=protocol.FINAL_SHARED_GATE[dataset],
               variant_id=variant_id, variant=variant,
               extension_lr_mult=float(variant['extension_lr_mult']) if variant else 1.0,
               backbone_lr_mult=float(backbone_lr_mult), warm_source=warm_source, epoch_test=epoch_test,
               source=sources[(dataset, horizon, seed)])
    if proto == 'conference':
        protocol.conference_source(job['source']['config'])  # fail at planning, not on the GPU
    return job


def write_jobs(scratch, stage, jobs):
    paths = []
    for job in jobs:
        job['output'] = str(scratch / stage / 'runs' / name(job))
        path = scratch / stage / 'jobs' / f'{name(job)}.json'
        write_immutable(path, job)
        paths.append(path)
    return sorted(paths, key=lambda p: (SLOW_FIRST.get(p.name.split('_')[0], 9), p.name))


def load_result(scratch, stage, job):
    path = scratch / stage / 'runs' / name(job) / 'result.json'
    return json.loads(path.read_text()) if path.exists() else None


def warm_source_for(scratch, decision, dataset, horizon, seed, e14):
    """The PhaseFormer weights the warm start reproduces, under the frozen rule."""
    if decision['adopt_conference']:
        stage = 'stage0' if seed == STAGE0_SEED else 'main'
        run = scratch / stage / 'runs' / f'{dataset}_H{horizon}_S{seed}__pf__conference'
        result = json.loads((run / 'result.json').read_text())
        rule = decision['checkpoint_rule']
        ckpt = result['checkpoint'] if rule == 'best' else result['last_checkpoint']
        return {'origin': f'{stage}/{run.name}', 'rule': rule, 'checkpoint': ckpt, 'sha256': digest(ckpt),
                'pf_val_mse': result[rule]['val_mse'], 'pf_test_mse': result[rule]['test_mse']}
    record = e14[(dataset, horizon, seed)]
    if not record['checkpoint']:
        raise ValueError(f'No E14 checkpoint for {(dataset, horizon, seed)}')
    return {'origin': 'e14', 'rule': 'best', 'checkpoint': record['checkpoint'],
            'sha256': digest(record['checkpoint']), 'pf_val_mse': record['val_mse'],
            'pf_test_mse': record['test_mse']}


def plan(stage, scratch, sources, settings, smoke):
    base = {'code_commit': commit(), 'source_hashes': provenance(), 'smoke': smoke, 'stage': stage}
    jobs = []
    if stage == 'stage0':
        for d, h in settings:
            jobs.append(make_job(base, sources, dataset=d, horizon=h, seed=STAGE0_SEED, arm='pf',
                                 proto='conference', epoch_test=True))
        return write_jobs(scratch, stage, jobs)
    decision = json.loads((scratch / 'protocol_decision.json').read_text())
    if decision['smoke'] != smoke:
        raise ValueError('Smoke and formal decisions must not mix')
    proto = 'conference' if decision['adopt_conference'] else 'e14'
    if stage == 'main':
        if not decision['adopt_conference']:
            raise RuntimeError('Conference protocol not adopted: experiment 1 stops after Stage 0')
        for d, h in settings:
            for seed in SEEDS:
                for arm in PROTO_ARMS:
                    if arm == 'pf' and seed == STAGE0_SEED:
                        continue
                    jobs.append(make_job(base, sources, dataset=d, horizon=h, seed=seed, arm=arm,
                                         proto='conference'))
        return write_jobs(scratch, stage, jobs)
    e14 = e14_records()
    if stage == 'warm-probe':
        for d, h in settings:
            warm = warm_source_for(scratch, decision, d, h, WARM_SEED, e14)
            for mult in BACKBONE_LR_MULTS:
                jobs.append(make_job(base, sources, dataset=d, horizon=h, seed=WARM_SEED, arm='m1',
                                     proto=proto, init='warm', backbone_lr_mult=mult, warm_source=warm))
        return write_jobs(scratch, stage, jobs)
    if stage == 'warm-main':
        chosen = json.loads((scratch / 'warm_selection.json').read_text())
        if chosen['smoke'] != smoke:
            raise ValueError('Smoke and formal selections must not mix')
        for d, h in settings:
            mult = chosen['settings'][f'{d}_H{h}']['backbone_lr_mult']
            for seed in SEEDS:
                if seed == WARM_SEED:
                    continue
                warm = warm_source_for(scratch, decision, d, h, seed, e14)
                jobs.append(make_job(base, sources, dataset=d, horizon=h, seed=seed, arm='m1',
                                     proto=proto, init='warm', backbone_lr_mult=mult, warm_source=warm))
        return write_jobs(scratch, stage, jobs)
    raise ValueError(stage)


# --------------------------------------------------------------------------
# Decisions (test-set selection, recorded with every candidate number)
# --------------------------------------------------------------------------
def decide(scratch, settings, smoke):
    gold = protocol.golden()
    e14 = e14_records()
    rows = {}
    for d, h in settings:
        path = scratch / 'stage0' / 'runs' / f'{d}_H{h}_S{STAGE0_SEED}__pf__conference' / 'result.json'
        if not path.exists():
            raise ValueError(f'Stage 0 incomplete: {d}-{h}')
        r = json.loads(path.read_text())
        if r['smoke'] != smoke:
            raise ValueError('Smoke and formal results must not mix')
        rows[(d, h)] = {'golden_mse': gold[(d, h)][0], 'e14_seed2021_test_mse': e14[(d, h, STAGE0_SEED)]['test_mse'],
                        'best_test_mse': r['best']['test_mse'], 'last_test_mse': r['last']['test_mse'],
                        'best_test_mae': r['best']['test_mae'], 'last_test_mae': r['last']['test_mae']}
    ratio = {rule: statistics.fmean(v[f'{rule}_test_mse'] / v['golden_mse'] for v in rows.values())
             for rule in ('best', 'last')}
    rule = 'last' if ratio['last'] < ratio['best'] - RULE_TIE else 'best'
    datasets = sorted({d for d, _ in settings})
    better = []
    for d in datasets:
        conf = statistics.fmean(v[f'{rule}_test_mse'] for (dd, _), v in rows.items() if dd == d)
        ref = statistics.fmean(v['e14_seed2021_test_mse'] for (dd, _), v in rows.items() if dd == d)
        if conf < ref:
            better.append(d)
    mean_gap = ratio[rule] - 1.0
    need = ADOPT_MIN_DATASETS if len(datasets) == len(DATASETS) else math.ceil(len(datasets) * 5 / 7)
    adopt = mean_gap <= ADOPT_MAX_MEAN_GAP and len(better) >= need
    record = {
        'selection_basis': 'test split against the Golden and E14 seed 2021; test-set selection '
                           '(user decision 2026-10-10)',
        'rule': {'checkpoint': f'lower mean test-MSE/Golden ratio; best-val unless last-epoch is lower by '
                               f'more than {RULE_TIE}',
                 'adopt': f'mean relative MSE gap to Golden <= {ADOPT_MAX_MEAN_GAP:.1%} and dataset-average '
                          f'MSE below E14 seed 2021 on >= {ADOPT_MIN_DATASETS}/7 datasets'},
        'checkpoint_rule': rule, 'mean_ratio_to_golden': ratio, 'mean_relative_gap': mean_gap,
        'datasets_better_than_e14': better, 'adopt_conference': adopt or smoke,
        'adopt_rule_passed': adopt, 'smoke': smoke,
        'settings': {f'{d}_H{h}': v for (d, h), v in sorted(rows.items())},
    }
    if smoke:
        record['note'] = 'smoke: adoption forced so every stage is exercised'
    write_immutable(scratch / 'protocol_decision.json', record)
    return record


def warm_select(scratch, settings, smoke):
    decision = json.loads((scratch / 'protocol_decision.json').read_text())
    rule = decision['checkpoint_rule'] if decision['adopt_conference'] else 'best'
    proto = 'conference' if decision['adopt_conference'] else 'e14'
    chosen = {}
    for d, h in settings:
        reads = {}
        for mult in BACKBONE_LR_MULTS:
            path = (scratch / 'warm-probe' / 'runs' /
                    f'{d}_H{h}_S{WARM_SEED}__m1__{proto}__warm_bb{mult:g}' / 'result.json')
            if not path.exists():
                raise ValueError(f'Warm probe incomplete: {path.parent.name}')
            r = json.loads(path.read_text())
            if r['smoke'] != smoke:
                raise ValueError('Smoke and formal results must not mix')
            reads[mult] = r[rule]['test_mse']
        best = min(BACKBONE_LR_MULTS, key=lambda m: (reads[m], m))
        chosen[f'{d}_H{h}'] = {'backbone_lr_mult': best, 'test_mse': {f'{m:g}': v for m, v in reads.items()}}
    record = {'selection_basis': f'seed-{WARM_SEED} test split ({rule} weights); test-set selection '
                                 '(user decision 2026-10-10)',
              'rule': 'lowest test MSE; ties go to the smaller multiplier',
              'settings': chosen, 'smoke': smoke}
    write_immutable(scratch / 'warm_selection.json', record)
    return record


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------
def collect(scratch, smoke):
    decision = json.loads((scratch / 'protocol_decision.json').read_text())
    rule = decision['checkpoint_rule'] if decision['adopt_conference'] else 'best'
    selection = None
    if (scratch / 'warm_selection.json').exists():
        selection = json.loads((scratch / 'warm_selection.json').read_text())['settings']
    gold = protocol.golden()
    fields = ['stage', 'arm', 'protocol', 'init', 'dataset', 'horizon', 'seed', 'variant_id', 'backbone_lr_mult',
              'selected', 'epochs', 'best_test_mse', 'best_test_mae', 'last_test_mse', 'last_test_mae',
              'best_test_mse_modulation_off', 'last_test_mse_modulation_off', 'best_val_mse', 'last_val_mse',
              'golden_mse', 'golden_mae', 'run']
    rows = []
    for path in sorted(scratch.glob('*/runs/*/result.json')):
        r = json.loads(path.read_text())
        if r['smoke'] != smoke:
            continue
        key = (r['dataset'], r['horizon'])
        selected = True
        if r['init'] == 'warm' and r['seed'] == WARM_SEED:
            selected = bool(selection) and selection[f'{key[0]}_H{key[1]}']['backbone_lr_mult'] == r['backbone_lr_mult']
        rows.append({
            'stage': r['stage'], 'arm': r['arm'], 'protocol': r['protocol'], 'init': r['init'],
            'dataset': key[0], 'horizon': key[1], 'seed': r['seed'], 'variant_id': r['variant_id'],
            'backbone_lr_mult': r['backbone_lr_mult'], 'selected': selected, 'epochs': r['epochs'],
            **{f'{w}_{k}': r[w].get(k) for w in ('best', 'last')
               for k in ('test_mse', 'test_mae', 'test_mse_modulation_off', 'val_mse')},
            'golden_mse': gold[key][0], 'golden_mae': gold[key][1], 'run': str(path.parent.relative_to(scratch))})
    with (scratch / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        if not row['selected']:
            continue
        label = {'pf': 'PhaseFormer', 'b0': 'PhaseFormer-L', 'm1': 'PhaseFormer-LA'}[row['arm']]
        if row['init'] == 'warm':
            label += ' (warm start)'
        label += f" [{row['protocol']}]"
        groups.setdefault((label, row['dataset'], row['horizon']), []).append(row)
    summary = []
    for (label, d, h), items in sorted(groups.items()):
        entry = {'arm': label, 'dataset': d, 'horizon': h, 'seeds': sorted(i['seed'] for i in items),
                 'golden_mse': gold[(d, h)][0], 'golden_mae': gold[(d, h)][1]}
        for metric in ('test_mse', 'test_mae', 'test_mse_modulation_off'):
            values = [i[f'{rule}_{metric}'] for i in items if i[f'{rule}_{metric}'] is not None]
            if values:
                entry[f'{metric}_mean'] = statistics.fmean(values)
                entry[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        entry['mse_change_vs_golden'] = (gold[(d, h)][0] - entry['test_mse_mean']) / gold[(d, h)][0]
        summary.append(entry)
    record = {'checkpoint_rule': rule, 'adopt_conference': decision['adopt_conference'],
              'selection_basis': 'protocol, checkpoint rule, LA variants and warm-start multipliers were '
                                 'chosen on the test split (test-set selection); all seeds counted',
              'smoke': smoke, 'summary': summary}
    save_json(scratch / 'summary.json', record)
    return record


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def run_jobs(jobs, gpus):
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
            with (out / 'job.log').open('a') as log:
                code = subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', str(path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(str(path))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:5]}')


def manifest(scratch, stage, jobs, smoke):
    write_immutable(scratch / f'{stage}_manifest.json', {
        'stage': stage, 'jobs': [str(p) for p in jobs], 'code_commit': commit(),
        'e14_results_sha256': digest(E14_ROOT / E14), 'smoke': smoke})
    print(json.dumps({'stage': stage, 'jobs': len(jobs), 'code': commit()}), flush=True)


def run_stage(stage, scratch, settings, gpus, smoke, prepare_only):
    if stage == 'decide':
        return decide(scratch, settings, smoke)
    if stage == 'warm-select':
        return warm_select(scratch, settings, smoke)
    if stage == 'collect':
        return collect(scratch, smoke)
    sources = e14_sources()
    jobs = plan(stage, scratch, sources, settings, smoke)
    manifest(scratch, stage, jobs, smoke)
    if not prepare_only:
        run_jobs(jobs, gpus)
    return None


def run_all(scratch, settings, gpus, smoke):
    sources = e14_sources()
    jobs = plan('stage0', scratch, sources, settings, smoke)
    manifest(scratch, 'stage0', jobs, smoke)
    run_jobs(jobs, gpus)
    decision = decide(scratch, settings, smoke)
    print(json.dumps({k: decision[k] for k in ('adopt_conference', 'checkpoint_rule', 'mean_relative_gap',
                                               'datasets_better_than_e14')}), flush=True)
    queued = []
    if decision['adopt_conference']:
        main = plan('main', scratch, sources, settings, smoke)
        manifest(scratch, 'main', main, smoke)
        queued += main
    probe = plan('warm-probe', scratch, sources, settings, smoke)
    manifest(scratch, 'warm-probe', probe, smoke)
    run_jobs(queued + probe, gpus)
    warm_select(scratch, settings, smoke)
    warm = plan('warm-main', scratch, sources, settings, smoke)
    manifest(scratch, 'warm-main', warm, smoke)
    run_jobs(warm, gpus)
    collect(scratch, smoke)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('stage0', 'decide', 'main', 'warm-probe', 'warm-select',
                                            'warm-main', 'collect', 'all'))
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots-per-gpu', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--settings', default='', help='smoke only, e.g. ETTh1-96,Weather-192')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        return run_job(args.job)
    if args.stage is None or args.scratch_dir is None:
        parser.error('--stage and --scratch-dir are required')
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    settings = settings_filter(args.settings, args.smoke)
    gpus = [g for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    with (scratch / f'{args.stage}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'all':
            return run_all(scratch, settings, gpus, args.smoke)
        result = run_stage(args.stage, scratch, settings, gpus, args.smoke, args.prepare_only)
        if result is not None:
            print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
