"""Conference training protocol of PhaseFormer and the final PhaseFormer-LA configuration.

Plan: docs/PhaseFormer_VLDBJ_LA_protocol_warmstart_plan.md (experiments 1/2).

The ICLR numbers (the Golden of docs/PhaseFormer_gold_standard.md) were
produced by the per-dataset scripts ``run_{dataset}.py`` of commit d9cb0b3
(ETTh1-H720 from ff14c16). ``CONFERENCE`` is their *effective* configuration:
the scripts' ``PhaseFormerConfig`` hardcodes ``phase_attn_dropout=0.1``,
``phase_attn_window=None`` and ``phase_use_pos_embed=True`` (ff14c16's ETTh1
script reads dropout and positional embedding from its dict), so the Traffic-96
dict entries ``phase_attn_window=12``/``phase_use_pos_embed=False`` never
reached the model. tests/test_vldbj_proto.py re-derives the table from git.

Architecture, data pipeline, Huber loss (the conference loss is
``nn.HuberLoss/delta``, identical at delta=1) and the constant-learning-rate
Adam are shared with E14; ``conference_source`` overrides only the fields that
differ and checks the architecture is the conference one. The conference
scripts fixed seed 2021 (ETTh1-H720: 2026); the experiments keep seeds
2021-2023 as the paper states.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

CONFERENCE_SOURCES = {
    # dataset: (commit, script, git blob of the script)
    'Electricity': ('d9cb0b3', 'run_electricity.py', 'ed8dd12542a1641faad3ca1797fd029cbbc90f4c'),
    'ETTh1': ('ff14c16', 'run_etth1.py', '6a11a15cda37d99a9fd39cf44c6322718d582ab7'),
    'ETTh2': ('d9cb0b3', 'run_etth2.py', 'f275c8c597f9788cacc6a414ec49174d2d3ca028'),
    'ETTm1': ('d9cb0b3', 'run_ettm1.py', '7eac5d6e179b981a273f7a5c872cfbe28f84524f'),
    'ETTm2': ('d9cb0b3', 'run_ettm2.py', '2a0e17a104c1bc1cf696f55cfa4cec06e39850d1'),
    'Traffic': ('d9cb0b3', 'run_traffic.py', '40a1353a6122e5eddf8de8cfcbd8116d1bda0ef5'),
    'Weather': ('d9cb0b3', 'run_weather.py', '4d4b1537726ae03c22608555f469e947641b2fa2'),
}

# Architecture fields must already agree with the E14 source; training fields
# and the effective attention options are overridden.
ARCH_FIELDS = ('layers', 'latent_dim', 'phase_encoder_hidden', 'predictor_hidden',
               'phase_num_routers', 'phase_attn_heads')
EFFECTIVE_FIELDS = ('phase_attn_dropout', 'phase_attn_window', 'phase_use_pos_embed', 'huber_delta')
# PhaseFormerPresetConfig / make_exp_args defaults for fields a source may omit.
PRESET_DEFAULTS = {'phase_attn_dropout': 0.1, 'phase_attn_window': None, 'phase_use_pos_embed': True,
                   'huber_delta': 1.0, 'patience': 8}

_SMALL = dict(layers=2, latent_dim=8, phase_encoder_hidden=32, predictor_hidden=64,
              phase_num_routers=8, phase_attn_heads=1)
_WIDE = dict(layers=1, latent_dim=128, phase_encoder_hidden=16, predictor_hidden=32,
             phase_num_routers=4, phase_attn_heads=4)
_COMMON = dict(phase_attn_dropout=0.1, phase_attn_window=None, phase_use_pos_embed=True,
               huber_delta=1.0, learning_rate=0.001, max_epochs=30, patience=8)


def _entry(arch, batch, **overrides):
    return {**_COMMON, **arch, 'batch_size': batch, **overrides}


CONFERENCE = {
    ('ETTh1', 96): _entry(dict(_SMALL, layers=3, latent_dim=4, phase_encoder_hidden=16, predictor_hidden=32), 256),
    ('ETTh1', 192): _entry(dict(_SMALL, layers=3, latent_dim=4, phase_encoder_hidden=16, predictor_hidden=32), 256),
    ('ETTh1', 336): _entry(dict(_SMALL, layers=3, latent_dim=4, phase_encoder_hidden=16, predictor_hidden=32), 256),
    ('ETTh1', 720): _entry(dict(layers=3, latent_dim=32, phase_encoder_hidden=128, predictor_hidden=256,
                                phase_num_routers=16, phase_attn_heads=2), 256,
                           phase_attn_dropout=0.0, huber_delta=0.3, learning_rate=0.00015,
                           max_epochs=70, patience=14),
    ('ETTh2', 96): _entry(dict(_SMALL, layers=1), 256),
    ('ETTh2', 192): _entry(dict(_SMALL, layers=1), 256),
    ('ETTh2', 336): _entry(dict(_SMALL, layers=1), 256),
    ('ETTh2', 720): _entry(dict(layers=1, latent_dim=4, phase_encoder_hidden=8, predictor_hidden=8,
                                phase_num_routers=4, phase_attn_heads=1), 256),
    ('ETTm1', 96): _entry(_SMALL, 16),
    ('ETTm1', 192): _entry(_SMALL, 16),
    ('ETTm1', 336): _entry(dict(_SMALL, layers=1), 16),
    ('ETTm1', 720): _entry(_SMALL, 16),
    ('ETTm2', 96): _entry(_SMALL, 16),
    ('ETTm2', 192): _entry(dict(_SMALL, layers=1), 16),
    ('ETTm2', 336): _entry(dict(_SMALL, layers=1), 16),
    ('ETTm2', 720): _entry(dict(_SMALL, layers=1), 16),
    ('Weather', 96): _entry(dict(_SMALL, layers=3), 16),
    ('Weather', 192): _entry(_SMALL, 16),
    ('Weather', 336): _entry(_SMALL, 16),
    ('Weather', 720): _entry(_SMALL, 16),
    ('Electricity', 96): _entry(_SMALL, 16, learning_rate=0.002),
    ('Electricity', 192): _entry(_WIDE, 16),
    ('Electricity', 336): _entry(_SMALL, 16),
    ('Electricity', 720): _entry(_WIDE, 16),
    ('Traffic', 96): _entry(dict(layers=2, latent_dim=32, phase_encoder_hidden=64, predictor_hidden=128,
                                 phase_num_routers=1, phase_attn_heads=8), 16),
    ('Traffic', 192): _entry(_WIDE, 16),
    ('Traffic', 336): _entry(_WIDE, 16),
    ('Traffic', 720): _entry(_WIDE, 16),
}

# Final PhaseFormer-LA configuration of the paper (commit 87c524a): the pilot
# selection (research_runs/vldbj_m1_pilot/selection.json) plus the eight
# per-setting tuning switches that survived the confirm round (agent-log
# 2026-10-11). Values are indices into scripts.vldbj_m1.run.TUNE_GRID; every
# other setting uses the main-stage configuration. L (b0) always uses the
# main-stage configuration. These choices were made on the test split
# (test-set selection, user decision 2026-10-10).
FINAL_HEAD = 'S2'
FINAL_SHARED_GATE = {'ETTh1': False, 'ETTh2': True, 'ETTm1': True, 'ETTm2': True,
                     'Weather': False, 'Electricity': False, 'Traffic': False}
FINAL_VARIANTS = {('ETTm1', 96): 1, ('ETTm2', 96): 5, ('Electricity', 720): 1, ('Traffic', 720): 4,
                  ('Weather', 96): 5, ('Weather', 192): 0, ('Weather', 336): 5, ('Weather', 720): 3}

GOLD_DOC = ROOT / 'docs/PhaseFormer_gold_standard.md'


def golden():
    """{(dataset, horizon): (mse, mae)} from the gold-standard table."""
    rows = {}
    pattern = re.compile(r'^\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$')
    for line in GOLD_DOC.read_text().splitlines():
        m = pattern.match(line)
        if m:
            rows[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    if len(rows) != 28:
        raise ValueError(f'Expected 28 Golden settings, found {len(rows)}')
    return rows


def config_hash(cfg):
    payload = json.dumps({k: v for k, v in cfg.items() if k != 'config_hash'}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def conference_source(source):
    """E14 phase-only source config -> the same backbone under the conference protocol."""
    key = (source['dataset'], int(source['horizon']))
    target = CONFERENCE[key]
    cfg = copy.deepcopy(source)
    hp = cfg['hyperparams']
    for field in ARCH_FIELDS:
        if hp[field] != target[field]:
            raise ValueError(f'{key}: E14 {field}={hp[field]} differs from the conference {target[field]}')
    changed = {}
    for field in EFFECTIVE_FIELDS + ('learning_rate', 'patience'):
        before = hp.get(field, PRESET_DEFAULTS.get(field))
        if before != target[field]:
            changed[field] = [before, target[field]]
        hp[field] = target[field]
    hp['train_epochs'] = target['max_epochs']
    for field in ('batch_size', 'max_epochs'):
        if cfg[field] != target[field]:
            changed[field] = [cfg[field], target[field]]
        cfg[field] = target[field]
    cfg['protocol'] = 'conference'
    cfg['protocol_changes'] = changed
    cfg['e14_config_hash'] = source.get('config_hash')
    cfg['config_hash'] = config_hash(cfg)
    return cfg
