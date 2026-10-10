"""Compressed-branch M1 for the VLDBJ extension of PhaseFormer.

The ICLR backbone is unchanged. The level branch reads cycle means of the
anchored history with rank 3 (Theorem 4 of the extension: the best linear
complement has output rank at most the number of horizon profiles). M1 adds the
centered multiplicative phase modulation on top of static level/shape fusion.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.centered_mod_interaction.model import IsolatedBranch
from scripts.lowrank_token_interaction.model import state_hash
from scripts.run_lowrank_interaction_stage2 import construct as legacy_construct

ARMS = ('b0', 'm1')
RANK = 3
HEADS = {
    'S1': {'pool_factor': 24, 'decoder': 'learned'},
    'S2': {'pool_factor': 24, 'decoder': 'legendre'},
    'S3': {'pool_factor': 8, 'decoder': 'learned'},
}
PROFILE_COUNT = 3


def legendre_profiles(horizon, count=PROFILE_COUNT):
    """Displacement, tilt and curvature on the horizon, as (H, count).

    Discrete orthogonal polynomials on the forecast grid: the first column is
    the constant displacement and the others are exactly horizon-centered, as
    the profile model of Theorem 4 requires. Every column has unit RMS.
    """
    tau = torch.linspace(-1.0, 1.0, horizon, dtype=torch.float64)
    columns = []
    for power in range(count):
        v = tau ** power
        for u in columns:
            v = v - (v @ u) / (u @ u) * u
        columns.append(v / v.square().mean().sqrt())
    return torch.stack(columns, -1).float()


class ProfileLevelHead(nn.Module):
    """Anchored cycle-mean readout written onto fixed low-frequency profiles.

    Same encode/decode contract as ``PooledLowRankWeakPeriodResidualHead``;
    only the decoder is restricted to span{displacement, tilt, curvature}, so
    its parameter count does not depend on the horizon.
    """

    def __init__(self, seq_len, pred_len, *, pool_factor, rank):
        super().__init__()
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.pool_factor = int(pool_factor)
        self.pooled_len = math.ceil(self.seq_len / self.pool_factor)
        self.rank = int(rank)
        self.encoder = nn.Linear(self.pooled_len, self.rank)
        self.mix = nn.Linear(self.rank, PROFILE_COUNT)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.register_buffer('profiles', legendre_profiles(self.pred_len), persistent=False)

    def encode_latent(self, x):
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        return self.encoder(F.adaptive_avg_pool1d(centered, self.pooled_len)), last

    def decode_latent(self, latent, last):
        delta = self.mix(latent) @ self.profiles.T
        return delta.permute(0, 2, 1).contiguous() + last.expand(-1, self.pred_len, -1)

    def forward(self, x):
        return self.decode_latent(*self.encode_latent(x))


def prepare_config(source, arm, head, shared_gate):
    """Derive one registered config from an E14 phase-only source config."""
    if arm not in ARMS or head not in HEADS:
        raise ValueError((arm, head))
    if source.get('mechanism') != 'no_residual' or source.get('init_checkpoint'):
        raise ValueError('Source must be an E14 from-scratch phase-only config')
    cfg = copy.deepcopy(source)
    hp = cfg['hyperparams']
    if any(k.startswith('use_') and v and k not in {'use_huber_loss', 'use_revin'} for k, v in hp.items()):
        raise ValueError('Source enables an unrelated mechanism')
    hp.update(
        scheme_name='vldbj_m1', use_lowrank_interaction=True,
        weak_period_residual_head_type='pooled_lowrank',
        weak_period_residual_pool_factor=HEADS[head]['pool_factor'],
        weak_period_residual_rank=RANK, weak_period_residual_smooth_ratio=0.0,
        weak_period_residual_gate_init=0.2, trend_interaction_mode='none',
        lowrank_fusion_mode='components_static', lowrank_update_style='post_norm',
        lowrank_null_read=False, lowrank_selective_write=False, lowrank_content_mode='none')
    cfg.update(mechanism='vldbj_m1', checkpoint='', init_checkpoint='', evaluate_test=False)
    cfg['vldbj_m1'] = {'arm': arm, 'head': head, 'rank': RANK, 'shared_gate': bool(shared_gate),
                       'communication': 'tp_mod_centered' if arm == 'm1' else 'none',
                       'communication_width': 2, 'zero_init_exit': True,
                       'source_config_hash': source.get('config_hash')}
    return cfg


def _share_gates(model):
    """Replace per-variable static level/shape logits with two global scalars."""
    gate = model.lowrank_interaction_gate
    model.lowrank_interaction_gate = nn.Parameter(gate.detach()[..., :1].clone())
    fusion = model.lowrank_dynamic_fusion
    fusion.shape_bias_delta = nn.Parameter(fusion.shape_bias_delta.detach()[:, :1].clone())


def construct(config, workers=0):
    protocol = config['vldbj_m1']
    hp = config['hyperparams']
    if (config.get('init_checkpoint') or config.get('evaluate_test') or hp['trend_interaction_mode'] != 'none'
            or hp['lowrank_fusion_mode'] != 'components_static' or hp['weak_period_residual_rank'] != RANK):
        raise ValueError('Unregistered model protocol')
    model, args = legacy_construct(config, workers=workers)
    if not model.use_revin or model.period_len != 24 or args.training_args.loss_func != 'huber':
        raise AssertionError('Effective protocol changed')
    if model.predictor.use_mlp:
        raise ValueError('Cycle-mean constraint requires the shared linear predictor')
    branch = model.lowrank_interaction_branch
    if HEADS[protocol['head']]['decoder'] == 'legendre':
        branch.lowrank_head = ProfileLevelHead(model.seq_len, model.pred_len,
            pool_factor=HEADS[protocol['head']]['pool_factor'], rank=RANK)
    if protocol['shared_gate']:
        _share_gates(model)
    if protocol['arm'] == 'm1':
        model.lowrank_interaction_branch = IsolatedBranch(branch, model.latent_dim, 'tp_mod_centered')
    return model, args


def paired_models(source, arm, head, shared_gate):
    """Build ``arm`` with every b0 parameter copied from a same-seed b0."""
    import pytorch_lightning as pl
    pl.seed_everything(source['seed'], workers=True)
    reference, args = construct(prepare_config(source, 'b0', head, shared_gate))
    common = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    config = prepare_config(source, arm, head, shared_gate)
    pl.seed_everything(source['seed'], workers=True)
    model, _ = construct(config)
    current = model.state_dict()
    if not set(common) <= set(current):
        raise AssertionError('Baseline parameters removed')
    for name, value in common.items():
        if current[name].shape != value.shape:
            raise AssertionError(name)
        current[name].copy_(value)
    return model, reference, args, config, state_hash(common)


def parameter_breakdown(model):
    groups = {'backbone': 0, 'level_branch': 0, 'modulation': 0, 'fusion': 0}
    for name, p in model.named_parameters():
        if name.startswith('lowrank_interaction_branch.communication.'):
            groups['modulation'] += p.numel()
        elif name.startswith('lowrank_interaction_branch.'):
            groups['level_branch'] += p.numel()
        elif name.startswith(('lowrank_interaction_gate', 'lowrank_dynamic_fusion.')):
            groups['fusion'] += p.numel()
        else:
            groups['backbone'] += p.numel()
    groups['total'] = sum(groups.values())
    return groups
