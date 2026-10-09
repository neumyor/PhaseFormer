"""Shared width-two level writes conditioned on the current phase state."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn

from scripts.lowrank_token_interaction.model import (
    IsolatedBranch as PreviousBranch, validate_protocol, state_hash,
    prepare_config as previous_config)
from scripts.run_lowrank_interaction_stage2 import construct as legacy_construct, digest

ARMS = ('b0', 'tp_add', 'context_full', 'phase_only')
SUPPORTED = ARMS + ('context_zero', 'untied_write')


class Communication(nn.Module):
    def __init__(self, mode, rank, phase_dim, width=2):
        super().__init__()
        if mode not in SUPPORTED[1:]:
            raise ValueError(mode)
        self.mode = mode
        self.width = min(width, rank, phase_dim)
        if mode != 'phase_only':
            self.source = nn.Linear(rank, self.width, bias=False)
        if mode != 'tp_add':
            self.context = nn.Linear(phase_dim, self.width, bias=False)
        self.output = nn.Linear(self.width, phase_dim, bias=False)
        nn.init.zeros_(self.output.weight)
        if mode == 'untied_write':
            self.context_output = nn.Linear(self.width, phase_dim, bias=False)
            nn.init.zeros_(self.context_output.weight)

    def forward(self, trend, phase):
        if self.mode == 'tp_add':
            write = self.output(self.source(trend))
        else:
            context = phase.mean(-2)
            if self.mode == 'context_zero':
                context = torch.zeros_like(context)
            v = self.context(context)
            if self.mode == 'phase_only':
                write = -self.output(v)
            elif self.mode == 'untied_write':
                write = self.output(self.source(trend)) - self.context_output(v)
            else:
                write = self.output(self.source(trend) - v)
        delta = write.unsqueeze(-2).expand_as(phase)
        return trend, phase + delta, delta


class IsolatedBranch(PreviousBranch):
    def __init__(self, legacy, phase_dim, mode, width=2):
        # Reuse only the original encode/decode and diagnostic forward contract.
        nn.Module.__init__(self)
        if legacy.interaction_mode != 'none':
            raise ValueError('Expected unchanged no-interaction branch')
        self.lowrank_head = legacy.lowrank_head
        self.rank, self.phase_dim = legacy.rank, phase_dim
        self.communication = Communication(mode, self.rank, phase_dim, width)
        self.intervention = 'native'
        self.capture_diagnostics = False
        self.last_diagnostics = {}


def prepare_config(source, arm, epochs=30):
    if arm not in SUPPORTED:
        raise ValueError(arm)
    config = previous_config(source, 'b0', epochs)
    config['token_communication'] = {
        'arm': arm, 'width': 2, 'zero_init_exit': True,
        'experiment': 'phase_context_interaction', 'context_axis': 'phase_slots'}
    return config


def construct(config, workers=0):
    validate_protocol(config)
    hp = config['hyperparams']
    protocol = config['token_communication']
    if (config.get('init_checkpoint') or config.get('evaluate_test') or
            hp['trend_interaction_mode'] != 'none' or
            hp['lowrank_fusion_mode'] != 'components_static' or
            protocol.get('experiment') != 'phase_context_interaction' or
            protocol['width'] != 2 or not protocol['zero_init_exit']):
        raise ValueError('Unregistered model protocol')
    model, args = legacy_construct(config, workers=workers)
    arm = protocol['arm']
    if arm not in SUPPORTED:
        raise ValueError(arm)
    if not model.use_revin or model.period_len != 24 or args.training_args.loss_func != 'huber':
        raise AssertionError('Effective protocol changed')
    if arm != 'b0':
        model.lowrank_interaction_branch = IsolatedBranch(
            model.lowrank_interaction_branch, model.latent_dim, arm)
    return model, args


def random_models(source, arm, epochs):
    import pytorch_lightning as pl
    pl.seed_everything(source['seed'], workers=True)
    reference, args = construct(prepare_config(source, 'b0', epochs))
    common = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    config = prepare_config(source, arm, epochs)
    pl.seed_everything(source['seed'], workers=True)
    model, _ = construct(config)
    current = model.state_dict()
    if not set(common) <= set(current):
        raise AssertionError('Baseline parameters removed')
    for name, value in common.items():
        if current[name].shape != value.shape:
            raise AssertionError(name)
        current[name].copy_(value)
    if not all(torch.equal(model.state_dict()[k], v) for k, v in common.items()):
        raise AssertionError('Unmatched initialization')
    return model, reference, args, config, state_hash(common)


def load_model(run_dir, device):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    result = json.loads((run_dir / 'result.json').read_text())
    setting = f"{config['dataset']}_H{config['horizon']}_S{config['seed']}_val"
    if (result['setting'] != setting or result['arm'] != config['token_communication']['arm'] or
            result['checkpoint'] != config['checkpoint'] or result['pretrained_checkpoint_loaded'] or
            digest(config['checkpoint']) != result['checkpoint_sha256']):
        raise ValueError('Checkpoint provenance mismatch')
    model, args = construct(config)
    state = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), args
