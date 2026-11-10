"""Small latent messages on the unchanged PhaseFormer-L branch interface."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from scripts.run_lowrank_interaction_stage2 import construct as construct_legacy

ARMS = ('b0', 'pt_add', 'pt_correct', 'tp_add', 'tp_mod')


def validate_protocol(config):
    hp = config['hyperparams']
    allowed = {'use_lowrank_interaction', 'use_residual_head', 'use_revin', 'use_huber_loss'}
    enabled = {k for k, v in hp.items() if k.startswith('use_') and v}
    if enabled - allowed:
        raise ValueError(f'Unrelated feature enabled: {enabled - allowed}')
    loss = 'huber' if hp.get('use_huber_loss', True) else str(hp.get('loss_func', 'mse')).lower()
    if loss != 'huber' or not hp.get('use_revin', True) or not hp.get('use_residual_head', True):
        raise ValueError('Effective Huber/RevIN/residual protocol changed')
    if hp.get('period_len', 24) != 24 or hp.get('weak_residual_asymmetric_component', 'none') != 'none':
        raise ValueError('Period or input decomposition changed')
    if hp.get('weak_period_residual_head_type') != 'pooled_lowrank':
        raise ValueError('Legacy low-rank head required')
    if hp.get('weak_period_residual_rank') != config['horizon'] // 8:
        raise ValueError('Original rank required')
    if hp.get('weak_period_residual_pool_factor') != 1 or hp.get('weak_period_residual_smooth_ratio', 0) != 0:
        raise ValueError('Pooling or smoothing changed')


class TokenCommunication(nn.Module):
    def __init__(self, mode, rank, phase_dim, width=2):
        super().__init__()
        if mode not in ARMS[1:]:
            raise ValueError(mode)
        if min(rank, phase_dim, width) < 1:
            raise ValueError('Communication dimensions must be positive')
        self.mode = mode
        self.width = min(width, rank, phase_dim)
        if mode.startswith('pt_'):
            self.source = nn.Linear(phase_dim, self.width, bias=False)
            if mode == 'pt_correct':
                self.state = nn.Linear(rank, self.width, bias=False)
            self.output = nn.Linear(self.width, rank, bias=False)
        else:
            self.source = nn.Linear(rank, self.width, bias=False)
            self.output = nn.Linear(self.width, phase_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, trend, phase):
        if self.mode.startswith('pt_'):
            message = self.source(phase.mean(-2))
            if self.mode == 'pt_correct':
                message = message - self.state(trend)
            delta = self.output(message)
            return trend + delta, phase, delta
        message = self.output(self.source(trend)).unsqueeze(-2)
        delta = message.expand_as(phase) if self.mode == 'tp_add' else phase * message
        return trend, phase + delta, delta


class IsolatedBranch(nn.Module):
    """Replace only a private model instance's legacy no-interaction branch."""
    def __init__(self, legacy, phase_dim, mode, width=2):
        super().__init__()
        if legacy.interaction_mode != 'none':
            raise ValueError('Only the legacy no-interaction branch can be wrapped')
        self.lowrank_head = legacy.lowrank_head
        self.rank = legacy.rank
        self.phase_dim = phase_dim
        self.communication = TokenCommunication(mode, self.rank, phase_dim, width)
        self.intervention = 'native'
        self.capture_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, x, phase_latent):
        trend, last = self.lowrank_head.encode_latent(x)
        if self.intervention == 'native':
            updated_trend, updated_phase, delta = self.communication(trend, phase_latent)
        elif self.intervention == 'off':
            updated_trend, updated_phase = trend, phase_latent
            delta = torch.zeros_like(trend if self.communication.mode.startswith('pt_') else phase_latent)
        else:
            raise ValueError(self.intervention)
        if self.capture_diagnostics:
            original = trend if self.communication.mode.startswith('pt_') else phase_latent
            dimensions = (-1,) if original.ndim == 3 else (-2, -1)
            update_norm = delta.square().sum(dimensions).sqrt()
            original_norm = original.square().sum(dimensions).sqrt()
            self.last_diagnostics = {
                'update_norm': update_norm.detach(),
                'original_norm': original_norm.detach(),
                'relative_update_norm': (update_norm / original_norm.clamp_min(1e-8)).detach(),
            }
        readout = self.lowrank_head.decode_latent(updated_trend, last)
        if getattr(self, 'level_readout', 'native') == 'zero':
            readout = torch.zeros_like(readout)
        return readout, {
            'updated_phase_latent': updated_phase,
            'rank_latent': updated_trend,
            'phase_attention': None,
            'joint_attention': None,
        }


def prepare_config(source, arm, epochs=30):
    if arm not in ARMS:
        raise ValueError(arm)
    validate_protocol(source)
    config = copy.deepcopy(source)
    if source.get('init_checkpoint'):
        raise ValueError('Source must be a from-scratch protocol')
    config.update(checkpoint='', init_checkpoint='', evaluate_test=False, max_epochs=epochs)
    config['hyperparams'].update(
        use_lowrank_interaction=True, trend_interaction_mode='none',
        lowrank_fusion_mode='components_static', lowrank_update_style='post_norm',
        lowrank_null_read=False, lowrank_selective_write=False, lowrank_content_mode='none',
        weak_period_residual_gate_init=0.2, train_epochs=epochs, patience=8)
    config['token_communication'] = {'arm': arm, 'width': 2, 'zero_init_exit': True}
    return config


def construct(config, workers=0):
    validate_protocol(config)
    hp = config['hyperparams']
    if hp['trend_interaction_mode'] != 'none' or hp['lowrank_fusion_mode'] != 'components_static':
        raise ValueError('This experiment fixes the static component output gate')
    if config.get('init_checkpoint') or config.get('evaluate_test'):
        raise ValueError('Pretrained initialization and test selection are forbidden')
    model, args = construct_legacy(config, workers=workers)
    if not model.use_revin or model.period_len != 24 or args.training_args.loss_func != 'huber':
        raise AssertionError('Effective model protocol differs')
    protocol = config['token_communication']
    arm = protocol['arm']
    if arm not in ARMS:
        raise ValueError(arm)
    if arm != 'b0':
        model.lowrank_interaction_branch = IsolatedBranch(
            model.lowrank_interaction_branch, model.latent_dim, arm, protocol['width'])
    return model, args


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def random_models(source, arm, epochs):
    import pytorch_lightning as pl
    reference_config = prepare_config(source, 'b0', epochs)
    pl.seed_everything(source['seed'], workers=True)
    reference, args = construct(reference_config)
    common = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    config = prepare_config(source, arm, epochs)
    pl.seed_everything(source['seed'], workers=True)
    model, _ = construct(config)
    current = model.state_dict()
    if not set(common) <= set(current):
        raise AssertionError('Candidate removed baseline parameters')
    for name, value in common.items():
        if current[name].shape != value.shape:
            raise AssertionError(name)
        current[name].copy_(value)
    if not all(torch.equal(model.state_dict()[k], v) for k, v in common.items()):
        raise AssertionError('Unmatched common initialization')
    return model, reference, args, config, state_hash(common)


def load_model(run_dir, device):
    import json
    from scripts.run_lowrank_interaction_stage2 import digest
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    result = json.loads((run_dir / 'result.json').read_text())
    setting = f"{config['dataset']}_H{config['horizon']}_S{config['seed']}_val"
    if result['setting'] != setting or result['arm'] != config['token_communication']['arm']:
        raise ValueError('Checkpoint/config identity mismatch')
    if result['checkpoint'] != config['checkpoint'] or result['pretrained_checkpoint_loaded']:
        raise ValueError('Checkpoint initialization provenance mismatch')
    if digest(config['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    model, args = construct(config)
    state = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), args
