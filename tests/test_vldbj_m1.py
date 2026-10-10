"""Checks for the compressed-branch M1 used in the VLDBJ extension."""
import unittest

import torch

from scripts.vldbj_m1.model import (
    HEADS, ProfileLevelHead, legendre_profiles, paired_models, parameter_breakdown, prepare_config)
from src.models.phaseformer_presets import get_base_hyperparams


def e14_source(dataset='ETTh1', horizon=96, seed=2021, batch=4):
    hp = get_base_hyperparams(dataset, horizon)
    hp.update(seed=seed, scheme_name='no_residual', period_len=24, loss_func='huber',
              use_huber_loss=True, train_epochs=30, use_residual_head=False)
    return dict(dataset=dataset, horizon=horizon, lookback=720, mechanism='no_residual', period=24,
                percent=100, max_epochs=30, seed=seed, loss='huber', batch_size=batch,
                evaluate_test=False, input_hypothesis='none', input_variant='full',
                intervention_seed=9102, input_d1_period=0.0, input_d1_sigma=0.0,
                input_d2_recent_length=0, input_d3_component='', init_checkpoint='',
                hyperparams=hp, config_hash='test')


def forward(model, x):
    b = x.shape[0]
    marks = torch.zeros(b, x.shape[1], 4)
    dec = torch.zeros(b, model.pred_len, x.shape[2])
    out, _, _ = model(x, marks, dec, torch.zeros(b, model.pred_len, 4))
    return out[:, -model.pred_len:]


class VLDBJM1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')

    def test_profiles_and_head_size_independent_of_horizon(self):
        for h in (96, 720):
            p = legendre_profiles(h)
            self.assertEqual(tuple(p.shape), (h, 3))
            # Displacement plus two horizon-centered profiles.
            torch.testing.assert_close(p[:, 1:].mean(0), torch.zeros(2), atol=1e-6, rtol=0)
            torch.testing.assert_close(p.T @ p / h, torch.eye(3), atol=1e-5, rtol=0)
            torch.testing.assert_close(p[:, 0], torch.ones(h), atol=0, rtol=0)
            head = ProfileLevelHead(720, h, pool_factor=24, rank=3)
            self.assertEqual(sum(q.numel() for q in head.parameters()), 30 * 3 + 3 + 3 * 3 + 3)
            x = torch.randn(2, 720, 5)
            torch.testing.assert_close(head(x), x[:, -1:].expand(-1, h, -1), atol=0, rtol=0)

    def test_paired_initialization_and_parameter_budget(self):
        x = torch.randn(3, 720, 7)
        for head in HEADS:
            for shared in (False, True):
                m1, b0, *_ = paired_models(e14_source(), 'm1', head, shared)
                m1.eval(), b0.eval()
                with torch.no_grad():
                    torch.testing.assert_close(forward(m1, x), forward(b0, x), atol=0, rtol=0)
                counts = parameter_breakdown(m1)
                self.assertEqual(counts['modulation'], 2 * (3 + m1.latent_dim))
                self.assertEqual(counts['fusion'], 2 if shared else 2 * 7)
                expected_branch = {'S1': 31 * 3 + 4 * 96, 'S2': 105, 'S3': 91 * 3 + 4 * 96}[head]
                self.assertEqual(counts['level_branch'], expected_branch)

    def test_shared_gate_forward_on_many_channels(self):
        m1, _, _, cfg, _ = paired_models(e14_source('Traffic', 96, batch=2), 'm1', 'S2', True)
        self.assertEqual(parameter_breakdown(m1)['fusion'], 2)
        with torch.no_grad():
            y = forward(m1.eval(), torch.randn(2, 720, 862))
        self.assertEqual(tuple(y.shape), (2, 96, 862))
        self.assertTrue(cfg['vldbj_m1']['shared_gate'])

    def test_nonzero_modulation_keeps_phase_cycle_means(self):
        m1, b0, *_ = paired_models(e14_source(), 'm1', 'S2', False)
        torch.nn.init.normal_(m1.lowrank_interaction_branch.communication.output.weight)
        torch.nn.init.normal_(m1.lowrank_interaction_branch.lowrank_head.mix.weight)
        b0.load_state_dict({k: v for k, v in m1.state_dict().items() if k in b0.state_dict()})
        captured = {}
        for name, model in (('m1', m1), ('b0', b0)):
            model.predictor.register_forward_hook(lambda mod, inp, out, n=name: captured.__setitem__(n, out))
        x = torch.randn(2, 720, 7).double()
        with torch.no_grad():
            forward(m1.double().eval(), x)
            forward(b0.double().eval(), x)
        # Predictor output is (B, C, phase slots, future cycles): the means over
        # slots, i.e. every future cycle mean, must coincide.
        self.assertGreater((captured['m1'] - captured['b0']).abs().max(), 1e-6)
        torch.testing.assert_close(captured['m1'].mean(2), captured['b0'].mean(2), atol=1e-10, rtol=0)

    def test_rejects_non_phase_only_source(self):
        bad = e14_source()
        bad['mechanism'] = 'weak_residual'
        with self.assertRaises(ValueError):
            prepare_config(bad, 'm1', 'S2', False)

    def test_default_variant_matches_main_protocol(self):
        cfg = prepare_config(e14_source(), 'm1', 'S2', False)
        self.assertEqual(cfg['vldbj_m1']['communication_width'], 2)
        self.assertEqual(cfg['vldbj_m1']['extension_lr_mult'], 1.0)
        self.assertEqual(cfg['vldbj_m1']['gate_init'], 0.2)
        self.assertEqual(cfg['hyperparams']['weak_period_residual_gate_init'], 0.2)

    def test_variant_width_and_gate_reinit(self):
        source = e14_source()
        variant = {'communication_width': 4, 'extension_lr_mult': 10.0, 'gate_init': 0.5}
        m1, b0, *_ = paired_models(source, 'm1', 'S2', False, variant)
        # Width 4 clamps to rank 3: source 3x3 plus output 3xd.
        width = min(4, 3, m1.latent_dim)
        self.assertEqual(parameter_breakdown(m1)['modulation'], 3 * width + width * m1.latent_dim)
        torch.testing.assert_close(torch.sigmoid(m1.lowrank_interaction_gate),
                                   torch.full_like(m1.lowrank_interaction_gate, 0.5), atol=1e-6, rtol=0)
        # Everything except the two fusion logits is still the same-seed b0 parameter.
        state = m1.state_dict()
        for name, value in b0.state_dict().items():
            if 'shape_bias_delta' in name:
                continue
            torch.testing.assert_close(state[name], value, atol=0, rtol=0)

    def test_variant_rejects_unregistered_fields(self):
        with self.assertRaises(ValueError):
            prepare_config(e14_source(), 'm1', 'S2', False, {'communication_width': 5})
        with self.assertRaises(ValueError):
            prepare_config(e14_source(), 'm1', 'S2', False, {'bogus_knob': 1})

    def test_extension_lr_multiplier_groups(self):
        from scripts.vldbj_m1.run import _attach_extension_lr_multiplier
        m1, *_ = paired_models(e14_source(), 'm1', 'S2', False,
                                {'communication_width': 2, 'extension_lr_mult': 10.0, 'gate_init': 0.2})
        _attach_extension_lr_multiplier(m1, 10.0)
        result = m1.configure_optimizers()
        optimizer = result[0][0] if isinstance(result, (list, tuple)) else result
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[1]['lr'], 10.0 * optimizer.param_groups[0]['lr'])
        self.assertEqual(sum(p.numel() for g in optimizer.param_groups for p in g['params']),
                         sum(p.numel() for p in m1.parameters()))
        extension_names = {n for n, _ in m1.named_parameters()
                           if n.startswith(('lowrank_interaction_branch.', 'lowrank_interaction_gate',
                                           'lowrank_dynamic_fusion.'))}
        grouped = {n for g in optimizer.param_groups for p in g['params']
                   for n, q in m1.named_parameters() if q is p}
        self.assertEqual(grouped, set(n for n, _ in m1.named_parameters()))
        self.assertGreater(len(extension_names), 0)


if __name__ == '__main__':
    unittest.main()
