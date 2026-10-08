"""Mechanism-level checks; runnable with unittest without pytest plugins."""

import copy
import unittest

import torch

from src.models.lowrank_interaction import LowRankPhaseInteractionBranch
from scripts.run_lowrank_interaction_stage2 import ARMS, construct, load_i0
from src.models.phaseformer_presets import build_hyperparams


class LowrankStage2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.x = torch.randn(2, 48, 3)
        self.phase = torch.randn(2, 3, 6, 8)

    def branch(self, mode="phase_to_trend", **kwargs):
        return LowRankPhaseInteractionBranch(48, 16, 8, pool_factor=1, rank=4,
                                            interaction_mode=mode, **kwargs)

    def test_legacy_default_no_new_parameters(self):
        baseline = self.branch(mode="none")
        pred, diag = baseline(self.x, self.phase)
        torch.testing.assert_close(pred, baseline.lowrank_head(self.x), rtol=0, atol=0)
        self.assertTrue(torch.equal(diag['updated_phase_latent'], self.phase))
        self.assertTrue(all(k.startswith('lowrank_head.') for k in baseline.state_dict()))

    def test_parallel_reads_original_states_and_both_directions_train(self):
        branch=self.branch('parallel_bidirectional')
        # The legacy decoder starts at zero; make its rank readout observable.
        torch.nn.init.normal_(branch.lowrank_head.decoder.weight)
        latent,last=branch.lowrank_head.encode_latent(self.x)
        expected_trend,_=branch._phase_to_trend(latent,self.phase)
        expected_phase,_=branch._trend_to_phase(latent,self.phase)
        pred,diag=branch(self.x,self.phase)
        torch.testing.assert_close(pred,branch.lowrank_head.decode_latent(expected_trend,last),atol=0,rtol=0)
        torch.testing.assert_close(diag['updated_phase_latent'],expected_phase,atol=0,rtol=0)
        sequential_phase,_=branch._trend_to_phase(expected_trend,self.phase)
        self.assertGreater(float((expected_phase-sequential_phase).abs().max()),1e-5)
        (pred.square().mean()+diag['updated_phase_latent'].square().mean()).backward()
        for parameter in [branch.trend_out.weight,branch.phase_out.weight,
                          branch.lowrank_head.encoder.weight,branch.lowrank_head.decoder.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0)

    def test_scaled_identity_both_directions(self):
        for mode, flags in [("phase_to_trend", {}), ("trend_to_phase", {}),
                            ("phase_to_trend", {"null_read": True}),
                            ("trend_to_phase", {"selective_write": True})]:
            branch = self.branch(mode, update_style="scaled_pre_norm", **flags)
            pred, diag = branch(self.x, self.phase)
            torch.testing.assert_close(pred, branch.lowrank_head(self.x), rtol=0, atol=0)
            self.assertTrue(torch.equal(diag['updated_phase_latent'], self.phase))

    def test_zero_scale_gradient_starts_then_message_learns(self):
        for mode in ("phase_to_trend", "trend_to_phase"):
            branch = self.branch(mode, update_style="scaled_pre_norm")
            latent, _ = branch.lowrank_head.encode_latent(self.x)
            def objective():
                if mode == 'phase_to_trend':
                    return branch._phase_to_trend(latent.detach(), self.phase)[0].square().sum()
                return branch._trend_to_phase(latent.detach(), self.phase)[0].square().sum()
            objective().backward()
            self.assertGreater(abs(branch.interaction_scale.grad.item()), 0)
            with torch.no_grad():
                branch.interaction_scale.fill_(0.1)
            branch.zero_grad()
            objective().backward()
            proj = branch.trend_out if mode == 'phase_to_trend' else branch.phase_out
            self.assertGreater(proj.weight.grad.abs().sum().item(), 0)

    def test_null_rejects_bias_as_well_as_values(self):
        for mode, flags in [("phase_to_trend", {"null_read": True}),
                            ("trend_to_phase", {"selective_write": True})]:
            branch = self.branch(mode, update_style="scaled_pre_norm", **flags)
            with torch.no_grad():
                branch.interaction_scale.fill_(1)
                branch.null_score.fill_(1000)
                for name, p in branch.named_parameters():
                    if name.endswith('bias') and not name.startswith('lowrank_head'):
                        p.fill_(3)
            pred, diag = branch(self.x, self.phase)
            torch.testing.assert_close(pred, branch.lowrank_head(self.x), rtol=0, atol=0)
            self.assertTrue(torch.equal(diag['updated_phase_latent'], self.phase))

    def test_selective_gate_depends_on_phase_query(self):
        branch = self.branch('trend_to_phase', update_style='scaled_pre_norm', selective_write=True)
        branch.capture_diagnostics = True
        with torch.no_grad():
            branch.interaction_scale.fill_(1)
        _, diag = branch(self.x, self.phase)
        gates = branch.last_diagnostics['acceptance']
        self.assertGreater(gates.std(-1).mean().item(), 0)
        diag['updated_phase_latent'].square().sum().backward()
        self.assertGreater(branch.phase_query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(branch.trend_key.weight.grad.abs().sum().item(), 0)

    def test_dense_single_key_remains_broadcast(self):
        branch = self.branch('trend_to_phase', update_style='pre_norm')
        _, diag = branch(self.x, self.phase)
        delta = diag['updated_phase_latent'] - self.phase
        torch.testing.assert_close(delta, delta[:, :, :1].expand_as(delta))

    def test_explicit_projection_matches_mha(self):
        branch = self.branch(update_style='pre_norm')
        t, _ = branch.lowrank_head.encode_latent(self.x)
        got, _ = branch._phase_to_trend(t, self.phase)
        t0 = t.reshape(6, 1, 4)
        p0 = self.phase.reshape(6, 6, 8)
        output, _ = branch.phase_to_trend_attention(branch.trend_query(branch.trend_norm(t0)),
            branch.phase_key(branch.phase_norm(p0)), branch.phase_value(branch.phase_norm(p0)))
        expected = t0 + branch.trend_out(output)
        torch.testing.assert_close(got, expected.reshape_as(got))

    def test_full_config_plumbing_and_initial_output(self):
        hp = build_hyperparams('ETTh1', 96, 'original')
        hp.update(use_lowrank_interaction=True, trend_interaction_mode='none',
                  weak_period_residual_rank=12, weak_period_residual_pool_factor=1,
                  weak_period_residual_head_type='pooled_lowrank', weak_period_residual_gate_init=0.2)
        cfg = {'dataset': 'ETTh1', 'lookback': 720, 'horizon': 96, 'hyperparams': hp,
               'batch_size': 2, 'percent': 100, 'period': 24, 'input_hypothesis': 'none',
               'input_variant': 'full', 'intervention_seed': 9102, 'input_d1_period': 0.,
               'input_d1_sigma': 0., 'input_d2_recent_length': 0, 'input_d3_component': ''}
        base, _ = construct(cfg, workers=0)
        base.eval()
        x = torch.randn(2, 720, 7)
        with torch.no_grad():
            ref = base(x, None, None, None)[0]
        for arm in ('pt_scaled', 'tp_scaled', 'pt_null', 'tp_selective'):
            candidate = copy.deepcopy(cfg)
            candidate['hyperparams'].update(ARMS[arm])
            model, _ = construct(candidate, workers=0)
            load_i0(model, base.state_dict())
            model.eval()
            self.assertEqual(model.lowrank_interaction_branch.rank, 12)
            self.assertFalse(model.use_trend_token)
            with torch.no_grad():
                got = model(x, None, None, None)[0]
            torch.testing.assert_close(got, ref, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
