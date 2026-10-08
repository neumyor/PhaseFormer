"""Information-flow and protocol checks for the isolated communication study."""
import copy
import json
from pathlib import Path
import tempfile
import signal
import sys
import threading
import time
import unittest
from unittest.mock import patch

import torch

from scripts.lowrank_token_interaction.model import (
    ARMS, TokenCommunication, IsolatedBranch, construct, prepare_config, random_models, validate_protocol)
from scripts.lowrank_token_interaction.run import (
    sources_from_manifest, write_immutable, run_child, STOP, CHILDREN, CHILDREN_LOCK, request_stop)
from scripts.lowrank_token_interaction.analyze import off_prediction
from scripts.run_lowrank_interaction_stage2 import construct as legacy_construct
from src.models.lowrank_interaction import LowRankPhaseInteractionBranch
from src.models.phaseformer_presets import build_hyperparams


def source_config():
    hp = build_hyperparams('ETTh1', 96, 'original')
    hp.update(use_lowrank_interaction=True, trend_interaction_mode='none',
        weak_period_residual_rank=12, weak_period_residual_pool_factor=1,
        weak_period_residual_smooth_ratio=0, weak_period_residual_head_type='pooled_lowrank',
        weak_period_residual_gate_init=0.2, period_len=24, loss_func='huber', use_huber_loss=True)
    return {'dataset': 'ETTh1', 'lookback': 720, 'horizon': 96, 'hyperparams': hp,
        'batch_size': 2, 'percent': 100, 'period': 24, 'seed': 2021, 'loss': 'huber',
        'init_checkpoint': '', 'evaluate_test': False, 'input_hypothesis': 'none',
        'input_variant': 'full', 'intervention_seed': 9102, 'input_d1_period': 0.,
        'input_d1_sigma': 0., 'input_d2_recent_length': 0, 'input_d3_component': ''}


class CommunicationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)
        self.t = torch.randn(3, 2, 4)
        self.z = torch.randn(3, 2, 6, 8)

    def test_equations_and_direction_for_both_dimension_orderings(self):
        for rank, dim in [(4, 8), (8, 4), (1, 2)]:
            t, z = torch.randn(3, 2, rank), torch.randn(3, 2, 6, dim)
            for arm in ARMS[1:]:
                module = TokenCommunication(arm, rank, dim)
                self.assertEqual(module.width, min(2, rank, dim))
                torch.nn.init.normal_(module.output.weight)
                got_t, got_z, delta = module(t, z)
                if arm.startswith('pt_'):
                    message = module.source(z.mean(-2))
                    if arm == 'pt_correct':
                        message = message - module.state(t)
                    torch.testing.assert_close(got_t, t + module.output(message))
                    self.assertIs(got_z, z)
                    # Pooled read is invariant to reordering phase slots.
                    torch.testing.assert_close(module(t, z.flip(-2))[0], got_t)
                else:
                    message = module.output(module.source(t)).unsqueeze(-2)
                    torch.testing.assert_close(got_z, z + (message if arm == 'tp_add' else z * message))
                    self.assertIs(got_t, t)
                self.assertTrue(torch.isfinite(delta).all())

    def test_zero_exit_and_multi_step_gradient_startup_with_zero_decoder(self):
        x, target = torch.randn(3, 48, 2), torch.randn(3, 24, 2)
        for arm in ARMS[1:]:
            legacy = LowRankPhaseInteractionBranch(48, 24, 8, pool_factor=1, rank=4, interaction_mode='none')
            module = IsolatedBranch(legacy, 8, arm)
            pred, diag = module(x, self.z)
            torch.testing.assert_close(pred, legacy.lowrank_head(x), atol=0, rtol=0)
            torch.testing.assert_close(diag['updated_phase_latent'], self.z, atol=0, rtol=0)
            optimizer = torch.optim.SGD(module.parameters(), lr=.02)
            observed = {name: False for name, _ in module.named_parameters() if name.startswith('communication.')}
            for _ in range(5):
                optimizer.zero_grad()
                pred, diag = module(x, self.z)
                loss = (pred - target).square().mean() + (diag['updated_phase_latent'] - 1).square().mean()
                loss.backward()
                for name, p in module.named_parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all())
                        if name in observed and p.grad.abs().sum() > 0:
                            observed[name] = True
                optimizer.step()
            self.assertTrue(all(observed.values()), (arm, observed))

    def test_source_inputs_receive_gradients_and_off_preserves_both_states(self):
        for arm in ARMS[1:]:
            t = self.t.clone().requires_grad_()
            z = self.z.clone().requires_grad_()
            m = TokenCommunication(arm, 4, 8)
            torch.nn.init.normal_(m.output.weight)
            nt, nz, _ = m(t, z)
            loss = nt.square().mean() if arm.startswith('pt_') else nz.square().mean()
            gt, gz = torch.autograd.grad(loss, (t, z), allow_unused=True)
            self.assertTrue(gt.abs().sum() > 0)
            self.assertTrue(gz.abs().sum() > 0)

    def test_isolated_full_model_identity_restore_and_no_legacy_mutation(self):
        source = source_config()
        original = copy.deepcopy(source)
        x = torch.randn(2, 720, 7)
        reference_state = None
        common_hashes = []
        for arm in ARMS:
            model, baseline, _, config, common = random_models(source, arm, 1)
            common_hashes.append(common)
            model.eval()
            baseline.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x)[0], baseline(x)[0], atol=0, rtol=0)
            reference_state = baseline.state_dict()
            if arm != 'b0':
                torch.nn.init.normal_(model.lowrank_interaction_branch.communication.output.weight, std=.1)
            restored, _ = construct(config)
            restored.load_state_dict(model.state_dict(), strict=True)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(x)[0], model(x)[0], atol=0, rtol=0)
            # Output gates remain independent of the sample.
            gates = model.lowrank_dynamic_fusion.last_gate
            torch.testing.assert_close(gates, gates[:1].expand_as(gates), atol=0, rtol=0)
            if arm != 'b0':
                branch = model.lowrank_interaction_branch
                branch.intervention = 'off'
                with torch.no_grad():
                    off = model(x)[0]
                torch.testing.assert_close(off, baseline(x)[0], atol=0, rtol=0)
                branch.intervention = 'native'
        self.assertEqual(len(set(common_hashes)), 1)
        self.assertEqual(source, original)
        legacy, _ = legacy_construct(prepare_config(source, 'b0', 1), workers=0)
        self.assertNotIsInstance(legacy.lowrank_interaction_branch, IsolatedBranch)
        self.assertEqual(set(legacy.state_dict()), set(reference_state))

    def test_off_restores_intervention_even_if_forecast_fails(self):
        legacy = LowRankPhaseInteractionBranch(48, 24, 8, pool_factor=1, rank=4, interaction_mode='none')
        branch = IsolatedBranch(legacy, 8, 'pt_add')
        from types import SimpleNamespace
        model = SimpleNamespace(lowrank_interaction_branch=branch)
        with patch('scripts.lowrank_token_interaction.analyze.forecast', side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):
                off_prediction(model, [], 'cpu')
        self.assertEqual(branch.intervention, 'native')

    def test_protocol_rejects_unrelated_features_loss_and_rank_changes(self):
        for key, value in [('use_phase_warp', True), ('use_revin', False),
                           ('weak_residual_asymmetric_component', 'trend'), ('period_len', 12),
                           ('weak_period_residual_rank', 13)]:
            source = source_config()
            source['hyperparams'][key] = value
            with self.assertRaises(ValueError):
                validate_protocol(source)
        source = source_config()
        source['hyperparams'].update(use_huber_loss=False, loss_func='mae')
        with self.assertRaises(ValueError):
            validate_protocol(source)

    def test_manifest_exact_coverage_and_immutable_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'source.json'
            source = {'setting': 'ETTh1_H96_S2021_val', 'config': source_config()}
            value = {'study': 'level', 'smoke': False, 'sources': [source]}
            p.write_text(json.dumps(value))
            self.assertEqual(len(sources_from_manifest(p, 'ETTh1:96:2021')), 1)
            value['sources'].append(source)
            p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                sources_from_manifest(p, 'ETTh1:96:2021,Weather:96:2021')
            q = Path(directory) / 'registered.json'
            write_immutable(q, {'epochs': 30})
            write_immutable(q, {'epochs': 30})
            with self.assertRaises(ValueError):
                write_immutable(q, {'epochs': 8})

    def test_stop_handler_is_lock_free_and_child_is_reaped(self):
        STOP.clear()
        try:
            with CHILDREN_LOCK:
                request_stop(signal.SIGTERM, None)
            self.assertTrue(STOP.is_set())
            STOP.clear()
            timer = threading.Timer(.2, request_stop)
            timer.start()
            start = time.monotonic()
            code = run_child([sys.executable, '-c', 'import time; time.sleep(60)'])
            timer.join()
            self.assertNotEqual(code, 0)
            self.assertLess(time.monotonic() - start, 5)
            self.assertFalse(CHILDREN)
        finally:
            STOP.clear()


if __name__ == '__main__':
    unittest.main()
