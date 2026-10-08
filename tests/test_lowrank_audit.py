import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.analyze_lowrank_six import replay_fusion, component_errors, oracle_component_gate, separated_shuffle_shifts, mix, evaluation_context, interaction_prediction
from src.models.lowrank_fusion import LowRankDynamicFusion
from src.models.phase_adapters import RevIN


class FusionReplayTests(unittest.TestCase):
    def test_direction_intervention_restores_mode_on_success_and_failure(self):
        branch=SimpleNamespace(interaction_mode='parallel_bidirectional')
        model=SimpleNamespace(lowrank_interaction_branch=branch)
        with patch('scripts.analyze_lowrank_six.forecast',side_effect=lambda *args:(branch.interaction_mode,None)):
            self.assertEqual(interaction_prediction(model,[],None,'phase_to_trend'),'phase_to_trend')
        self.assertEqual(branch.interaction_mode,'parallel_bidirectional')
        with patch('scripts.analyze_lowrank_six.forecast',side_effect=RuntimeError('inference failure')):
            with self.assertRaises(RuntimeError):interaction_prediction(model,[],None,'none')
        self.assertEqual(branch.interaction_mode,'parallel_bidirectional')

    def test_test_split_resolves_validation_selected_training_run(self):
        source={'setting':'Weather_H96_S2021_test','training_setting':'Weather_H96_S2021_val'}
        runs,setting,split=evaluation_context(Path('/tmp/test_audit'),source,
            {'evaluation_split':'test','training_scratch_dir':'/tmp/training'})
        self.assertEqual((runs,setting,split),(Path('/tmp/training/runs'),'Weather_H96_S2021_val','test'))
        self.assertEqual(evaluation_context(Path('/tmp/original'),{'setting':'val_setting'},{}),
                         (Path('/tmp/original/runs'),'val_setting','val'))
        with self.assertRaises(ValueError):evaluation_context(Path('/tmp/x'),source,{'evaluation_split':'train'})

    def test_oracle_uses_shared_window_gate_and_decomposition_sums_to_mse(self):
        torch.manual_seed(81)
        p,t=torch.randn(4,96,2),torch.randn(4,96,2)
        gates=torch.tensor([.3,.8]).expand(4,2,2)
        truth=mix(p,t,gates,24)
        oracle=oracle_component_gate(p,t,truth,24)
        torch.testing.assert_close(oracle,gates)
        torch.testing.assert_close(mix(p,t,oracle,24),truth)
        level,shape=component_errors(p,truth,24)
        torch.testing.assert_close(level+shape,(p-truth).double().square().mean(1))
        degenerate=oracle_component_gate(p,p,truth,24)
        self.assertTrue(torch.isfinite(degenerate).all())

    def test_shuffle_seeds_never_overlap_forecast_windows(self):
        shifts=separated_shuffle_shifts(1000,96)
        self.assertEqual(len(shifts),5)
        self.assertEqual(shifts,separated_shuffle_shifts(1000,96))
        self.assertTrue(all(min(s,1000-s)>=96 for s in shifts))
        with self.assertRaises(ValueError):separated_shuffle_shifts(191,96)

    def test_replay_and_gate_interventions_match_model_order_exactly(self):
        torch.manual_seed(31)
        device = torch.device('cpu')
        # Dataset batches are float64; forecast casts them before RevIN.
        history = torch.randn(3, 720, 2, dtype=torch.float64) * 200 + 80
        batch = [history]
        p, t = torch.randn(3, 96, 2), torch.randn(3, 96, 2)
        trend, phase = torch.randn(3, 2, 12), torch.randn(3, 2, 24, 4)
        phase_history = torch.randn(3, 2, 24, 30)
        bias = torch.zeros(1, 1, 2)
        for use_revin in (False, True):
            model = SimpleNamespace(use_revin=use_revin, period_len=24, revin=RevIN(2))
            _, stats = model.revin.normalize(history.float())
            for mode in ('disagreement', 'components', 'components_static', 'components_level_dynamic',
                         'components_level_without'):
                fusion = LowRankDynamicFusion(12, 4, 2, 24, mode)
                for intervention in ('native', 'mean', 'shuffle'):
                    gate = torch.rand(3, 2, 2 if fusion.components else 1)
                    if intervention == 'mean':
                        gate = gate.mean(0, keepdim=True).expand_as(gate)
                    elif intervention == 'shuffle':
                        gate = gate.roll(1, 0)
                    fusion.override_gate = gate
                    expected = fusion(p, t, trend, phase, phase_history, bias)
                    if use_revin:
                        expected = model.revin.denormalize(expected, stats)
                    actual = replay_fusion(model, batch, device, gate, {'phase': p, 'trend': t})
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
