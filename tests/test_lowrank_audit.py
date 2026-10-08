import unittest
from types import SimpleNamespace

import torch

from scripts.analyze_lowrank_six import replay_fusion
from src.models.lowrank_fusion import LowRankDynamicFusion
from src.models.phase_adapters import RevIN


class FusionReplayTests(unittest.TestCase):
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
            for mode in ('disagreement', 'components', 'components_static'):
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
