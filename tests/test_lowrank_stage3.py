import copy
import unittest

import torch

from src.models.lowrank_interaction import LowRankPhaseInteractionBranch, PhaseContentCorrection


class Stage3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.t = torch.randn(6, 1, 12)
        self.p = torch.randn(6, 24, 4)

    def test_identity_parameter_matching_and_centering(self):
        counts = []
        for mode in ('trend', 'phase', 'centered'):
            addon = PhaseContentCorrection(12, 4, 8, mode)
            counts.append(sum(p.numel() for p in addon.parameters()))
            self.assertTrue(torch.equal(addon(self.t, self.p), torch.zeros_like(self.p)))
            with torch.no_grad(): addon.scale.fill_(1)
            update = addon(self.t, self.p)
            if mode == 'trend':
                torch.testing.assert_close(update, update[:, :1].expand_as(update))
            if mode == 'centered':
                torch.testing.assert_close(update.mean(1), torch.zeros(6, 4), atol=1e-7, rtol=0)
                self.assertGreater(torch.linalg.matrix_rank(update[0]).item(), 1)
        self.assertEqual(len(set(counts)), 1)

    def test_gradient_startup_and_phase_alignment(self):
        for mode in ('trend', 'phase', 'centered'):
            addon = PhaseContentCorrection(12, 4, 8, mode)
            target = torch.randn_like(self.p)
            (addon(self.t, self.p)-target).square().sum().backward()
            self.assertGreater(addon.scale.grad.abs().item(), 0)
            with torch.no_grad(): addon.scale.fill_(.1)
            addon.zero_grad()
            native = addon(self.t, self.p)
            (native-target).square().sum().backward()
            for param in addon.parameters():
                self.assertIsNotNone(param.grad)
                self.assertGreater(param.grad.abs().sum().item(), 0)
            addon.intervention = 'roll'
            rolled = addon(self.t, self.p)
            if mode == 'trend':
                torch.testing.assert_close(native, rolled, rtol=0, atol=0)
            else:
                torch.testing.assert_close(rolled, native.roll(12, 1), atol=1e-7, rtol=1e-5)
                self.assertGreater((rolled-native).abs().max().item(), 0)

    def test_control_pooling_matches_adaptive_definition(self):
        for rank, width in ((12, 4), (90, 8), (4, 8)):
            addon = PhaseContentCorrection(rank, width, 8, 'trend')
            value = torch.randn(3, 1, rank, requires_grad=True)
            explicit = torch.stack([value[..., a:b].mean(-1) for a,b in addon.context_bins], -1)
            reference = torch.nn.functional.adaptive_avg_pool1d(value, width)
            torch.testing.assert_close(explicit, reference)
            expected = torch.autograd.grad(reference.square().sum(), value, retain_graph=True)[0]
            actual = torch.autograd.grad(explicit.square().sum(), value)[0]
            torch.testing.assert_close(actual, expected)

    def test_legacy_branch_identity_and_unchanged_decoder(self):
        base = LowRankPhaseInteractionBranch(48, 16, 4, pool_factor=1, rank=12,
            interaction_mode='trend_to_phase', update_style='pre_norm')
        x, p = torch.randn(2,48,3), self.p.reshape(2,3,24,4)
        for mode in ('trend', 'phase', 'centered'):
            candidate = LowRankPhaseInteractionBranch(48,16,4,pool_factor=1,rank=12,
                interaction_mode='trend_to_phase',update_style='pre_norm',content_mode=mode)
            status=candidate.load_state_dict(base.state_dict(),strict=False)
            self.assertTrue(all(k.startswith('content_correction.') for k in status.missing_keys))
            a, ad=base(x,p); b,bd=candidate(x,p)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            torch.testing.assert_close(ad['updated_phase_latent'],bd['updated_phase_latent'],atol=0,rtol=0)
            with torch.no_grad(): candidate.content_correction.scale.fill_(1)
            b,bd=candidate(x,p)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertGreater((ad['updated_phase_latent']-bd['updated_phase_latent']).abs().max().item(),0)


if __name__ == '__main__':
    unittest.main()
