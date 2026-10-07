import unittest

import torch

from src.models.lowrank_fusion import LowRankDynamicFusion
from src.models.lowrank_interaction import LowRankPhaseInteractionBranch


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.yp = torch.randn(3, 96, 2, requires_grad=True)
        self.yt = torch.randn(3, 96, 2, requires_grad=True)
        self.t = torch.randn(3, 2, 12, requires_grad=True)
        self.p = torch.randn(3, 2, 24, 4, requires_grad=True)
        self.history = torch.randn(3, 2, 24, 30)
        self.bias = torch.full((1, 1, 2), torch.logit(torch.tensor(.2)), requires_grad=True)

    def run_gate(self, module):
        return module(self.yp, self.yt, self.t, self.p, self.history, self.bias)

    def test_all_initializations_equal_static_and_gradients_start(self):
        expected = (1-self.bias.sigmoid())*self.yp + self.bias.sigmoid()*self.yt
        for mode in LowRankDynamicFusion.MODES:
            gate = LowRankDynamicFusion(12,4,2,24,mode)
            result = self.run_gate(gate)
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            grads = torch.autograd.grad(result.square().sum(), list(gate.parameters()), allow_unused=True)
            self.assertTrue(any(g is not None and g.abs().sum()>0 for g in grads))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))

    def test_gate_features_detached_but_both_forecasts_train(self):
        gate = LowRankDynamicFusion(12,4,2,24,'disagreement')
        torch.nn.init.normal_(gate.head[-1].weight)
        grads = torch.autograd.grad(self.run_gate(gate).square().sum(),
                                    [self.yp,self.yt,self.t,self.p], allow_unused=True)
        self.assertTrue(all(g is not None and g.abs().sum()>0 for g in grads[:2]))
        self.assertEqual(grads[2:], (None,None))

    def test_components_reconstruct_and_independently_control_cycle_means(self):
        gate = LowRankDynamicFusion(12,4,2,24,'components')
        g = torch.tensor([.8,.1])[None,None,:].expand(3,2,2)
        gate.override_gate = g
        out = self.run_gate(gate).reshape(3,4,24,2)
        p,t = self.yp.reshape(3,4,24,2),self.yt.reshape(3,4,24,2)
        pm,tm = p.mean(2,keepdim=True),t.mean(2,keepdim=True)
        expected = .2*pm+.8*tm+.9*(p-pm)+.1*(t-tm)
        torch.testing.assert_close(out,expected)
        torch.testing.assert_close(out.mean(2),(.2*pm+.8*tm).squeeze(2))

    def test_reliability_and_matching_capacity(self):
        template = torch.arange(24.).reshape(1,1,24,1).expand(3,2,24,30)
        torch.testing.assert_close(LowRankDynamicFusion.reliability(template),torch.ones(3,2))
        constant = torch.ones_like(template)
        torch.testing.assert_close(LowRankDynamicFusion.reliability(constant),torch.zeros(3,2))
        a=LowRankDynamicFusion(12,4,2,24,'disagreement')
        b=LowRankDynamicFusion(12,4,2,24,'disagreement_without')
        self.assertEqual([p.shape for p in a.parameters()],[p.shape for p in b.parameters()])

    def test_phase_receipt_interventions_preserve_mass(self):
        branch=LowRankPhaseInteractionBranch(720,96,4,pool_factor=1,rank=12,
            interaction_mode='trend_to_phase',update_style='scaled_pre_norm',selective_write=True)
        branch.capture_diagnostics=True
        x=torch.randn(3,720,2)
        branch(x,self.p); original=branch.last_diagnostics['acceptance']
        for mode in ('mean','roll'):
            branch.receipt_intervention=mode
            branch(x,self.p); altered=branch.last_diagnostics['acceptance']
            torch.testing.assert_close(original.mean(-1),altered.mean(-1))
        branch.receipt_intervention='native';branch(x,self.p)
        torch.testing.assert_close(original,branch.last_diagnostics['acceptance'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
