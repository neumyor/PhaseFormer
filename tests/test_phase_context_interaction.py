"""Mechanism, equivalence, gradient and experiment-isolation checks."""
import copy
import unittest

import torch

from scripts.phase_context_interaction.model import (
    ARMS, Communication, construct, prepare_config, random_models)
from src.models.phaseformer_presets import build_hyperparams


def source_config():
    hp = build_hyperparams('ETTh1', 96, 'original')
    hp.update(use_lowrank_interaction=True, trend_interaction_mode='none',
              weak_period_residual_rank=12, weak_period_residual_pool_factor=1,
              weak_period_residual_smooth_ratio=0, weak_period_residual_head_type='pooled_lowrank',
              weak_period_residual_gate_init=0.2, period_len=24, loss_func='huber', use_huber_loss=True)
    return dict(dataset='ETTh1', lookback=720, horizon=96, hyperparams=hp,
                batch_size=2, percent=100, period=24, seed=2021, loss='huber',
                init_checkpoint='', evaluate_test=False, input_hypothesis='none',
                input_variant='full', intervention_seed=9102, input_d1_period=0.,
                input_d1_sigma=0., input_d2_recent_length=0, input_d3_component='')


class PhaseContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_equations_dimensions_parameter_counts_and_slot_invariance(self):
        for r, d in [(4, 8), (8, 4), (1, 2)]:
            t, z = torch.randn(3, 2, r), torch.randn(3, 2, 6, d)
            k = min(2, r, d)
            for arm in ARMS[1:] + ('untied_write',):
                module = Communication(arm, r, d)
                nnames = dict(module.named_parameters())
                torch.nn.init.normal_(module.output.weight)
                if arm == 'untied_write':
                    torch.nn.init.normal_(module.context_output.weight)
                nt, nz, delta = module(t, z)
                if arm == 'tp_add':
                    expected = module.output(module.source(t))
                    count = k * (r+d)
                elif arm == 'phase_only':
                    expected = -module.output(module.context(z.mean(-2)))
                    count = 2*k*d
                elif arm == 'context_full':
                    expected = module.output(module.source(t)-module.context(z.mean(-2)))
                    count = k*(r+2*d)
                else:
                    expected = module.output(module.source(t))-module.context_output(module.context(z.mean(-2)))
                    count = k*(r+3*d)
                self.assertIs(nt, t)
                torch.testing.assert_close(nz, z+expected.unsqueeze(-2))
                torch.testing.assert_close(delta, expected.unsqueeze(-2).expand_as(z))
                torch.testing.assert_close(nz[..., 1:, :]-nz[..., :-1, :], z[..., 1:, :]-z[..., :-1, :])
                torch.testing.assert_close(module(t, z.flip(-2))[2], delta)
                self.assertEqual(sum(p.numel() for p in nnames.values()), count)

    def test_context_zero_equals_tp_add_outputs_and_active_gradients(self):
        a, b = Communication('tp_add', 4, 8).double(), Communication('context_zero', 4, 8).double()
        torch.nn.init.normal_(a.output.weight)
        b.source.load_state_dict(a.source.state_dict())
        b.output.load_state_dict(a.output.state_dict())
        t, z = torch.randn(3, 2, 4).double(), torch.randn(3, 2, 6, 8).double()
        torch.testing.assert_close(a(t,z)[1], b(t,z)[1], atol=0, rtol=0)
        a(t,z)[1].square().mean().backward()
        b(t,z)[1].square().mean().backward()
        for name in ('source.weight', 'output.weight'):
            torch.testing.assert_close(dict(a.named_parameters())[name].grad,
                                       dict(b.named_parameters())[name].grad, atol=0, rtol=0)
        self.assertEqual(b.context.weight.grad.abs().sum().item(), 0)

    def test_all_active_parameters_start_and_full_reads_both_sources(self):
        for arm in ARMS[1:] + ('untied_write',):
            module = Communication(arm, 4, 8)
            t, z = torch.randn(3, 2, 4), torch.randn(3, 2, 6, 8)
            target = torch.randn_like(z)
            opt = torch.optim.SGD(module.parameters(), lr=.05)
            seen = {n:False for n,_ in module.named_parameters()}
            for _ in range(5):
                opt.zero_grad()
                (module(t,z)[1]-target).square().mean().backward()
                for name,p in module.named_parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    seen[name] |= bool(p.grad.abs().sum()>0)
                opt.step()
            self.assertTrue(all(seen.values()), (arm,seen))
        module = Communication('context_full', 4, 8)
        torch.nn.init.normal_(module.output.weight)
        t, z = t.requires_grad_(), z.requires_grad_()
        gt,gz = torch.autograd.grad(module(t,z)[2].square().mean(), (t,z))
        self.assertGreater(gt.abs().sum(),0)
        self.assertGreater(gz.abs().sum(),0)

    def test_shared_joint_rank_and_linear_sign_equivalence(self):
        module = Communication('context_full',4,8).double()
        torch.nn.init.normal_(module.output.weight)
        joint = torch.cat((module.output.weight@module.source.weight,
                           -module.output.weight@module.context.weight),dim=1)
        self.assertLessEqual(torch.linalg.matrix_rank(joint).item(),2)
        t,z = torch.randn(3,2,4).double(),torch.randn(3,2,6,8).double()
        write = module.output(module.source(t)+torch.nn.functional.linear(z.mean(-2),-module.context.weight))
        torch.testing.assert_close(module(t,z)[2],write.unsqueeze(-2).expand_as(z))

    def test_full_model_initial_identity_restore_static_gate_and_source_immutability(self):
        source = source_config()
        saved = copy.deepcopy(source)
        hashes=[]
        x=torch.randn(2,720,7)
        for arm in ARMS:
            model,base,_,cfg,common=random_models(source,arm,1)
            hashes.append(common)
            model.eval();base.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x)[0],base(x)[0],atol=0,rtol=0)
            if arm!='b0':
                torch.nn.init.normal_(model.lowrank_interaction_branch.communication.output.weight,std=.1)
            restored,_=construct(cfg)
            restored.load_state_dict(model.state_dict(),strict=True)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(x)[0],model(x)[0],atol=0,rtol=0)
            gate=model.lowrank_dynamic_fusion.last_gate
            torch.testing.assert_close(gate,gate[:1].expand_as(gate),atol=0,rtol=0)
            if arm!='b0':
                model.lowrank_interaction_branch.intervention='off'
                with torch.no_grad():
                    torch.testing.assert_close(model(x)[0],base(x)[0],atol=0,rtol=0)
        self.assertEqual(len(set(hashes)),1)
        self.assertEqual(source,saved)

    def test_reject_checkpoint_test_and_changed_width(self):
        cfg=prepare_config(source_config(),'context_full',1)
        for field,value in [('init_checkpoint','pretrained.ckpt'),('evaluate_test',True)]:
            bad=copy.deepcopy(cfg);bad[field]=value
            with self.assertRaises(ValueError):construct(bad)
        cfg['token_communication']['width']=8
        with self.assertRaises(ValueError):construct(cfg)

    def test_benchmark_cache_release_preserves_parameters_and_forward(self):
        from scripts.phase_context_interaction.analyze import release_diagnostic_caches
        from scripts.phase_context_interaction.model import state_hash
        model,_,_,_,_=random_models(source_config(),'context_full',1)
        model.eval()
        x=torch.randn(2,720,7)
        with torch.no_grad():expected=model(x)[0]
        state=state_hash(model.state_dict())
        self.assertIsNotNone(model.last_phase_forecast)
        release_diagnostic_caches(model)
        self.assertIsNone(model.last_phase_forecast)
        self.assertEqual(state_hash(model.state_dict()),state)
        with torch.no_grad():torch.testing.assert_close(model(x)[0],expected,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
