"""Checks for the centered modulation hypothesis and paired experiment protocol."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from scripts.centered_mod_interaction.model import (
    ARMS, Communication, construct, prepare_config, random_models, state_hash)
from scripts.lowrank_token_interaction.model import TokenCommunication
from src.models.phaseformer_presets import build_hyperparams


def source_config():
    hp=build_hyperparams('ETTh1',96,'original')
    hp.update(use_lowrank_interaction=True,trend_interaction_mode='none',
        weak_period_residual_rank=12,weak_period_residual_pool_factor=1,
        weak_period_residual_smooth_ratio=0,weak_period_residual_head_type='pooled_lowrank',
        weak_period_residual_gate_init=.2,period_len=24,loss_func='huber',use_huber_loss=True)
    return dict(dataset='ETTh1',lookback=720,horizon=96,hyperparams=hp,
        batch_size=2,percent=100,period=24,seed=2021,loss='huber',init_checkpoint='',
        evaluate_test=False,input_hypothesis='none',input_variant='full',
        intervention_seed=9102,input_d1_period=0.,input_d1_sigma=0.,
        input_d2_recent_length=0,input_d3_component='')


class CenteredModTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')

    def test_formulas_dimensions_counts_and_nontrivial_zero_mean(self):
        for r,d in [(4,8),(8,4),(1,2)]:
            t,z = torch.randn(3,2,r).double(),torch.randn(3,2,6,d).double()
            for arm in ARMS[1:]:
                module = Communication(arm,r,d).double()
                torch.nn.init.normal_(module.output.weight)
                q = module.output(module.source(t)).unsqueeze(-2)
                expected = {'tp_add':q.expand_as(z), 'tp_mod':z*q,
                            'tp_mod_centered':(z-z.mean(-2,keepdim=True))*q}[arm]
                nt,nz,delta = module(t,z)
                self.assertIs(nt,t)
                torch.testing.assert_close(delta,expected,atol=0,rtol=0)
                torch.testing.assert_close(nz,z+delta,atol=0,rtol=0)
                self.assertEqual(sum(p.numel() for p in module.parameters()),min(2,r,d)*(r+d))
                if arm=='tp_mod_centered':
                    self.assertGreater(delta.abs().sum(),0)
                    torch.testing.assert_close(delta.sum(-2),torch.zeros(3,2,d,dtype=torch.double),atol=1e-12,rtol=0)
                    # The constraint also survives a shared affine phase predictor.
                    predictor=torch.nn.Linear(d,5).double()
                    change=predictor(nz)-predictor(z)
                    torch.testing.assert_close(change.mean(-2),torch.zeros(3,2,5,dtype=torch.double),atol=1e-12,rtol=0)

    def test_existing_tp_add_and_mod_are_exact_controls(self):
        for arm in ('tp_add','tp_mod'):
            old,new=TokenCommunication(arm,4,8),Communication(arm,4,8)
            torch.nn.init.normal_(old.output.weight)
            new.load_state_dict(old.state_dict(),strict=True)
            t,z=torch.randn(3,2,4),torch.randn(3,2,6,8)
            for a,b in zip(old(t,z),new(t,z)):
                torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_centered_shift_invariance_permutation_equivariance_and_both_sources(self):
        module=Communication('tp_mod_centered',4,8).double()
        torch.nn.init.normal_(module.output.weight)
        t,z=torch.randn(3,2,4).double().requires_grad_(),torch.randn(3,2,6,8).double().requires_grad_()
        delta=module(t,z)[2]
        torch.testing.assert_close(module(t,z+3)[2],delta,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(module(t,z.flip(-2))[2],delta.flip(-2))
        gt,gz=torch.autograd.grad(delta.square().mean(),(t,z))
        self.assertGreater(gt.abs().sum(),0)
        self.assertGreater(gz.abs().sum(),0)
        torch.testing.assert_close(module(t,torch.ones_like(z))[2],torch.zeros_like(z),atol=0,rtol=0)

    def test_gradient_startup_and_all_parameters_trainable(self):
        for arm in ARMS[1:]:
            module=Communication(arm,4,8)
            t,z=torch.randn(3,2,4),torch.randn(3,2,6,8)
            torch.testing.assert_close(module(t,z)[1],z,atol=0,rtol=0)
            target=torch.randn_like(z)
            opt=torch.optim.SGD(module.parameters(),lr=.05)
            seen={n:False for n,_ in module.named_parameters()}
            for step in range(5):
                opt.zero_grad()
                (module(t,z)[1]-target).square().mean().backward()
                for name,p in module.named_parameters():
                    self.assertTrue(p.requires_grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    seen[name]|=bool(p.grad.abs().sum()>0)
                if step==0:
                    self.assertEqual(module.source.weight.grad.abs().sum(),0)
                opt.step()
            self.assertTrue(all(seen.values()),(arm,seen))

    def test_full_model_initial_identity_restore_off_and_source_immutability(self):
        source=source_config();saved=copy.deepcopy(source);hashes=[]
        x=torch.randn(2,720,7)
        for arm in ARMS:
            model,base,_,cfg,common=random_models(source,arm,1)
            hashes.append(common);model.eval();base.eval()
            with torch.no_grad():torch.testing.assert_close(model(x)[0],base(x)[0],atol=0,rtol=0)
            if arm!='b0':torch.nn.init.normal_(model.lowrank_interaction_branch.communication.output.weight,std=.1)
            restored,_=construct(cfg);restored.load_state_dict(model.state_dict(),strict=True);restored.eval()
            with torch.no_grad():torch.testing.assert_close(restored(x)[0],model(x)[0],atol=0,rtol=0)
            gate=model.lowrank_dynamic_fusion.last_gate
            torch.testing.assert_close(gate,gate[:1].expand_as(gate),atol=0,rtol=0)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            if arm!='b0':
                model.lowrank_interaction_branch.intervention='off'
                with torch.no_grad():torch.testing.assert_close(model(x)[0],base(x)[0],atol=0,rtol=0)
        self.assertEqual(len(set(hashes)),1)
        self.assertEqual(source,saved)

    def test_full_output_cycle_mean_with_revin_unequal_static_gates(self):
        from scripts.centered_mod_interaction.analyze import check_cycle_mean
        model,_,_,_,_=random_models(source_config(),'tp_mod_centered',1)
        model.eval()
        torch.nn.init.normal_(model.lowrank_interaction_branch.communication.output.weight,std=.5)
        with torch.no_grad():model.lowrank_dynamic_fusion.shape_bias_delta.fill_(.8)
        x=torch.randn(8,720,7)
        batch=[x,torch.randn(8,96,7),torch.randn(8,720,4),torch.randn(8,96,4)]
        torch.set_float32_matmul_precision('medium')
        check=check_cycle_mean(model,batch,'cpu')
        self.assertTrue(check['passed'])
        self.assertEqual(torch.get_float32_matmul_precision(),'medium')
        torch.set_float32_matmul_precision('highest')
        model.lowrank_interaction_branch.intervention='native'
        with torch.no_grad():native=model(x)[0]
        model.lowrank_interaction_branch.intervention='off'
        with torch.no_grad():off=model(x)[0]
        self.assertGreater((native-off).abs().sum(),0)

    def test_full_joint_training_message_and_original_branches_receive_gradients(self):
        model,_,_,_,_=random_models(source_config(),'tp_mod_centered',1)
        model.train()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        x,target=torch.randn(2,720,7),torch.randn(2,96,7)
        seen={}
        for _ in range(5):
            opt.zero_grad()
            torch.nn.functional.huber_loss(model(x)[0],target,delta=.3).backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(),name)
                    seen[name]=seen.get(name,False) or bool(p.grad.abs().sum()>0)
            opt.step()
        for prefix in ('lowrank_interaction_branch.communication.',
                       'lowrank_interaction_branch.lowrank_head.',
                       'embedding.','predictor.','lowrank_dynamic_fusion.'):
            names=[n for n,_ in model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(names,prefix)
            self.assertTrue(all(seen.get(n,False) for n in names),(prefix,seen))

    def test_reject_unregistered_protocols(self):
        cfg=prepare_config(source_config(),'tp_mod_centered',1)
        for field,value in [('init_checkpoint','pretrained.ckpt'),('evaluate_test',True)]:
            bad=copy.deepcopy(cfg);bad[field]=value
            with self.assertRaises(ValueError):construct(bad)
        cfg['token_communication']['width']=8
        with self.assertRaises(ValueError):construct(cfg)
        for dims in [(0,8,2),(4,0,2),(4,8,0)]:
            with self.assertRaises(ValueError):Communication('tp_mod_centered',*dims)
        with self.assertRaises(ValueError):prepare_config(source_config(),'phase_only',1)

    def test_benchmark_cache_release_and_arithmetic(self):
        from scripts.centered_mod_interaction.analyze import release_diagnostic_caches,communication_ops
        model,_,_,_,_=random_models(source_config(),'tp_mod_centered',1);model.eval()
        x=torch.randn(2,720,7)
        with torch.no_grad():expected=model(x)[0]
        state=state_hash(model.state_dict());release_diagnostic_caches(model)
        self.assertIsNone(model.last_phase_forecast)
        self.assertEqual(state_hash(model.state_dict()),state)
        with torch.no_grad():torch.testing.assert_close(model(x)[0],expected,atol=0,rtol=0)
        # r=4,d=8,k=2,P=6: projection=14+24, centered update=40+8+144.
        self.assertEqual(communication_ops('tp_mod_centered',4,8,6,2),460)
        self.assertEqual(communication_ops('tp_mod',4,8,6),134)

    def test_rubric_requires_both_controls_and_negative_setting_protection(self):
        from scripts.centered_mod_interaction.analyze import pilot_decision
        rows=[]
        for dataset in ('ETTh1','ETTm2','Weather'):
            for seed in (2021,2022):
                for arm in ARMS:
                    mse=.98 if arm=='tp_mod_centered' else 1.
                    row=dict(setting=f'{dataset}_{seed}',dataset=dataset,config_id=arm,mse=mse,mae=mse,
                        mean_candidate_shape_mse=mse*.2,parameter_count=10000,
                        candidate_mae_p95=mse,candidate_mae_top1_mean=mse,
                        added_compute_ratio_upper_bound=.02,epoch_train_median_seconds=1.,
                        cycle_mean_check={'passed':True} if arm=='tp_mod_centered' else None)
                    for tag in ('batch1','native_batch'):
                        row.update({tag+'_latency_vs_b0':1.,tag+'_p95_vs_b0':1.,tag+'_peak_vs_b0':1.})
                    rows.append(row)
        table=pd.DataFrame(rows)
        self.assertTrue(pilot_decision(table)['advance_to_controls'])
        table.loc[(table.config_id=='tp_mod_centered')&(table.dataset=='ETTm2'),'mse']=1.001
        self.assertFalse(pilot_decision(table)['checks']['b0_ettm2'])
        self.assertFalse(pilot_decision(table)['advance_to_controls'])

    def test_manifest_settings_and_no_source_checkpoint_reuse(self):
        from scripts.centered_mod_interaction.run import DEFAULT_SETTINGS,sources_from_manifest
        cfg=source_config();cfg.update(horizon=720,checkpoint='old-success.ckpt')
        cfg['hyperparams']['weak_period_residual_rank']=90
        sources=[]
        for spec in DEFAULT_SETTINGS.split(','):
            dataset,horizon,seed=spec.split(':')
            current=copy.deepcopy(cfg);current.update(dataset=dataset,horizon=int(horizon),seed=int(seed))
            sources.append({'setting':f'{dataset}_H{horizon}_S{seed}_val','config':current})
        original=copy.deepcopy(sources)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json'
            path.write_text(json.dumps({'study':'level','smoke':False,'sources':sources}))
            loaded=sources_from_manifest(path,DEFAULT_SETTINGS)
            self.assertEqual(len(loaded),6)
            self.assertTrue(all(s['config']['checkpoint']=='' for s in loaded))
            self.assertTrue(all(s['config']['init_checkpoint']=='' for s in loaded))
            with self.assertRaises(ValueError):sources_from_manifest(path,'ETTh2:720:2021')
        self.assertEqual(sources,original)


if __name__=='__main__':
    unittest.main()
