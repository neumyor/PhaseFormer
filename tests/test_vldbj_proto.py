"""Checks for experiments 1/2 (conference protocol, warm start) of the VLDBJ extension."""
import ast
import collections
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

from scripts.vldbj_m1 import proto, protocol
from scripts.vldbj_m1.run import TUNE_GRID
from src.models.phaseformer_presets import get_base_hyperparams


def e14_source(dataset='ETTh1', horizon=96, seed=2021, batch=4):
    """Phase-only source config in the E14 schema (as in tests/test_vldbj_m1.py)."""
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

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [(d, h) for d in proto.DATASETS for h in proto.HORIZONS]


def conference_script_config(dataset, horizon):
    """Effective config of the conference script, re-derived from git."""
    commit, script, blob = protocol.CONFERENCE_SOURCES[dataset]
    if subprocess.check_output(['git', 'rev-parse', f'{commit}:{script}'], cwd=ROOT, text=True).strip() != blob:
        raise AssertionError(f'{commit}:{script} blob changed')
    source = subprocess.check_output(['git', 'show', f'{commit}:{script}'], cwd=ROOT, text=True)
    module = ast.parse(source)
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'get_best_config_for_horizon')
    namespace = {}
    exec(compile(ast.Module([fn], []), script, 'exec'), namespace)
    best = namespace['get_best_config_for_horizon'](horizon)
    batch = int(next(line.split('=')[1] for line in source.splitlines()
                     if 'exp_args.training_args.batch_size =' in line))
    reads_dict = "best_config['phase_attn_dropout']" in source
    return {
        **{k: best[k] for k in protocol.ARCH_FIELDS},
        'learning_rate': best['learning_rate'], 'batch_size': batch,
        # The d9cb0b3 config class hardcodes these three; ff14c16 reads dropout
        # and positional embedding from the dict. The window is always None.
        'phase_attn_dropout': best['phase_attn_dropout'] if reads_dict else 0.1,
        'phase_use_pos_embed': best['phase_use_pos_embed'] if reads_dict else True,
        'phase_attn_window': None,
        'huber_delta': best.get('huber_delta', 1.0),
        'max_epochs': best.get('train_epochs', 30), 'patience': best.get('patience', 8),
    }


def job(dataset='ETTh1', horizon=96, seed=2021, arm='m1', init='scratch', proto_name='e14', batch=4):
    variant_id = protocol.FINAL_VARIANTS.get((dataset, horizon)) if arm == 'm1' else None
    return {'dataset': dataset, 'horizon': horizon, 'seed': seed, 'arm': arm, 'init': init,
            'protocol': proto_name, 'head': protocol.FINAL_HEAD,
            'shared_gate': protocol.FINAL_SHARED_GATE[dataset], 'variant_id': variant_id,
            'variant': dict(TUNE_GRID[variant_id]) if variant_id is not None else None,
            'source': {'config': e14_source(dataset, horizon, seed, batch)}}


class ConferenceProtocolTests(unittest.TestCase):
    def test_table_matches_conference_scripts(self):
        try:
            subprocess.check_output(['git', 'cat-file', '-e', 'd9cb0b3'], cwd=ROOT)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest('conference commits unavailable')
        for dataset, horizon in SETTINGS:
            expected = conference_script_config(dataset, horizon)
            got = {k: protocol.CONFERENCE[(dataset, horizon)][k] for k in expected}
            self.assertEqual(got, expected, (dataset, horizon))

    def test_conference_source_overrides_training_fields_only(self):
        src = e14_source('Weather', 96, batch=64)
        src['hyperparams']['learning_rate'] = 0.0003
        cfg = protocol.conference_source(src)
        self.assertEqual(cfg['batch_size'], 16)
        self.assertEqual(cfg['hyperparams']['learning_rate'], 0.001)
        self.assertEqual(cfg['protocol_changes']['batch_size'], [64, 16])
        self.assertEqual(cfg['protocol_changes']['learning_rate'], [0.0003, 0.001])
        for field in protocol.ARCH_FIELDS:
            self.assertEqual(cfg['hyperparams'][field], src['hyperparams'][field])
        self.assertEqual(src['batch_size'], 64)  # the source is not mutated
        self.assertNotEqual(cfg['config_hash'], src['config_hash'])

    def test_traffic96_uses_the_effective_attention_options(self):
        cfg = protocol.conference_source(e14_source('Traffic', 96))
        self.assertIsNone(cfg['hyperparams']['phase_attn_window'])
        self.assertTrue(cfg['hyperparams']['phase_use_pos_embed'])
        self.assertIn('phase_attn_window', cfg['protocol_changes'])

    def test_etth1_720_epochs_and_learning_rate(self):
        src = e14_source('ETTh1', 720)
        src['max_epochs'] = 30
        cfg = protocol.conference_source(src)
        self.assertEqual((cfg['max_epochs'], cfg['hyperparams']['patience']), (70, 14))
        self.assertEqual(cfg['hyperparams']['learning_rate'], 0.00015)
        self.assertEqual(cfg['hyperparams']['huber_delta'], 0.3)

    def test_rejects_architecture_mismatch(self):
        src = e14_source('ETTm1', 96)
        src['hyperparams']['latent_dim'] = 16
        with self.assertRaises(ValueError):
            protocol.conference_source(src)

    def test_golden_table(self):
        gold = protocol.golden()
        self.assertEqual(len(gold), 28)
        self.assertEqual(gold[('Weather', 96)], (0.148, 0.195))
        self.assertEqual(gold[('Traffic', 720)], (0.428, 0.270))

    def test_final_configuration(self):
        self.assertEqual(len(protocol.FINAL_VARIANTS), 8)
        self.assertEqual(TUNE_GRID[protocol.FINAL_VARIANTS[('Weather', 96)]],
                         {'communication_width': 4, 'extension_lr_mult': 10.0, 'gate_init': 0.5})


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')

    def test_pf_initialization_is_the_shared_backbone(self):
        for proto_name in ('e14', 'conference'):
            pf, *_ = proto.build_model(job(arm='pf', proto_name=proto_name))
            b0, *_ = proto.build_model(job(arm='b0', proto_name=proto_name))
            b0_state = b0.state_dict()
            for key, value in pf.state_dict().items():
                torch.testing.assert_close(b0_state[key], value, atol=0, rtol=0)
            extra = set(b0_state) - set(pf.state_dict())
            self.assertTrue(extra and all(k.startswith(proto.EXTENSION_PREFIXES) for k in extra))

    def test_warm_start_reproduces_phaseformer(self):
        for dataset, horizon in (('ETTh1', 96), ('Weather', 96), ('ETTm2', 96)):
            pf, *_ = proto.build_model(job(dataset, horizon, arm='pf'))
            with torch.no_grad():   # stand-in for trained weights
                for p in pf.parameters():
                    p.add_(0.05 * torch.randn_like(p))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'best.ckpt'
                torch.save({'state_dict': pf.state_dict()}, path)
                warm = {'checkpoint': str(path), 'sha256': proto.digest(path)}
                m1, _, cfg, _ = proto.build_model(job(dataset, horizon, arm='m1', init='warm'))
                missing = proto.load_warm(m1, warm)
            self.assertTrue(all(k.startswith(proto.EXTENSION_PREFIXES) for k in missing))
            self.assertEqual(cfg['vldbj_m1']['gate_init'], proto.WARM_GATE_INIT)
            # The final variant keeps its width and extension learning rate.
            variant_id = protocol.FINAL_VARIANTS.get((dataset, horizon))
            if variant_id is not None:
                self.assertEqual(cfg['vldbj_m1']['communication_width'],
                                 TUNE_GRID[variant_id]['communication_width'])
            torch.testing.assert_close(torch.sigmoid(m1.lowrank_interaction_gate),
                                       torch.full_like(m1.lowrank_interaction_gate, proto.WARM_GATE_INIT),
                                       atol=1e-7, rtol=0)
            channels = pf.enc_in if hasattr(pf, 'enc_in') else m1.lowrank_interaction_gate.shape[-1]
            x = torch.randn(2, 720, {'ETTh1': 7, 'ETTm2': 7, 'Weather': 21}[dataset])
            with torch.no_grad():
                y_pf, y_m1 = forward(pf.eval(), x), forward(m1.eval(), x)
            # Level and shape gates at 1e-3 with q = 0: the forecast moves by
            # 1e-3 of the (anchor - phase) difference.
            gap = (y_m1 - y_pf).abs().max() / y_pf.abs().max()
            self.assertLess(gap.item(), 5e-3, (dataset, horizon, channels))

    def test_warm_start_refuses_changed_or_foreign_checkpoints(self):
        pf, *_ = proto.build_model(job(arm='pf'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.ckpt'
            torch.save({'state_dict': pf.state_dict()}, path)
            m1, *_ = proto.build_model(job(arm='m1', init='warm'))
            with self.assertRaises(ValueError):
                proto.load_warm(m1, {'checkpoint': str(path), 'sha256': 'stale'})
            state = dict(pf.state_dict())
            state.pop(next(k for k in state if not k.startswith(proto.EXTENSION_PREFIXES)))
            torch.save({'state_dict': state}, path)
            with self.assertRaises(ValueError):
                proto.load_warm(m1, {'checkpoint': str(path), 'sha256': proto.digest(path)})

    def test_lr_groups(self):
        m1, *_ = proto.build_model(job('Weather', 96, arm='m1'))
        proto.attach_lr_groups(m1, 0.1, 10.0)
        optimizer = m1.configure_optimizers()
        optimizer = optimizer[0][0] if isinstance(optimizer, (list, tuple)) else optimizer
        base = float(m1.args.training_args.learning_rate)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [0.1 * base, 10.0 * base])
        self.assertEqual(sum(p.numel() for g in optimizer.param_groups for p in g['params']),
                         sum(p.numel() for p in m1.parameters()))
        pf, *_ = proto.build_model(job(arm='pf'))
        proto.attach_lr_groups(pf, 0.5, 1.0)
        optimizer = pf.configure_optimizers()
        optimizer = optimizer[0][0] if isinstance(optimizer, (list, tuple)) else optimizer
        self.assertEqual(len(optimizer.param_groups), 1)


class PipelineTests(unittest.TestCase):
    """Planning, decisions and collection on synthetic results (no training)."""

    @staticmethod
    def sources():
        return {(d, h, s): {'config': e14_source(d, h, s, batch=4), 'source_run': 'x'}
                for d, h in SETTINGS for s in proto.SEEDS}

    @staticmethod
    def write_result(scratch, stage, j, best_mse, last_mse, smoke=False):
        out = scratch / stage / 'runs' / proto.name(j)
        (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (out / 'checkpoints' / 'best.ckpt').write_text(f'best {best_mse}')
        (out / 'checkpoints' / 'last.pt').write_text(f'last {last_mse}')
        record = {'stage': stage, 'arm': j['arm'], 'protocol': j['protocol'], 'init': j['init'],
                  'dataset': j['dataset'], 'horizon': j['horizon'], 'seed': j['seed'],
                  'variant_id': j.get('variant_id'), 'backbone_lr_mult': j.get('backbone_lr_mult', 1.0),
                  'epochs': 10, 'smoke': smoke,
                  'checkpoint': str(out / 'checkpoints' / 'best.ckpt'),
                  'last_checkpoint': str(out / 'checkpoints' / 'last.pt'),
                  'best': {'test_mse': best_mse, 'test_mae': best_mse, 'val_mse': 0.5, 'val_mae': 0.5},
                  'last': {'test_mse': last_mse, 'test_mae': last_mse, 'val_mse': 0.6, 'val_mae': 0.6}}
        (out / 'result.json').write_text(json.dumps(record))

    def e14(self, offset=0.01):
        gold = protocol.golden()
        return {(d, h, s): {'test_mse': gold[(d, h)][0] * (1 + offset), 'test_mae': 0.0, 'val_mse': 0.5,
                            'checkpoint': ''}
                for d, h in SETTINGS for s in proto.SEEDS}

    def test_full_pipeline_counts_and_rules(self):
        gold = protocol.golden()
        sources = self.sources()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(proto, 'e14_records', return_value=self.e14(0.02)), \
                mock.patch.object(proto, 'commit', return_value='c'), \
                mock.patch.object(proto, 'provenance', return_value={}):
            scratch = Path(tmp)
            stage0 = [json.loads(p.read_text()) for p in proto.plan('stage0', scratch, sources, SETTINGS, False)]
            self.assertEqual(len(stage0), 28)
            self.assertTrue(all(j['arm'] == 'pf' and j['epoch_test'] and j['protocol'] == 'conference'
                                for j in stage0))
            # Best-val within 0.5% of the Golden, last-epoch worse: adopt, rule best.
            for j in stage0:
                g = gold[(j['dataset'], j['horizon'])][0]
                self.write_result(scratch, 'stage0', j, g * 1.005, g * 1.03)
            decision = proto.decide(scratch, SETTINGS, False)
            self.assertTrue(decision['adopt_conference'])
            self.assertEqual(decision['checkpoint_rule'], 'best')
            self.assertEqual(len(decision['datasets_better_than_e14']), 7)
            main = [json.loads(p.read_text()) for p in proto.plan('main', scratch, sources, SETTINGS, False)]
            self.assertEqual(len(main), 224)
            counts = collections.Counter(j['arm'] for j in main)
            self.assertEqual(counts, {'pf': 56, 'b0': 84, 'm1': 84})
            for j in main:
                if j['arm'] == 'm1':
                    self.assertEqual(j['variant_id'], protocol.FINAL_VARIANTS.get((j['dataset'], j['horizon'])))
                else:
                    self.assertIsNone(j['variant'])
            probe = [json.loads(p.read_text()) for p in proto.plan('warm-probe', scratch, sources, SETTINGS, False)]
            self.assertEqual(len(probe), 56)
            self.assertTrue(all(j['warm_source']['origin'].startswith('stage0/') for j in probe))
            for j in main:
                self.write_result(scratch, 'main', j, 0.3, 0.31)
            for j in probe:
                mse = 0.2 if (j['dataset'] == 'Weather') == (j['backbone_lr_mult'] == 1.0) else 0.25
                self.write_result(scratch, 'warm-probe', j, mse, mse)
            chosen = proto.warm_select(scratch, SETTINGS, False)['settings']
            self.assertEqual(chosen['Weather_H96']['backbone_lr_mult'], 1.0)
            self.assertEqual(chosen['ETTh1_H96']['backbone_lr_mult'], 0.1)
            warm = [json.loads(p.read_text()) for p in proto.plan('warm-main', scratch, sources, SETTINGS, False)]
            self.assertEqual(len(warm), 56)
            self.assertTrue(all(j['warm_source']['origin'].startswith('main/') for j in warm))
            for j in warm:
                self.write_result(scratch, 'warm-main', j, 0.2, 0.2)
            summary = proto.collect(scratch, False)['summary']
            arms = collections.Counter(e['arm'] for e in summary)
            self.assertEqual(set(arms.values()), {28})
            self.assertEqual(len(arms), 4)
            for entry in summary:
                self.assertEqual(len(entry['seeds']), 3, entry)

    def test_not_adopted_stops_experiment1_and_warm_starts_from_e14(self):
        gold = protocol.golden()
        sources = self.sources()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            e14 = self.e14(0.0)
            for key in e14:
                path = scratch / 'e14' / f'{key}.ckpt'
                path.parent.mkdir(exist_ok=True)
                path.write_text(str(key))
                e14[key]['checkpoint'] = str(path)
            with mock.patch.object(proto, 'e14_records', return_value=e14), \
                    mock.patch.object(proto, 'commit', return_value='c'), \
                    mock.patch.object(proto, 'provenance', return_value={}):
                for p in proto.plan('stage0', scratch, sources, SETTINGS, False):
                    j = json.loads(p.read_text())
                    g = gold[(j['dataset'], j['horizon'])][0]
                    self.write_result(scratch, 'stage0', j, g * 1.03, g * 1.01)
                decision = proto.decide(scratch, SETTINGS, False)
                # Last-epoch is lower by more than the tie margin, but 1% gap and
                # no dataset beats E14 (equal to the Golden here): not adopted.
                self.assertEqual(decision['checkpoint_rule'], 'last')
                self.assertFalse(decision['adopt_conference'])
                with self.assertRaises(RuntimeError):
                    proto.plan('main', scratch, sources, SETTINGS, False)
                probe = [json.loads(p.read_text())
                         for p in proto.plan('warm-probe', scratch, sources, SETTINGS, False)]
                self.assertTrue(all(j['protocol'] == 'e14' and j['warm_source']['origin'] == 'e14'
                                    for j in probe))

    def test_decision_is_immutable(self):
        gold = protocol.golden()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(proto, 'e14_records', return_value=self.e14(0.02)), \
                mock.patch.object(proto, 'commit', return_value='c'), \
                mock.patch.object(proto, 'provenance', return_value={}):
            scratch = Path(tmp)
            for p in proto.plan('stage0', scratch, self.sources(), SETTINGS, False):
                j = json.loads(p.read_text())
                g = gold[(j['dataset'], j['horizon'])][0]
                self.write_result(scratch, 'stage0', j, g, g)
            proto.decide(scratch, SETTINGS, False)
            j = json.loads(next((scratch / 'stage0' / 'jobs').glob('*.json')).read_text())
            self.write_result(scratch, 'stage0', j, 9.0, 9.0)
            with self.assertRaises(ValueError):
                proto.decide(scratch, SETTINGS, False)


if __name__ == '__main__':
    unittest.main()
