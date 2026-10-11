"""Checks for experiment 3 (wide LA-only search)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from scripts.vldbj_m1 import protocol, wide
from scripts.vldbj_m1.model import paired_models
from src.models.phaseformer_presets import get_base_hyperparams

SETTINGS = [(d, h) for d in wide.DATASETS for h in wide.HORIZONS]


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


class VariantBoundTests(unittest.TestCase):
    def test_gate_lower_bound_is_inclusive_and_extremes_pass(self):
        for gate in (1e-4, 0.99):
            variant = {'gate_init': gate, 'extension_lr_mult': 100.0, 'communication_width': 3}
            model, _, _, cfg, _ = paired_models(e14_source(), 'm1', 'S2', False, variant)
            self.assertEqual(cfg['vldbj_m1']['gate_init'], gate)
            torch.testing.assert_close(torch.sigmoid(model.lowrank_interaction_gate),
                                       torch.full_like(model.lowrank_interaction_gate, gate),
                                       atol=1e-7, rtol=0)
        with self.assertRaises(ValueError):
            paired_models(e14_source(), 'm1', 'S2', False,
                          {'gate_init': 1e-5, 'extension_lr_mult': 1.0, 'communication_width': 2})

    def test_default_configuration_unchanged(self):
        _, _, _, cfg, _ = paired_models(e14_source(), 'm1', 'S2', False)
        self.assertEqual(cfg['vldbj_m1'],
                         {'arm': 'm1', 'head': 'S2', 'rank': 3, 'shared_gate': False,
                          'communication': 'tp_mod_centered', 'communication_width': 2,
                          'extension_lr_mult': 1.0, 'gate_init': 0.2, 'zero_init_exit': True,
                          'source_config_hash': 'test'})


class GridTests(unittest.TestCase):
    def test_r2_neighbors_deterministic_clamped_capped(self):
        winner = {'gate_init': 0.1, 'extension_lr_mult': 30.0, 'communication_width': 2,
                  'backbone_lr_mult': 1.0}
        grid = wide.r2_variants(winner)
        self.assertLessEqual(len(grid), wide.R2_CAP)
        keys = [wide.variant_key({k: c[k] for k in ('gate_init', 'extension_lr_mult',
                                                     'communication_width')}, c['bb'])
                for c in grid]
        self.assertEqual(len(keys), len(set(keys)))
        # Gate and lr neighbours, the two other widths, two backbone multipliers and the
        # two extreme probes are all present.
        gates = {c['gate_init'] for c in grid}
        lrs = {c['extension_lr_mult'] for c in grid}
        widths = {c['communication_width'] for c in grid}
        bbs = {c['bb'] for c in grid}
        self.assertEqual(gates, {wide.snap(0.1 / 3, 1e-4, 0.99), 0.1, 0.3, 0.99})
        self.assertEqual(lrs, {10.0, 30.0, 90.0, 100.0})
        self.assertEqual(widths, {2, 3, 4})
        self.assertEqual(bbs, {0.3, 1.0, 3.0})
        # Clamping at the boundaries of the registered ranges.
        low = wide.r2_variants({'gate_init': 1e-4, 'extension_lr_mult': 0.1,
                                'communication_width': 4, 'backbone_lr_mult': 1.0})
        self.assertTrue(all(c['gate_init'] >= 1e-4 and c['extension_lr_mult'] <= 100.0 for c in low))

    def test_r2_cap_shrinks_with_elapsed_time(self):
        import datetime as dt
        start = dt.datetime(2026, 10, 11, 22, 45)
        cases = [(0.5, wide.R2_CAP), (3.5, 10), (4.5, 6), (5.5, 3)]
        for hours, expected in cases:
            self.assertEqual(wide.r2_cap(start + dt.timedelta(hours=hours), start), expected)


class PipelineTests(unittest.TestCase):
    """select/confirm/collect on fabricated runs (no training)."""

    @staticmethod
    def write_run(scratch, stage, dataset, horizon, seed, variant, bb, mse, off, smoke=False):
        out = wide.name(dataset, horizon, seed, variant, bb)
        run = scratch / stage / 'runs' / out
        run.mkdir(parents=True, exist_ok=True)
        (run / 'config.json').write_text(json.dumps(
            {'vldbj_m1': {'gate_init': variant['gate_init'],
                          'extension_lr_mult': variant['extension_lr_mult'],
                          'communication_width': variant['communication_width']}}))
        (run / 'result.json').write_text(json.dumps(
            {'name': out, 'smoke': smoke, 'dataset': dataset, 'horizon': horizon, 'seed': seed,
             'backbone_lr_mult': bb, 'epochs': 20,
             'best': {'test_mse': mse, 'test_mae': mse + 0.03,
                      'test_mse_modulation_off': off, 'test_mae_modulation_off': off + 0.03}}))
        return run

    @staticmethod
    def fake_current():
        out = {}
        for dataset, horizon in SETTINGS:
            native = 0.30
            out[(dataset, horizon)] = {
                'per_seed': {str(s): {'native': native, 'native_mae': 0.33,
                                      'off': native * 1.05, 'off_mae': 0.35}
                             for s in (2021, 2022, 2023)},
                'native_mean': native, 'mae_mean': 0.33,
                'config': 'current (paper final configuration)'}
        return out

    def test_select_confirm_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            win = {'gate_init': 0.001, 'extension_lr_mult': 10.0, 'communication_width': 2}
            near = {'gate_init': 0.5, 'extension_lr_mult': 1.0, 'communication_width': 2}
            # ETTh1-96: the near-off gate wins clearly; ETTm2-96: a wide run wins only
            # via its modulation-off read; Traffic-96: no wide run beats current.
            self.write_run(scratch, 'r1', 'ETTh1', 96, 2021, win, 1.0, 0.2850, 0.29)
            self.write_run(scratch, 'r1', 'ETTh1', 96, 2021, near, 1.0, 0.2999, 0.31)
            self.write_run(scratch, 'r1', 'ETTm2', 96, 2021, near, 1.0, 0.31, 0.2950)
            self.write_run(scratch, 'r1', 'Traffic', 96, 2021, near, 3.0, 0.2995, 0.32)
            with mock.patch.object(wide, 'current_records', return_value=self.fake_current()), \
                    mock.patch.object(wide, 'commit', return_value='c'), \
                    mock.patch.object(wide, 'provenance', return_value={}):
                provisional = wide.select(scratch, False)['settings']
                self.assertTrue(provisional['ETTh1_H96']['needs_confirm'])
                self.assertEqual(provisional['ETTh1_H96']['best']['variant'], win)
                # The off-read candidate of ETTm2-96 wins over its native read.
                self.assertEqual(provisional['ETTm2_H96']['best']['read'], 'off')
                self.assertTrue(provisional['ETTm2_H96']['needs_confirm'])
                # Below the 0.2% switch margin: keep the current configuration.
                self.assertFalse(provisional['Traffic_H96']['needs_confirm'])
                self.assertEqual(provisional['Weather_H96']['best']['stage'], 'current')
                self.assertEqual(provisional['Weather_H96']['candidate_count'], 2)
                sources = {(d, h, s): {'config': {}} for d, h in SETTINGS for s in (2021, 2022, 2023)}
                jobs = wide.confirm_jobs(scratch, sources, False)
                names = sorted(Path(json.loads(p.read_text())['output']).name for p in jobs)
                self.assertEqual(names, [wide.name('ETTh1', 96, s, win, 1.0) for s in (2022, 2023)] +
                                        [wide.name('ETTm2', 96, s, near, 1.0) for s in (2022, 2023)])
                # Confirm runs exist for ETTh1 but not ETTm2 (deadline case).
                for seed in (2022, 2023):
                    self.write_run(scratch, 'confirm', 'ETTh1', 96, seed, win, 1.0, 0.286, 0.29)
                selection = wide.collect(scratch, False)['settings']
                self.assertTrue(selection['ETTh1_H96']['switched'])
                self.assertEqual(selection['ETTh1_H96']['final_read'], 'native')
                self.assertAlmostEqual(selection['ETTh1_H96']['confirm']['mean'],
                                       (0.2850 + 0.286 + 0.286) / 3)
                self.assertEqual(selection['ETTm2_H96']['confirm'], 'incomplete (deadline); current configuration kept')
                self.assertFalse(selection['ETTm2_H96']['switched'])
                self.assertFalse(selection['Traffic_H96']['switched'])
                self.assertFalse(selection['Weather_H96']['switched'])

    def test_current_modoff_switch_needs_no_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            current = self.fake_current()
            for (dataset, horizon), record in current.items():
                record['per_seed'] = {s: {'native': 0.30, 'native_mae': 0.33,
                                           'off': 0.29, 'off_mae': 0.32}
                                      for s in ('2021', '2022', '2023')}
            with mock.patch.object(wide, 'current_records', return_value=current), \
                    mock.patch.object(wide, 'commit', return_value='c'), \
                    mock.patch.object(wide, 'provenance', return_value={}):
                provisional = wide.select(scratch, False)['settings']
                best = provisional['ETTh1_H96']['best']
                self.assertEqual((best['stage'], best['read']), ('current', 'off'))
                self.assertFalse(provisional['ETTh1_H96']['needs_confirm'])
                selection = wide.collect(scratch, False)['settings']['ETTh1_H96']
                self.assertTrue(selection['switched'])
                self.assertEqual(selection['final_read'], 'off')
                self.assertAlmostEqual(selection['final_seed_mse']['2022'], 0.29)


if __name__ == '__main__':
    unittest.main()
