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
        self.assertEqual(gates, {wide.snap(0.1 / 3, 1e-4, wide.GATE_MAX), 0.1, 0.3, wide.GATE_MAX})
        self.assertEqual(lrs, {10.0, 30.0, 90.0, 1000.0})
        self.assertEqual(widths, {2, 3, 4})
        self.assertEqual(bbs, {0.3, 1.0, 3.0})
        # Clamping at the boundaries of the registered ranges.
        low = wide.r2_variants({'gate_init': 1e-4, 'extension_lr_mult': 0.1,
                                'communication_width': 4, 'backbone_lr_mult': 1.0})
        self.assertTrue(all(c['gate_init'] >= 1e-4 and c['extension_lr_mult'] <= wide.LR_MAX for c in low))

    def test_r2_cap_shrinks_with_elapsed_time(self):
        import datetime as dt
        start = dt.datetime(2026, 10, 11, 22, 45)
        cases = [(0.5, wide.R2_CAP), (3.5, 10), (4.5, 6), (5.5, 3)]
        for hours, expected in cases:
            self.assertEqual(wide.r2_cap(start + dt.timedelta(hours=hours), start), expected)


class PipelineTests(unittest.TestCase):
    """select/confirm/collect on fabricated runs (no training)."""

    @staticmethod
    def write_run(scratch, stage, dataset, horizon, seed, variant, bb, mse, off, smoke=False,
                  head='S2', shared=None):
        shared = wide.protocol.FINAL_SHARED_GATE[dataset] if shared is None else shared
        out = wide.name(dataset, horizon, seed, variant, bb, head, shared)
        run = scratch / stage / 'runs' / out
        run.mkdir(parents=True, exist_ok=True)
        (run / 'config.json').write_text(json.dumps(
            {'vldbj_m1': {'gate_init': variant['gate_init'],
                          'extension_lr_mult': variant['extension_lr_mult'],
                          'communication_width': variant['communication_width'],
                          'head': head, 'shared_gate': shared}}))
        (run / 'result.json').write_text(json.dumps(
            {'name': out, 'smoke': smoke, 'dataset': dataset, 'horizon': horizon, 'seed': seed,
             'backbone_lr_mult': bb, 'epochs': 20,
             'best': {'test_mse': mse, 'test_mae': mse + 0.03,
                      'test_mse_modulation_off': off, 'test_mae_modulation_off': off + 0.03}}))
        return run

    @staticmethod
    def fake_baseline(scratch):
        """Fabricated exp3 final selection: every setting kept at 0.30 / 0.33."""
        settings = {}
        for dataset, horizon in SETTINGS:
            settings[f'{dataset}_H{horizon}'] = {
                'switched': False,
                'final_config': 'exp3-final: current (paper final configuration)',
                'final_read': 'native',
                'final_seed_mse': {str(s): 0.30 for s in (2021, 2022, 2023)},
                'final_seed_mae': {str(s): 0.33 for s in (2021, 2022, 2023)},
                'provisional': None}
        (scratch / 'wide_selection.json').write_text(json.dumps({'settings': settings}))

    def test_select_confirm_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            self.fake_baseline(scratch)
            win = {'gate_init': 0.001, 'extension_lr_mult': 10.0, 'communication_width': 2}
            near = {'gate_init': 0.5, 'extension_lr_mult': 1.0, 'communication_width': 2}
            # ETTh1-96: a wide run wins clearly; ETTm2-96: a wide run wins only via
            # its modulation-off read; Traffic-96: no wide run beats the baseline.
            self.write_run(scratch, 'r1', 'ETTh1', 96, 2021, win, 1.0, 0.2850, 0.29)
            self.write_run(scratch, 'r1', 'ETTh1', 96, 2021, near, 1.0, 0.2999, 0.31)
            self.write_run(scratch, 'r1', 'ETTm2', 96, 2021, near, 1.0, 0.31, 0.2950)
            self.write_run(scratch, 'r1', 'Traffic', 96, 2021, near, 3.0, 0.2995, 0.32)
            with mock.patch.object(wide, 'commit', return_value='c'), \
                    mock.patch.object(wide, 'provenance', return_value={}):
                provisional = wide.select(scratch, False)['settings']
                self.assertTrue(provisional['ETTh1_H96']['needs_confirm'])
                self.assertEqual(provisional['ETTh1_H96']['best']['variant'], win)
                self.assertEqual(provisional['ETTh1_H96']['best']['stage'], 'r1')
                # The off-read candidate of ETTm2-96 wins over its native read.
                self.assertEqual(provisional['ETTm2_H96']['best']['read'], 'off')
                self.assertTrue(provisional['ETTm2_H96']['needs_confirm'])
                # Below the 0.2% switch margin: keep the baseline.
                self.assertFalse(provisional['Traffic_H96']['needs_confirm'])
                # Settings without wide runs fall back to the baseline candidate.
                self.assertEqual(provisional['Weather_H96']['best']['stage'], 'baseline')
                self.assertEqual(provisional['Weather_H96']['candidate_count'], 1)
                sources = {(d, h, s): {'config': {}} for d, h in SETTINGS for s in (2021, 2022, 2023)}
                jobs = wide.confirm_jobs(scratch, sources, False)
                names = sorted(Path(json.loads(p.read_text())['output']).name for p in jobs)
                self.assertEqual(names, [wide.name('ETTh1', 96, s, win, 1.0) for s in (2022, 2023)] +
                                        [wide.name('ETTm2', 96, s, near, 1.0) for s in (2022, 2023)])
                # Confirm runs exist for ETTh1 but not ETTm2 (deadline case).
                for seed in (2022, 2023):
                    self.write_run(scratch, 'confirm', 'ETTh1', 96, seed, win, 1.0, 0.286, 0.29)
                selection = wide.collect(scratch, False)['settings']
                self.assertTrue((scratch / 'wide_selection_r3.json').exists())
                self.assertTrue(selection['ETTh1_H96']['switched'])
                self.assertEqual(selection['ETTh1_H96']['final_read'], 'native')
                self.assertAlmostEqual(selection['ETTh1_H96']['confirm']['mean'],
                                       (0.2850 + 0.286 + 0.286) / 3)
                self.assertEqual(selection['ETTm2_H96']['confirm'], 'incomplete (deadline); baseline kept')
                self.assertFalse(selection['ETTm2_H96']['switched'])
                self.assertFalse(selection['Traffic_H96']['switched'])
                self.assertEqual(selection['Weather_H96']['final_config'],
                               'exp3-final: current (paper final configuration)')

    def test_r3_grid_boundary_probes_and_head_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            settings = json.loads((scratch / 'wide_selection.json').read_text()) if False else None
            self.fake_baseline(scratch)
            # One setting's baseline sits on the searched bounds: lr 100 and bb 3.
            record = json.loads((scratch / 'wide_selection.json').read_text())
            record['settings']['ETTh1_H96'].update(
                switched=True,
                final_config='wide: lr100',
                provisional={'variant': {'gate_init': 0.1, 'extension_lr_mult': 100.0,
                                         'communication_width': 2},
                             'backbone_lr_mult': 3.0})
            (scratch / 'wide_selection.json').write_text(json.dumps(record))
            sources = {(d, h, s): {'config': {}} for d, h in SETTINGS for s in (2021, 2022, 2023)}
            with mock.patch.object(wide, 'commit', return_value='c'), \
                    mock.patch.object(wide, 'provenance', return_value={}):
                jobs = [json.loads(p.read_text()) for p in wide.stage_jobs(scratch, 'r3', sources, False)]
            etth1 = [j for j in jobs if j['dataset'] == 'ETTh1' and j['horizon'] == 96]
            names = {Path(j['output']).name for j in etth1}
            self.assertIn('ETTh1_H96_S2021__m1__g0.1_lr1000_w2_bb3', names)
            self.assertIn('ETTh1_H96_S2021__m1__g0.1_lr100_w2_bb10', names)
            self.assertTrue(all(j['head'] == 'S2' for j in etth1))
            heads = [j for j in jobs if j['dataset'] in wide.R3_DATASETS and j['head'] != 'S2']
            self.assertEqual(len(heads), 12 * 4)
            self.assertEqual(len({Path(j['output']).name for j in heads}), 12 * 4)
            # Default names stay backward compatible with the round-2 layout.
            self.assertEqual(wide.name('ETTh1', 96, 2021, None, 1.0),
                             'ETTh1_H96_S2021__m1__g0.2_lr1_w2_bb1')
            self.assertNotEqual(wide.name('ETTh1', 96, 2021, None, 1.0, 'S1', True),
                                wide.name('ETTh1', 96, 2021, None, 1.0))

    def test_baseline_records_parse_the_previous_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            self.fake_baseline(scratch)
            record = json.loads((scratch / 'wide_selection.json').read_text())
            record['settings']['ETTh1_H96'].update(
                switched=True, final_read='off',
                provisional={'variant': {'gate_init': 0.5, 'extension_lr_mult': 1.0,
                                         'communication_width': 3},
                             'backbone_lr_mult': 0.3})
            (scratch / 'wide_selection.json').write_text(json.dumps(record))
            baseline = wide.baseline_records(scratch)
            self.assertEqual(len(baseline), 28)
            entry = baseline[('ETTh1', 96)]
            self.assertEqual((entry['read'], entry['backbone_lr_mult']), ('off', 0.3))
            self.assertEqual(entry['variant']['communication_width'], 3)
            # Kept settings fall back to the pilot-tuned configuration.
            kept = baseline[('ETTm1', 96)]  # FINAL_VARIANTS has a tune-grid entry here
            self.assertEqual(kept['variant'], wide.TUNE_GRID[wide.protocol.FINAL_VARIANTS[('ETTm1', 96)]])
            self.assertEqual(kept['read'], 'native')


if __name__ == '__main__':
    unittest.main()
