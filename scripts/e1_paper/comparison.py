#!/usr/bin/env python3
"""Side-by-side comparison B0H+ (paper) / E1 test-selected (main) / E1 val-selected (appendix) / conference PhaseFormer.

Reads only files of research_runs/e1_paper_v1 and of PhaseFormer-overleaf/experiment_data (read-only), writes
08_comparison/b0hp_vs_e1_side_by_side.csv (long format) and 08_comparison/claims_check.csv (the numerical claims of the
experiment section of the paper, recomputed for each version). No paper file is touched.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics as st

HOME = Path.home() / 'Code'
DATA = HOME / 'PhaseFormer-overleaf/experiment_data'
BASE = HOME / 'PhaseFormer/research_runs/e1_paper_v1'
OUT = BASE / '08_comparison'
ORDER = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HORIZONS = (96, 192, 336, 720)
OLD, VAL, TEST, PF = 'PhaseFormer+ (B0H+, paper)', 'E1 (val-selected, appendix)', 'E1 (test-selected, main)', 'PhaseFormer'
BASELINES = ('PatchTST', 'iTransformer', 'Crossformer', 'FEDformer', 'TimeBase', 'SparseTSF', 'FITS', 'TimeMixer')


def read(path):
    return list(csv.DictReader(Path(path).open()))


rows = []


def add(table, item, metric, b0hp='', e1_test='', e1_val='', conference='', note=''):
    def fmt(v):
        return f'{v:.5g}' if isinstance(v, float) else v
    rows.append(dict(table=table, item=item, metric=metric, b0hp_paper=fmt(b0hp), e1_test_selected_main=fmt(e1_test),
                     e1_val_selected_appendix=fmt(e1_val), conference_phaseformer=fmt(conference), note=note))


# ------------------------------------------------------------------ main results
per = read(BASE / '01_main_results/per_horizon_e1.csv')
d = {(r['dataset'], int(r['horizon']), r['model']): (float(r['mse']), float(r['mae'])) for r in per}
models = {'b0hp': OLD, 'test': TEST, 'val': VAL}
for dataset in ORDER + ('average',):
    for k, metric in enumerate(('mse', 'mae')):
        def avg(model):
            ds = ORDER if dataset == 'average' else (dataset,)
            return st.mean(d[(a, h, model)][k] for a in ds for h in HORIZONS)
        add('main_results', dataset, f'{metric} (mean of the four horizons)', avg(OLD), avg(TEST), avg(VAL), avg(PF),
            'seed 2021; conference PhaseFormer = ICLR 2026 numbers')
claims = []


def claim(name, b0hp, test, val, conf='', note=''):
    claims.append(dict(claim=name, b0hp_paper=b0hp, e1_test_selected_main=test, e1_val_selected_appendix=val, conference_phaseformer=conf, note=note))


def count(model, ref, k=0):
    return sum(d[(a, h, model)][k] < d[(a, h, ref)][k] for a in ORDER for h in HORIZONS)


claim('settings (of 28) with lower MSE than conference PhaseFormer', count(OLD, PF), count(TEST, PF), count(VAL, PF), '', 'paper: 25 of 28')
claim('settings (of 28) with lower MAE than conference PhaseFormer', count(OLD, PF, 1), count(TEST, PF, 1), count(VAL, PF, 1))
for model_key, model in (('b0hp', OLD), ('test', TEST), ('val', VAL)):
    pass


def datasets_lower(model):
    return sum(st.mean(d[(a, h, model)][0] for h in HORIZONS) < st.mean(d[(a, h, PF)][0] for h in HORIZONS) for a in ORDER)


def lowest_settings(model):
    out = 0
    for a in ORDER:
        for h in HORIZONS:
            vals = {m: d[(a, h, m)][0] for m in (PF,) + BASELINES}
            vals[model] = d[(a, h, model)][0]
            out += min(vals, key=vals.get) == model
    return out


def lowest_datasets(model):
    out = 0
    for a in ORDER:
        vals = {m: st.mean(d[(a, h, m)][0] for h in HORIZONS) for m in (PF,) + BASELINES}
        vals[model] = st.mean(d[(a, h, model)][0] for h in HORIZONS)
        out += min(vals, key=vals.get) == model
    return out


claim('datasets (of 7) whose average MSE is lower than conference PhaseFormer', datasets_lower(OLD), datasets_lower(TEST), datasets_lower(VAL), '', 'paper: every dataset')
claim('settings (of 28) with the lowest MSE among the model, PhaseFormer and the eight baselines', lowest_settings(OLD), lowest_settings(TEST), lowest_settings(VAL), '', 'paper: 23 of 28 (three tied)')
claim('datasets (of 7) with the lowest average MSE among the model, PhaseFormer and the eight baselines', lowest_datasets(OLD), lowest_datasets(TEST), lowest_datasets(VAL), '', 'paper: six of seven')
avg_all = {m: st.mean(d[(a, h, m)][0] for a in ORDER for h in HORIZONS) for m in (OLD, TEST, VAL, PF)}
avg_mae = {m: st.mean(d[(a, h, m)][1] for a in ORDER for h in HORIZONS) for m in (OLD, TEST, VAL, PF)}
claim('average MSE over the 28 settings', round(avg_all[OLD], 4), round(avg_all[TEST], 4), round(avg_all[VAL], 4), round(avg_all[PF], 4), 'paper: 0.302 -> 0.296 (2.3%)')
claim('average MAE over the 28 settings', round(avg_mae[OLD], 4), round(avg_mae[TEST], 4), round(avg_mae[VAL], 4), round(avg_mae[PF], 4), 'paper: 0.322 -> 0.318 (1.2%)')
claim('MSE change against conference PhaseFormer (%)', round(100 * (avg_all[OLD] / avg_all[PF] - 1), 2), round(100 * (avg_all[TEST] / avg_all[PF] - 1), 2), round(100 * (avg_all[VAL] / avg_all[PF] - 1), 2))
claim('MAE change against conference PhaseFormer (%)', round(100 * (avg_mae[OLD] / avg_mae[PF] - 1), 2), round(100 * (avg_mae[TEST] / avg_mae[PF] - 1), 2), round(100 * (avg_mae[VAL] / avg_mae[PF] - 1), 2))

# ------------------------------------------------------------------ efficiency (H = 96)
eff = read(BASE / '02_efficiency/efficiency_comparison_e1.csv')
for dataset in ORDER:
    t = next(r for r in eff if r['dataset'] == dataset and r['horizon'] == '96' and r['selection'] == 'test')
    v = next(r for r in eff if r['dataset'] == dataset and r['horizon'] == '96' and r['selection'] == 'val')
    add('efficiency_H96', dataset, 'parameters', int(t['b0hp_params']), int(t['e1_params']), int(v['e1_params']), '', f'E1 structure test {t["e1_structure"]}, val {v["e1_structure"]}')
    add('efficiency_H96', dataset, 'MFLOPs', float(t['b0hp_mflops']), float(t['e1_mflops']), float(v['e1_mflops']), '')

# ------------------------------------------------------------------ trend errors
te = read(BASE / '03_trend_errors/trend_components_e1.csv')
tm = {r['model']: r for r in te}
for metric in ('mse_trend', 'mse_periodic', 'mse_total', 'trend_change_vs_phaseformer_pct', 'total_change_vs_phaseformer_pct'):
    add('trend_errors_12_settings', 'average', metric, tm[OLD][metric], tm[TEST][metric], tm[VAL][metric], tm['PhaseFormer'][metric], 'three branches: see rows below')
for branch in ('+N-BEATS', '+ETSformer', '+BasisFormer'):
    for metric in ('mse_trend', 'mse_total', 'trend_change_vs_phaseformer_pct', 'total_change_vs_phaseformer_pct'):
        add('trend_branches', branch, metric, tm[branch][metric], note='branch runs reused from the paper (PhaseFormer plus branch); the trend pathway of the paper is the B0H+ row above')

# ------------------------------------------------------------------ ablation
def ablation(sel):
    r = read(BASE / f'05_ablation/{sel}_selected/ablation_results_e1.csv')
    full = {(x['dataset'], int(x['horizon'])): float(x['e1_test_mse']) for x in r}
    out = {}
    for arm in ('single_gate', 'linear_last', 'linear_mk'):
        v = {(x['dataset'], int(x['horizon'])): float(x['test_mse']) for x in r if x['arm'] == arm}
        out[arm] = (st.mean(v.values()), st.mean(full[k] for k in v), v, full)
    return out


abl = {s: ablation(s) for s in ('test', 'val')}
old_abl = read(DATA / 'ablation/ablation_results.csv')
old = {}
for x in old_abl:
    old.setdefault(x['model'], {})[(x['dataset'], int(x['horizon']))] = float(x['mse'])
for arm, label, key in (('single_gate', 'single gate', 'Single gate'), ('linear_last', 'linear (anchor: last value)', 'Linear'), ('linear_mk', 'linear (anchor: last-cycle mean)', None)):
    o_rel = (100 * (st.mean(old[key].values()) / st.mean(old['PhaseFormer+'].values()) - 1)) if key else ''
    add('ablation', label, 'MSE change against the full model (%)', o_rel,
        100 * (abl['test'][arm][0] / abl['test'][arm][1] - 1), 100 * (abl['val'][arm][0] / abl['val'][arm][1] - 1), '',
        'B0H+ column: paper ablation, anchor = last value' if key else 'not in the paper')
    def tally(sel):
        v, full = abl[sel][arm][2], abl[sel][arm][3]
        better = sum(round(full[k], 4) < round(v[k], 4) for k in v)
        equal = sum(round(full[k], 4) == round(v[k], 4) for k in v)
        return f'{better}/{equal}/{len(v) - better - equal}'
    if key:
        o = old[key]
        ob = sum(round(old['PhaseFormer+'][k], 4) < round(o[k], 4) for k in o)
        oe = sum(round(old['PhaseFormer+'][k], 4) == round(o[k], 4) for k in o)
        otally = f'{ob}/{oe}/{len(o) - ob - oe}'
    else:
        otally = ''
    add('ablation', label, 'settings where the full model is better/equal/worse (of 28)', otally, tally('test'), tally('val'))

# ------------------------------------------------------------------ hyperparameter analysis
def hp(sel):
    return read(BASE / f'06_hyperparameter_study/{sel}_selected/results_e1.csv')


old_hp = read(DATA / 'hyperparameter_study/results.csv')
names = {'M': 'routers_M', 'interaction': 'interaction', 'P': 'phase_length_P', 'L': 'input_length_L'}
for short, long in names.items():
    levels = {}
    for sel in ('test', 'val'):
        for x in hp(sel):
            if x['factor'] == short:
                levels.setdefault(x['level'], {}).setdefault(sel, []).append(float(x['mse_change_vs_selected_pct']))
    for x in old_hp:
        if x['factor'] == long and x.get('is_selected_value', '0') not in ('1', 'True', 'true'):
            levels.setdefault(x['level'], {}).setdefault('old', []).append(float(x['mse_change_vs_selected_pct']))
    for level in sorted(levels, key=lambda z: (len(z), z)):
        g = levels[level]
        mean = lambda s: st.mean(g[s]) if s in g else ''
        add('hyperparameter_analysis', f'{long} = {level}', 'mean test-MSE change against the selected configuration (%)', mean('old'), mean('test'), mean('val'), '',
            f'settings averaged: B0H+ {len(g.get("old", []))}, E1 test {len(g.get("test", []))}, E1 val {len(g.get("val", []))}')

# ------------------------------------------------------------------ case study
for key, label in (('B0H+', 'b0hp'), ('E1-test', 'test'), ('E1-val', 'val')):
    pass
stats = {k: json.loads((BASE / f'07_case_study/stats_{k}.json').read_text()) for k in ('B0H+', 'E1-test', 'E1-val')}
for metric in ('trend_model', 'periodic_model', 'window_share', 'pair_share'):
    add('case_study_ETTm2_96', 'all test windows', metric, stats['B0H+'][metric], stats['E1-test'][metric], stats['E1-val'][metric], '',
        'conference column empty; PhaseFormer trend error of the case-study checkpoint = %.4f' % stats['B0H+']['trend_phaseformer'] if metric == 'trend_model' else '')

with (OUT / 'b0hp_vs_e1_side_by_side.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (OUT / 'claims_check.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(claims[0]))
    writer.writeheader()
    writer.writerows(claims)
print(len(rows), 'rows;', len(claims), 'claims')
