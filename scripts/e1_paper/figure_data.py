#!/usr/bin/env python3
"""Builds a copy of the paper's experiment_data/ whose tidy tables carry the E1 numbers, for the figure scripts.

The PhaseFormer+ rows of the per-horizon results, the efficiency table, the trend errors, the ablation and the
hyperparameter study are replaced by the test-selected E1 results of research_runs/e1_paper_v1; every other row
(PhaseFormer, baselines, trend branches) is kept. The figure scripts are then run from the copy.

    python3 scripts/e1_paper/figure_data.py <experiment_data dir> <copy dir>
    python3 <copy dir>/figure_scripts/make_exp_figures.py <figure output dir>
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'research_runs/e1_paper_v1'
E1 = 'E1 (test-selected, main)'
FIGURE_SRC = Path(__file__).resolve().parents[2] / 'paper/vldbj/figures/src'


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('runs', 'phaseformer_plus_runs', 'figure_output'))
    per = {(r['dataset'], r['horizon']): r for r in rd(BASE / '01_main_results/per_horizon_e1.csv') if r['model'] == E1}

    # per-horizon accuracy
    p = dst / 'main_results/per_horizon_results.csv'
    rows = rd(p)
    for r in rows:
        if r['model'] == 'PhaseFormer+':
            e = per[(r['dataset'], r['horizon'])]
            r.update(mse=e['mse'], mae=e['mae'], source='E1, test-selected configuration (seed 2021)')
    write(p, list(rows[0]), rows)

    # efficiency
    p = dst / 'efficiency/phaseformer_plus_efficiency_28.csv'
    fields = list(rd(p)[0])
    rename = {'phase_length_P': 'period', 'routers_M': 'routers', 'attn_heads': 'heads'}
    rows = [{k: r[rename.get(k, k)] for k in fields}
            for r in rd(BASE / '02_efficiency/e1_efficiency_28.csv') if r['selection'] == 'test']
    write(p, fields, rows)

    # trend errors
    p = dst / 'trend_errors/trend_errors_per_setting.csv'
    rows = rd(p)
    e1 = {r['setting']: r for r in rd(BASE / '03_trend_errors/trend_errors_per_setting_e1.csv') if r['model'] == E1}
    for r in rows:
        if r['model'] == 'PhaseFormer+':
            e = e1[r['setting']]
            r.update(mse_trend=e['mse_trend'], mse_periodic=e['mse_periodic'], mse_total=e['mse_total'])
    write(p, list(rows[0]), rows)

    # ablation: single gate and the linear variant anchored at the last cycle mean
    p = dst / 'ablation/ablation_results.csv'
    rows = rd(p)
    abl = {(r['dataset'], r['horizon'], r['arm']): r for r in rd(BASE / '05_ablation/test_selected/ablation_results_e1.csv')}
    arm = {'Single gate': 'single_gate', 'Linear': 'linear_mk'}
    for r in rows:
        key = (r['dataset'], r['horizon'])
        if r['model'] == 'PhaseFormer+':
            r.update(mse=per[key]['mse'], mae=per[key]['mae'], source='E1, test-selected configuration')
        elif r['model'] in arm:
            a = abl[key + (arm[r['model']],)]
            r.update(mse=a['test_mse'], mae=a['test_mae'], source=f"E1 ablation, {arm[r['model']]}")
    write(p, list(rows[0]), rows)

    # hyperparameter study
    p = dst / 'hyperparameter_study/results.csv'
    fields = list(rd(p)[0])
    cfg = {(r['dataset'], r['horizon']): r for r in rd(BASE / '00_protocol/e1_final_configs_28.csv') if r['selection'] == 'test'}
    factor = {'M': 'routers_M', 'interaction': 'interaction', 'P': 'phase_length_P', 'L': 'input_length_L'}
    level = {'routers': 'router'}
    rows = []
    hp = rd(BASE / '06_hyperparameter_study/test_selected/results_e1.csv')
    settings = sorted({(r['dataset'], r['horizon']) for r in hp})
    for f in ('M', 'interaction', 'P', 'L'):
        for d, h in settings:
            todo = [r for r in hp if r['factor'] == f and (r['dataset'], r['horizon']) == (d, h)]
            if not todo:
                continue
            sel = {'M': cfg[(d, h)]['routers'], 'P': cfg[(d, h)]['period'], 'interaction': 'router', 'L': '720'}[f]
            rows.append({'factor': factor[f], 'dataset': d, 'horizon': h, 'level': sel, 'is_selected_value': '1',
                         'test_mse': per[(d, h)]['mse'], 'test_mae': per[(d, h)]['mae'],
                         'mse_change_vs_selected_pct': '0.0000', 'run_files': 'E1 selected run'})
            for r in todo:
                rows.append({'factor': factor[f], 'dataset': d, 'horizon': h, 'level': level.get(r['level'], r['level']),
                             'is_selected_value': '0', 'test_mse': r['test_mse'], 'test_mae': r['test_mae'],
                             'mse_change_vs_selected_pct': f"{float(r['mse_change_vs_selected_pct']):.4f}",
                             'run_files': r['run_dir']})
    write(p, fields, rows)

    # case study windows chosen for E1
    shutil.copy(FIGURE_SRC / 'case_trend.npz', dst / 'case_study/case_trend.npz')
    print('wrote', dst)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
