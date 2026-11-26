#!/usr/bin/env python3
"""Tables of batch 1 of the E1 replication: main results, efficiency and trend errors (local, no GPU).

Reads the paper data in PhaseFormer-overleaf/experiment_data and the E1 files of research_runs/e1_paper_v1, writes
01_main_results, 02_efficiency and 03_trend_errors tables next to them. PhaseFormer+ of the paper (B0H+) is labelled
'PhaseFormer+ (B0H+, paper)'; E1 rows carry the selection that produced them ('test' = test-selected, the main tables (the selection of the B0H+ configurations of the paper);
'val' = validation-selected, the appendix analysis).
"""
from __future__ import annotations

import csv
from pathlib import Path
import statistics as st

HOME = Path.home() / 'Code'
DATA = HOME / 'PhaseFormer-overleaf/experiment_data'
BASE = HOME / 'PhaseFormer/research_runs/e1_paper_v1'
ORDER = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HORIZONS = (96, 192, 336, 720)
OLD, VAL, TEST = 'PhaseFormer+ (B0H+, paper)', 'E1 (val-selected, appendix)', 'E1 (test-selected, main)'


def read(path):
    return list(csv.DictReader(Path(path).open()))


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_results():
    paper = read(DATA / 'main_results/per_horizon_results.csv')
    final = read(BASE / '00_protocol/e1_final_configs_28.csv')
    rows = []
    for r in paper:
        model = OLD if r['model'] == 'PhaseFormer+' else r['model']
        rows.append(dict(dataset=r['dataset'], horizon=r['horizon'], model=model, mse=r['mse'], mae=r['mae'], source=r['source']))
    for r in final:
        rows.append(dict(dataset=r['dataset'], horizon=r['horizon'], model=VAL if r['selection'] == 'val' else TEST,
                         mse=r['test_mse'], mae=r['test_mae'], source=f'{r["stage"]}: {r["run_dir"]} (seed 2021)'))
    write(BASE / '01_main_results/per_horizon_e1.csv', rows)
    table1 = []
    models = sorted({r['model'] for r in rows})
    for dataset in ORDER:
        for model in models:
            vals = [(float(r['mse']), float(r['mae'])) for r in rows if r['dataset'] == dataset and r['model'] == model]
            if len(vals) == 4:
                table1.append(dict(dataset=dataset, model=model, mse=round(st.mean(v[0] for v in vals), 5),
                                   mae=round(st.mean(v[1] for v in vals), 5), source='mean of the four horizons of per_horizon_e1.csv'))
    write(BASE / '01_main_results/table1_e1.csv', table1)
    configs = [{k: r[k] for k in ('selection', 'dataset', 'horizon', 'stage', 'latent_dim', 'routers', 'layers', 'period', 'heads',
                                  'head', 'shared_gate', 'huber_delta', 'gate_init', 'gate_mult', 'readout_mult', 'backbone_mult',
                                  'attn_dropout', 'pos_dropout', 'pred_dropout', 'revin_affine', 'state_dim', 'params', 'val_mse',
                                  'test_mse', 'test_mae', 'run_dir')} for r in final]
    write(BASE / '01_main_results/e1_configs_28.csv', configs)
    lines = ['# 主结果摘要（E1，seed 2021）', '', '七个数据集的平均（四个 horizon 的均值再平均）：', '',
             '| 模型 | 平均 MSE | 平均 MAE |', '|---|---:|---:|']
    for model in (OLD, VAL, TEST, 'PhaseFormer', 'TimeBase', 'SparseTSF', 'FITS', 'PatchTST', 'iTransformer', 'Crossformer', 'FEDformer', 'TimeMixer'):
        means = [(float(t['mse']), float(t['mae'])) for t in table1 if t['model'] == model]
        if len(means) == 7:
            lines.append(f'| {model} | {st.mean(m[0] for m in means):.4f} | {st.mean(m[1] for m in means):.4f} |')
    # 28 settings: how many have the lowest MSE among the paper models plus the E1 version
    for label, e1 in (('E1（验证集选出，附录）', VAL), ('E1（测试集选出，主表）', TEST)):
        pool = [m for m in models if m not in (OLD, VAL, TEST)] + [e1]
        wins = 0
        for dataset in ORDER:
            for h in HORIZONS:
                vals = {r['model']: float(r['mse']) for r in rows if r['dataset'] == dataset and int(r['horizon']) == h and r['model'] in pool}
                wins += int(min(vals, key=vals.get) == e1)
        lines.append('')
        lines.append(f'{label} 在 28 个 setting 里 MSE 最低（与 PhaseFormer 和八个基线比）：{wins} 个。')
    (BASE / '01_main_results/summary.md').write_text('\n'.join(lines) + '\n')


def efficiency():
    old = {(r['dataset'], int(r['horizon'])): r for r in read(DATA / 'efficiency/phaseformer_plus_efficiency_28.csv')}
    e1 = read(BASE / '02_efficiency/e1_efficiency_28.csv')
    rows = []
    for r in e1:
        o = old[(r['dataset'], int(r['horizon']))]
        rows.append(dict(dataset=r['dataset'], horizon=r['horizon'], selection=r['selection'],
                         b0hp_params=o['params_total'], e1_params=r['params_total'],
                         b0hp_mflops=o['mflops_total'], e1_mflops=r['mflops_total'],
                         e1_phase_params=r['params_phase_pathway'], e1_trend_gate_params=r['params_trend_pathway_and_gates'],
                         e1_trend_mflops=r['mflops_trend_pathway'], e1_structure=f'd{r["latent_dim"]}_M{r["routers"]}_N{r["layers"]}_P{r["period"]}'))
    write(BASE / '02_efficiency/efficiency_comparison_e1.csv', rows)


def trend_errors():
    old = read(DATA / 'trend_errors/trend_errors_per_setting.csv')
    e1 = read(BASE / '03_trend_errors/e1_trend_scores.csv')
    rows = []
    for r in old:
        model = OLD if r['model'] == 'PhaseFormer+' else r['model']
        rows.append(dict(setting=r['setting'], model=model, mse_trend=r['mse_trend'], mse_periodic=r['mse_periodic'],
                         mse_total=r['mse_total'], seeds=r['n_seeds_averaged']))
    for r in e1:
        rows.append(dict(setting=f'{r["dataset"]}-{r["horizon"]}', model=VAL if r['selection'] == 'val' else TEST,
                         mse_trend=r['mse_trend'], mse_periodic=r['mse_periodic'], mse_total=r['mse_total'], seeds=1))
    write(BASE / '03_trend_errors/trend_errors_per_setting_e1.csv', rows)
    models = sorted({r['model'] for r in rows})
    settings = sorted({r['setting'] for r in rows})
    avg, lines = [], ['# 趋势误差摘要（12 个 setting 的平均）', '',
                      '| 模型 | MSE_trend | MSE_periodic | MSE_total | 趋势误差相对 PhaseFormer | 总误差相对 PhaseFormer |', '|---|---:|---:|---:|---:|---:|']
    base = {m: [float(r[m]) for r in rows if r['model'] == 'PhaseFormer'] for m in ('mse_trend', 'mse_periodic', 'mse_total')}
    order = ['DLinear', 'SparseTSF', 'FITS', 'CycleNet', 'PhaseFormer', '+N-BEATS', '+ETSformer', '+BasisFormer', OLD, VAL, TEST]
    for model in order:
        sel = {r['setting']: r for r in rows if r['model'] == model}
        if len(sel) != 12:
            continue
        mt = st.mean(float(sel[s]['mse_trend']) for s in settings)
        mp = st.mean(float(sel[s]['mse_periodic']) for s in settings)
        mo = st.mean(float(sel[s]['mse_total']) for s in settings)
        dt = 100 * (mt / st.mean(base['mse_trend']) - 1)
        do = 100 * (mo / st.mean(base['mse_total']) - 1)
        avg.append(dict(model=model, mse_trend=round(mt, 6), mse_periodic=round(mp, 6), mse_total=round(mo, 6),
                        trend_change_vs_phaseformer_pct=round(dt, 2), total_change_vs_phaseformer_pct=round(do, 2)))
        lines.append(f'| {model} | {mt:.4f} | {mp:.4f} | {mo:.4f} | {dt:+.1f}% | {do:+.1f}% |')
    write(BASE / '03_trend_errors/trend_components_e1.csv', avg)
    for e1_label in (VAL, TEST):
        pool = [m for m in order if m != OLD and m not in (VAL, TEST)] + [e1_label]
        best_trend = best_total = 0
        for s in settings:
            cells = {r['model']: r for r in rows if r['setting'] == s and r['model'] in pool}
            best_trend += int(min(cells, key=lambda m: float(cells[m]['mse_trend'])) == e1_label)
            best_total += int(min(cells, key=lambda m: float(cells[m]['mse_total'])) == e1_label)
        lines.append('')
        lines.append(f'{e1_label} 的趋势误差最低的 setting：{best_trend}/12；总误差最低的 setting：{best_total}/12（与 DLinear、SparseTSF、FITS、CycleNet、PhaseFormer 和三个分支比）。')
    (BASE / '03_trend_errors/summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main_results()
    efficiency()
    trend_errors()
    print('done')
