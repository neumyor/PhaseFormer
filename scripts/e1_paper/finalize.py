#!/usr/bin/env python3
"""Summaries of the ablation (05) and the hyperparameter analysis (06) of the E1 replication, next to the B0H+ numbers of the paper.

Reads 05_ablation/ablation_results_e1.csv and 06_hyperparameter_study/results_e1.csv (fetched from the server) and the paper data in
PhaseFormer-overleaf/experiment_data. Writes 05_ablation/ablation_summary.md and 06_hyperparameter_study/hp_summary.md.
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


def read(path):
    return list(csv.DictReader(Path(path).open()))


def ablation(sel):
    e1 = read(BASE / f'05_ablation/{sel}_selected/ablation_results_e1.csv')
    old = read(DATA / 'ablation/ablation_results.csv')
    main = {(r['dataset'], int(r['horizon'])): float(r['e1_test_mse']) for r in e1}
    arms = {}
    for r in e1:
        arms.setdefault(r['arm'], {})[(r['dataset'], int(r['horizon']))] = float(r['test_mse'])
    old_m = {}
    for r in old:
        old_m.setdefault(r['model'], {})[(r['dataset'], int(r['horizon']))] = float(r['mse'])
    label_sel = '测试集选出（主）' if sel == 'test' else '验证集选出（附录）'
    lines = [f'# 消融摘要（{label_sel}）', '',
             f'E1 列为{label_sel}的 E1 配置（seed 2021）；变体沿用该配置，不单独调参。"持平"按四位小数相等计。', '',
             '## 各变体相对完整模型的平均 MSE 变化（七个数据集，四个 horizon 的均值）', '',
             '| 变体 | 七数据集平均 MSE | 相对完整模型 | 完整模型更好 / 持平 / 更差（28 个 setting） | 完整模型更好的数据集数 |', '|---|---:|---:|---|---:|']
    full_all = st.mean(main[(d, h)] for d in ORDER for h in HORIZONS)

    def summarize(label, variant, reference):
        keys = [(d, h) for d in ORDER for h in HORIZONS if (d, h) in variant]
        if len(keys) < 28:
            lines.append(f'| {label} | 未完成（{len(keys)}/28） | | | |')
            return
        mean_all = st.mean(variant[k] for k in keys)
        ref_all = st.mean(reference[k] for k in keys)
        better = sum(round(reference[k], 4) < round(variant[k], 4) for k in keys)
        equal = sum(round(reference[k], 4) == round(variant[k], 4) for k in keys)
        worse = len(keys) - better - equal
        data_better = sum(st.mean(reference[(d, h)] for h in HORIZONS) < st.mean(variant[(d, h)] for h in HORIZONS) for d in ORDER)
        lines.append(f'| {label} | {mean_all:.4f} | {100 * (mean_all / ref_all - 1):+.2f}% | {better} / {equal} / {worse} | {data_better} |')

    for label, key in (('E1 单门控', 'single_gate'), ('E1 线性（锚点：最后一个值）', 'linear_last'), ('E1 线性（锚点：m_K）', 'linear_mk')):
        summarize(label, arms.get(key, {}), main)
    lines.append('')
    lines.append(f'E1 完整模型的七数据集平均 MSE：{full_all:.4f}。')
    lines += ['', '## 论文里 B0H+ 的同一统计（experiment_data/ablation/ablation_results.csv）', '',
              '| 变体 | 七数据集平均 MSE | 相对完整模型 | 完整模型更好 / 持平 / 更差 | 完整模型更好的数据集数 |', '|---|---:|---:|---|---:|']
    for label, key in (('B0H+ 单门控', 'Single gate'), ('B0H+ 线性', 'Linear')):
        summarize(label, old_m[key], old_m['PhaseFormer+'])
    check = [r for r in e1 if r['arm'] == 'baseline']
    if check:
        lines += ['', '## 复现检查（同一选出配置重训）', '', '| setting | 重训的测试 MSE | 选出 run 的测试 MSE | 相对差 |', '|---|---:|---:|---:|']
        for r in check:
            lines.append(f'| {r["dataset"]}-{r["horizon"]} | {float(r["test_mse"]):.5f} | {float(r["e1_test_mse"]):.5f} | '
                         f'{100 * (float(r["test_mse"]) / float(r["e1_test_mse"]) - 1):+.3f}% |')
    (BASE / f'05_ablation/ablation_summary_{sel}.md').write_text('\n'.join(lines) + '\n')


def hyperparameters(sel):
    e1 = read(BASE / f'06_hyperparameter_study/{sel}_selected/results_e1.csv')
    old = read(DATA / 'hyperparameter_study/results.csv')
    names = {'M': 'routers_M', 'interaction': 'interaction', 'P': 'phase_length_P', 'L': 'input_length_L'}

    def table(rows, factor_key, change_key, level_key):
        cells = {}
        for r in rows:
            if r[factor_key] in (names[f] for f in names) or r[factor_key] in names:
                pass
        return cells

    lines = [f'# 超参数分析摘要（{"测试集选出（主）" if sel == "test" else "验证集选出（附录）"}）', '',
             '每个水平的数字是该水平相对所选配置的测试 MSE 平均变化（%），在 Traffic、Electricity、Weather 的 H=96/336 上取平均；'
             '括号里是该水平参与平均的 setting 数。', '']
    for short, long in names.items():
        lines += [f'## {long}', '', f'| 水平 | E1（{"测试集选出，主" if sel == "test" else "验证集选出，附录"}） | B0H+（论文） |', '|---|---:|---:|']
        e1_levels, old_levels = {}, {}
        for r in e1:
            if r['factor'] == short:
                e1_levels.setdefault(r['level'], []).append(float(r['mse_change_vs_selected_pct']))
        for r in old:
            if r['factor'] == long and r.get('is_selected_value', '0') not in ('1', 'True', 'true'):
                old_levels.setdefault(r['level'], []).append(float(r['mse_change_vs_selected_pct']))
        keys = sorted(set(e1_levels) | set(old_levels), key=lambda x: (len(x), x))
        for k in keys:
            a = e1_levels.get(k)
            b = old_levels.get(k)
            lines.append(f'| {k} | {(f"{st.mean(a):+.2f}% ({len(a)})" if a else "—")} | {(f"{st.mean(b):+.2f}% ({len(b)})" if b else "—")} |')
        lines.append('')
    (BASE / f'06_hyperparameter_study/hp_summary_{sel}.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    for selection in ('test', 'val'):
        if (BASE / f'05_ablation/{selection}_selected/ablation_results_e1.csv').is_file():
            ablation(selection)
        if (BASE / f'06_hyperparameter_study/{selection}_selected/results_e1.csv').is_file():
            hyperparameters(selection)
    print('done')
