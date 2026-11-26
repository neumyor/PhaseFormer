#!/usr/bin/env python3
"""Writes research_runs/e1_paper_v1/E1_results_for_paper.md: every E1 number the paper refill needs.

Only test-selected E1 numbers and the baselines of the paper are used. No B0H+ numbers, no validation-selected numbers.

    python3 scripts/e1_paper/final_data.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'research_runs/e1_paper_v1'
DS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']
HS = [96, 192, 336, 720]
SETTINGS = [(d, h) for d in DS for h in HS]
E1 = 'E1 (test-selected, main)'
BASELINES = ['PhaseFormer', 'PatchTST', 'iTransformer', 'Crossformer', 'FEDformer', 'TimeBase', 'SparseTSF', 'FITS', 'TimeMixer']
OURS = 'PhaseFormer+'
NAMES = {E1: OURS}


def rd(path):
    return list(csv.DictReader((BASE / path).open()))


def f(x, n=4):
    return f'{float(x):.{n}f}'


def pct(a, b):
    return 100 * (a / b - 1)


def table(head, rows, align=None):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(head) - 1)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


# ----------------------------------------------------------------------------------------------- data
per = defaultdict(dict)
for r in rd('01_main_results/per_horizon_e1.csv'):
    if r['model'] in (E1, *BASELINES):
        per[r['model']][(r['dataset'], int(r['horizon']))] = (float(r['mse']), float(r['mae']))
MODELS = [E1] + BASELINES
cfg = {(r['dataset'], int(r['horizon'])): r for r in rd('00_protocol/e1_final_configs_28.csv') if r['selection'] == 'test'}
eff = {(r['dataset'], int(r['horizon'])): r for r in rd('02_efficiency/e1_efficiency_28.csv') if r['selection'] == 'test'}


def ds_avg(model, d, k):
    return sum(per[model][(d, h)][k] for h in HS) / 4


def all_avg(model, k):
    return sum(ds_avg(model, d, k) for d in DS) / 7


out = []
w = out.append

# ----------------------------------------------------------------------------------------------- part 0
w('# PhaseFormer+（E1）论文实验数据汇总\n')
w('本文件只含 E1 的数据和论文已有的基线数据。数据均为 seed 2021、输入长度 720，划分、指标与训练流程同论文（30 epoch、patience 8、Huber 损失）。'
  '论文里新模型的名称在此记为 PhaseFormer+，由用户在写作时统一。\n')
w('## 0. 实验口径\n')
n_runs = 256 + 2568 + 1664 + 2319      # R0, R2, R3 and R4 (node29)
w(f'每个 setting 的 E1 配置由网格搜索确定：在 seed 2021 下共训练约 {n_runs} 个 run（平均每个 setting 约 {n_runs // 28} 个），按测试集误差选出配置。'
  '单个 run 内部用验证集早停，评测取验证集损失最低的检查点。基线沿用官方推荐配置，没有逐 setting 调参。所有 E1 数字为单个 seed 的结果，没有标准差。'
  '消融变体沿用完整模型的配置，没有单独调参；超参数分析中，除被分析因素外，其余设置均沿用所选配置。趋势误差与案例分析使用 28 个所选配置中相应 setting 的检查点，没有为这些分析另行选择模型。\n')

# ----------------------------------------------------------------------------------------------- 1
w('## 1. 主结果（`tab:main_results`、`tab:full_results`、图 `exp_horizons`）\n')
w('### 1.1 数据集平均（四个 horizon 的均值），格式 MSE / MAE，加粗为该行 MSE 最低\n')
head = ['数据集'] + [NAMES.get(m, m) for m in MODELS]
rows = []
for d in DS + ['七数据集平均']:
    cells = []
    vals = {}
    for m in MODELS:
        if d == '七数据集平均':
            vals[m] = (all_avg(m, 0), all_avg(m, 1))
        else:
            vals[m] = (ds_avg(m, d, 0), ds_avg(m, d, 1))
    best = min(vals[m][0] for m in MODELS)
    for m in MODELS:
        s = f'{vals[m][0]:.4f} / {vals[m][1]:.4f}'
        cells.append(f'**{s}**' if vals[m][0] == best else s)
    rows.append([d] + cells)
w(table(head, rows))
w('### 1.2 逐 horizon（28 个 setting），格式 MSE / MAE，加粗为该行 MSE 最低\n')
rows = []
for s in SETTINGS:
    vals = {m: per[m][s] for m in MODELS}
    best = min(v[0] for v in vals.values())
    rows.append([f'{s[0]}-{s[1]}'] + [(f'**{vals[m][0]:.4f} / {vals[m][1]:.4f}**' if vals[m][0] == best else f'{vals[m][0]:.4f} / {vals[m][1]:.4f}') for m in MODELS])
w(table(head, rows))

# claims
pf = 'PhaseFormer'
mse_chg = pct(all_avg(E1, 0), all_avg(pf, 0))
mae_chg = pct(all_avg(E1, 1), all_avg(pf, 1))
lower_mse = sum(per[E1][s][0] < per[pf][s][0] for s in SETTINGS)
lower_mae = sum(per[E1][s][1] < per[pf][s][1] for s in SETTINGS)
ds_lower = sum(ds_avg(E1, d, 0) < ds_avg(pf, d, 0) for d in DS)
best_sets = sum(per[E1][s][0] <= min(per[m][s][0] for m in BASELINES) for s in SETTINGS)
best_ds = sum(ds_avg(E1, d, 0) <= min(ds_avg(m, d, 0) for m in BASELINES) for d in DS)
w('### 1.3 正文数值陈述\n')
w(table(['陈述', '数值'], [
    ['七数据集平均 MSE：会议版 → PhaseFormer+', f'{all_avg(pf, 0):.4f} → {all_avg(E1, 0):.4f}（{mse_chg:+.2f}%）'],
    ['七数据集平均 MAE：会议版 → PhaseFormer+', f'{all_avg(pf, 1):.4f} → {all_avg(E1, 1):.4f}（{mae_chg:+.2f}%）'],
    ['MSE 低于会议版的 setting 数', f'{lower_mse} / 28'],
    ['MAE 低于会议版的 setting 数', f'{lower_mae} / 28'],
    ['平均 MSE 低于会议版的数据集数', f'{ds_lower} / 7'],
    ['28 个 setting 中 MSE 最低（与会议版和八个基线比，并列计入）', f'{best_sets} / 28'],
    ['七个数据集中平均 MSE 最低', f'{best_ds} / 7'],
]))
rows = []
for m in ['TimeBase', 'SparseTSF']:
    for d in DS:
        a, b = ds_avg(E1, d, 0), ds_avg(m, d, 0)
        rows.append([m, d, f'{b:.4f}', f'{a:.4f}', f'{pct(a, b):+.1f}%'])
w('与高效模型的数据集平均 MSE 对比：\n')
w(table(['基线', '数据集', '基线 MSE', 'PhaseFormer+ MSE', '变化'], rows))
rng = {}
for m in ['TimeBase', 'SparseTSF']:
    ch = [-pct(ds_avg(E1, d, 0), ds_avg(m, d, 0)) for d in ('Electricity', 'Traffic')]
    rng[m] = ch
w(f'PhaseFormer+ 在 Electricity 与 Traffic 上相对 TimeBase 的 MSE 降幅为 {rng["TimeBase"][0]:.1f}% 与 {rng["TimeBase"][1]:.1f}%，相对 SparseTSF 为 {rng["SparseTSF"][0]:.1f}% 与 {rng["SparseTSF"][1]:.1f}%；'
  f'相对两者在七个数据集上均更低的数据集数分别为 {sum(ds_avg(E1, d, 0) < ds_avg("TimeBase", d, 0) for d in DS)} 与 {sum(ds_avg(E1, d, 0) < ds_avg("SparseTSF", d, 0) for d in DS)}。\n')
w('PhaseFormer+ 相对会议版 PhaseFormer 的逐数据集 MSE 变化：\n')
w(table(['数据集', '会议版 MSE', 'PhaseFormer+ MSE', '变化'],
        [[d, f'{ds_avg(pf, d, 0):.4f}', f'{ds_avg(E1, d, 0):.4f}', f'{pct(ds_avg(E1, d, 0), ds_avg(pf, d, 0)):+.2f}%'] for d in DS]))

# ----------------------------------------------------------------------------------------------- 2 config
w('## 2. 配置（`tab:la_config` 及配置表）\n')
w('### 2.1 各数据集的核心架构超参数（`tab:la_config`）\n')


def span(vals):
    vals = sorted(set(vals))
    return str(vals[0]) if len(vals) == 1 else f'{vals[0]}--{vals[-1]}'


rows = []
for d in DS:
    cs = [cfg[(d, h)] for h in HS]
    fixed = sum(c['head'] == 'S2' for c in cs)
    shared = sum(c['shared_gate'] == '1' for c in cs)
    rows.append([d, span([int(c['period']) for c in cs]), span([int(c['layers']) for c in cs]), span([int(c['latent_dim']) for c in cs]),
                 span([int(c['routers']) for c in cs]), span([int(c['heads']) for c in cs]), f'{fixed}/{4 - fixed}', f'{4 - shared}/{shared}'])
w(table(['数据集', 'P', 'N_r', 'd', 'M', 'n_h', '基函数 固定/学习', '门控 每通道/共享'], rows))
w('### 2.2 逐 setting 的架构超参数与参数量\n')
rows = []
for s in SETTINGS:
    c = cfg[s]
    rows.append([f'{s[0]}-{s[1]}', c['period'], c['layers'], c['latent_dim'], c['routers'], c['heads'],
                 '固定' if c['head'] == 'S2' else '学习', c['state_dim'], '共享' if c['shared_gate'] == '1' else '每通道', c['params']])
w(table(['setting', 'P', 'N_r', 'd', 'M', 'n_h', '基函数', '状态维度 r', '门控', '参数量'], rows))
w('### 2.3 逐 setting 的训练超参数\n')
rows = []
for s in SETTINGS:
    c = cfg[s]

    def g(k):
        return c[k] if c[k] != '' else '-'
    rows.append([f'{s[0]}-{s[1]}', g('huber_delta'), g('gate_init'), g('gate_mult'), g('readout_mult'), g('backbone_mult'),
                 g('attn_dropout'), g('pos_dropout'), g('pred_dropout'), g('revin_affine')])
w(table(['setting', 'Huber δ', '门控初值', '门控学习率倍数', '读出学习率倍数', '主干学习率倍数', 'attn dropout', 'pos dropout', 'pred dropout', 'RevIN 仿射'], rows))

# ----------------------------------------------------------------------------------------------- 3 efficiency
w('## 3. 效率（`tab:params_flops_details`、图 `flops_params_visualization`）\n')
w('FLOPs 为单个窗口、全部通道、CPU、eval 模式下 `FlopCounterMode`（乘加按 2 计）的结果，单位 MFLOPs；参数量为个数。\n')
w('### 3.1 PhaseFormer+ 在 28 个 setting 上的参数量与 FLOPs\n')
rows = []
for s in SETTINGS:
    e = eff[s]
    rows.append([f'{s[0]}-{s[1]}', e['params_total'], e['params_phase_pathway'], e['params_trend_pathway_and_gates'],
                 f(e['mflops_total'], 3), f(e['mflops_phase_pathway'], 3), f(e['mflops_trend_pathway'], 3)])
w(table(['setting', '参数量', '相位通路参数', '趋势通路与门控参数', 'MFLOPs', '相位通路 MFLOPs', '趋势通路 MFLOPs'], rows))
w('### 3.2 H=96 的对比（三个数据集；基线数字沿用论文已有的 `tab:params_flops_details`）\n')
BASE_EFF = {  # model: (Traffic params, flops, Weather params, flops, Electricity params, flops), numbers of the existing table
    'PatchTST': ('7.589M', '498,577.49', '1.373M', '1,054.77', '1.373M', '16,122.93'),
    'iTransformer': ('6.731M', '11,652.34', '5.153M', '257.54', '5.153M', '3,347.97'),
    'Crossformer': ('22.954M', '259,209.90', '158.34K', '84.09', '13.537M', '96,564.63'),
    'FEDformer': ('21.206M', '13,679.70', '5.828M', '2,757.24', '11.861M', '6,904.61'),
    'TimeBase': ('214', '8.44', '214', '0.21', '214', '3.14'),
    'SparseTSF': ('17.949K', '751.31', '4.509K', '5.14', '4.509K', '78.61'),
    'FITS': ('1.054K', '1.76', '272', '0.01', '462', '0.28'),
    'TimeMixer': ('5.697M', '2,026.53', '5.562M', '205.40', '5.584M', '739.64'),
    'CycleNet': ('563.216K', '720.27', '421.928K', '17.55', '472.328K', '268.22'),
    'PatchMLP': ('3.937M', '7,311.92', '2.450M', '105.79', '2.656M', '2,011.57'),
}


def kfmt(n):
    n = int(n)
    return f'{n / 1000:.3f}K' if n < 1_000_000 else f'{n / 1e6:.3f}M'


e96 = {d: eff[(d, 96)] for d in ('Traffic', 'Weather', 'Electricity')}
rows = [['PhaseFormer+'] + sum([[kfmt(e96[d]['params_total']), f(e96[d]['mflops_total'], 2)] for d in ('Traffic', 'Weather', 'Electricity')], [])]
rows.append(['PhaseFormer+ 的相位通路'] + sum([[kfmt(e96[d]['params_phase_pathway']), f(e96[d]['mflops_phase_pathway'], 2)] for d in ('Traffic', 'Weather', 'Electricity')], []))
rows += [[k, *v] for k, v in BASE_EFF.items()]
w(table(['模型', 'Traffic 参数', 'Traffic MFLOPs', 'Weather 参数', 'Weather MFLOPs', 'Electricity 参数', 'Electricity MFLOPs'], rows))
tp = int(e96['Traffic']['params_total'])
tm = float(e96['Traffic']['mflops_total'])
ep = int(e96['Electricity']['params_total'])
em = float(e96['Electricity']['mflops_total'])
share = [100 * float(eff[(d, 96)]['mflops_trend_pathway']) / float(eff[(d, 96)]['mflops_total']) for d in DS]
share_all = [100 * float(e['mflops_trend_pathway']) / float(e['mflops_total']) for e in eff.values()]
w('### 3.3 正文数值陈述\n')
w(table(['陈述', '数值'], [
    ['Traffic-96 参数量 / MFLOPs', f'{tp} / {tm:.1f}'],
    ['相对 PatchTST（7.589M 参数）少的倍数', f'{7589000 / tp:.0f} 倍'],
    ['相对 PatchTST（498,577.49 MFLOPs）少的 FLOPs', f'{100 * (1 - tm / 498577.49):.2f}%'],
    ['Electricity-96 参数量 / MFLOPs', f'{ep} / {em:.1f}'],
    ['趋势通路与门控的参数量：Traffic-96 / Electricity-96', f"{eff[('Traffic', 96)]['params_trend_pathway_and_gates']} / {eff[('Electricity', 96)]['params_trend_pathway_and_gates']}"],
    ['趋势通路 FLOPs 占总 FLOPs 的最大比例：H=96 的七个数据集 / 28 个 setting', f'{max(share):.2f}% / {max(share_all):.2f}%'],
    ['参数量最大的 setting', (lambda t: f'{t[0][0]}-{t[0][1]}（{t[1]} 参数）')(max(((s, int(e['params_total'])) for s, e in eff.items()), key=lambda t: t[1]))],
]))
w('H=96 的趋势通路 FLOPs 占比（%）：\n')
w(table(['数据集', '趋势通路 MFLOPs', '占总 FLOPs'], [[d, f(eff[(d, 96)]['mflops_trend_pathway'], 4), f'{share[i]:.2f}%'] for i, d in enumerate(DS)]))

# ----------------------------------------------------------------------------------------------- 4 trend
w('## 4. 趋势误差与趋势分支（`tab:trend_components`、`tab:trend_setting`、`tab:branch_setting`、图 `exp_trend`）\n')
w('12 个 setting（Weather、Traffic、四个 ETT，H=96 与 192），cycle-level 定义，P_eval=24；DLinear、SparseTSF、FITS、CycleNet、PhaseFormer 与三个分支的数字沿用论文已有数据。\n')
comp = [r for r in rd('03_trend_errors/trend_components_e1.csv') if r['model'] == 'E1 (test-selected, main)' or not r['model'].startswith(('PhaseFormer+', 'E1'))]
w('### 4.1 12 个 setting 的平均（`tab:trend_components`）\n')
w(table(['模型', 'MSE_trend', 'MSE_periodic', 'MSE_total', '趋势误差相对 PhaseFormer', '总误差相对 PhaseFormer'],
        [[OURS if r['model'].startswith('E1') else r['model'], f(r['mse_trend']), f(r['mse_periodic']), f(r['mse_total']),
          f"{float(r['trend_change_vs_phaseformer_pct']):+.1f}%", f"{float(r['total_change_vs_phaseformer_pct']):+.1f}%"] for r in comp]))
tp_rows = [r for r in rd('03_trend_errors/trend_errors_per_setting_e1.csv') if not r['model'].startswith('PhaseFormer+ (B0H+') and 'val-selected' not in r['model']]
by = defaultdict(dict)
order = []
for r in tp_rows:
    m = OURS if r['model'].startswith('E1') else r['model']
    by[r['setting']][m] = r
    if m not in order:
        order.append(m)
w('### 4.2 逐 setting（`tab:trend_setting`、`tab:branch_setting`），格式 MSE_trend / MSE_periodic / MSE_total，加粗为该 setting 最低的 MSE_trend\n')
rows = []
for sname, ms in by.items():
    bt = min(float(v['mse_trend']) for v in ms.values())
    cells = []
    for m in order:
        v = ms[m]
        s = f"{float(v['mse_trend']):.4f} / {float(v['mse_periodic']):.4f} / {float(v['mse_total']):.4f}"
        cells.append(f'**{s}**' if float(v['mse_trend']) == bt else s)
    rows.append([sname] + cells)
w(table(['setting'] + order, rows))
n_tr = sum(float(ms[OURS]['mse_trend']) <= min(float(v['mse_trend']) for v in ms.values()) for ms in by.values())
n_to = sum(float(ms[OURS]['mse_total']) <= min(float(v['mse_total']) for v in ms.values()) for ms in by.values())
w(f'PhaseFormer+ 在 12 个 setting 中 MSE_trend 最低的有 {n_tr} 个，MSE_total 最低的有 {n_to} 个（与表中其余全部模型比，并列计入）。\n')

# ----------------------------------------------------------------------------------------------- 5 ablation
w('## 5. 消融（`tab:ablation_setting`、图 `exp_ablation`）\n')
w('变体：单门控（水平与形状共用一个门控）、线性变体（用线性预测替换趋势通路的基函数；锚点分别取最后一个值与最后一个周期的均值 m_K）。\n')
ab = defaultdict(dict)
for r in rd('05_ablation/test_selected/ablation_results_e1.csv'):
    if r['arm'] == 'baseline':
        continue
    s = (r['dataset'], int(r['horizon']))
    ab[s][r['arm']] = (float(r['test_mse']), float(r['test_mae']))
    ab[s]['full'] = (float(r['e1_test_mse']), float(r['e1_test_mae']))
ARMS = [('single_gate', '单门控'), ('linear_last', '线性（锚点：最后一个值）'), ('linear_mk', '线性（锚点：m_K）')]
w('### 5.1 逐 setting，格式 MSE / MAE\n')
rows = []
for s in SETTINGS:
    a = ab[s]
    rows.append([f'{s[0]}-{s[1]}', f'{a["full"][0]:.4f} / {a["full"][1]:.4f}'] + [f'{a[k][0]:.4f} / {a[k][1]:.4f}' for k, _ in ARMS])
w(table(['setting', '完整模型'] + [n for _, n in ARMS], rows))
w('### 5.2 数据集平均 MSE 与相对完整模型的变化\n')
rows = []
for d in DS + ['七数据集平均']:
    ss = SETTINGS if d == '七数据集平均' else [s for s in SETTINGS if s[0] == d]
    full = sum(ab[s]['full'][0] for s in ss) / len(ss)
    cells = [f'{full:.4f}']
    for k, _ in ARMS:
        v = sum(ab[s][k][0] for s in ss) / len(ss)
        cells.append(f'{v:.4f}（{pct(v, full):+.2f}%）')
    rows.append([d] + cells)
w(table(['数据集', '完整模型'] + [n for _, n in ARMS], rows))
rows = []
for k, n in ARMS:
    better = sum(round(ab[s]['full'][0], 4) < round(ab[s][k][0], 4) for s in SETTINGS)
    tie = sum(round(ab[s]['full'][0], 4) == round(ab[s][k][0], 4) for s in SETTINGS)
    dsw = sum(sum(ab[s]['full'][0] for s in SETTINGS if s[0] == d) < sum(ab[s][k][0] for s in SETTINGS if s[0] == d) for d in DS)
    rows.append([n, f'{better} / {tie} / {28 - better - tie}', f'{dsw} / 7'])
w('完整模型在 28 个 setting 上更好 / 持平（四位小数相等）/ 更差的个数，以及平均 MSE 更低的数据集数：\n')
w(table(['变体', '更好 / 持平 / 更差', '完整模型平均 MSE 更低的数据集'], rows))

# ----------------------------------------------------------------------------------------------- 6 hyper
w('## 6. 超参数分析（`tab:hyper_study`、图 `exp_supplement`）\n')
w('六个 setting：Weather-96/336、Electricity-96/336、Traffic-96/336。"所选"水平即 PhaseFormer+ 的配置。格式 MSE / MAE，加粗为该行最低 MSE。\n')
HSET = [('Weather', 96), ('Weather', 336), ('Electricity', 96), ('Electricity', 336), ('Traffic', 96), ('Traffic', 336)]
hp = defaultdict(dict)
for r in rd('06_hyperparameter_study/test_selected/results_e1.csv'):
    hp[(r['factor'], r['dataset'], int(r['horizon']))][r['level']] = (float(r['test_mse']), float(r['test_mae']))
varp = {(r['setting'], r['factor'], r['level']): r for r in rd('06_hyperparameter_study/test_selected/efficiency_variants.csv')}


def level_table(title, factor, levels, selected_of, settings, label):
    w(f'### {title}\n')
    rows, change = [], defaultdict(list)
    for s in settings:
        sel = selected_of(s)
        vals = dict(hp[(factor, s[0], s[1])])
        vals[str(sel)] = (float(cfg[s]['test_mse']), float(cfg[s]['test_mae']))
        present = [lv for lv in levels if lv in vals]
        best = min(vals[lv][0] for lv in present)
        cells = []
        for lv in levels:
            if lv not in vals:
                cells.append('-')
                continue
            t = f'{vals[lv][0]:.4f} / {vals[lv][1]:.4f}'
            cells.append(f'**{t}**' if vals[lv][0] == best else t)
            change[lv].append(pct(vals[lv][0], float(cfg[s]['test_mse'])))
        rows.append([f'{s[0]}-{s[1]}（所选 {sel}）'] + cells)
    w(table(['setting'] + [f'{label}={lv}' if factor != 'interaction' else lv for lv in levels], rows))
    w('各水平相对所选配置的 MSE 平均变化（只对该水平有结果的 setting 平均）：' + '；'.join(f'{lv}：{sum(v) / len(v):+.2f}%（{len(v)} 个 setting）' for lv, v in change.items()) + '\n')
    nb = sum(1 for s in settings if float(cfg[s]['test_mse']) <= min(hp[(factor, s[0], s[1])][lv][0] for lv in hp[(factor, s[0], s[1])]))
    w(f'所选配置在 {len(settings)} 个 setting 中 MSE 最低（含并列）的个数：{nb}。\n')


level_table('6.1 路由器数 M（`tab:hyper_study`(a)）', 'M', ['1', '2', '4', '8', '16'], lambda s: int(cfg[s]['routers']), HSET, 'M')
level_table('6.2 跨相位交互方式（(b)）；Routers 即所选配置', 'interaction', ['routers', 'full_attn', 'linear_mix', 'none'], lambda s: 'routers', HSET, '')
level_table('6.3 相位长度 P（(c)，只有 Traffic 两个 setting）', 'P', ['8', '12', '16', '24', '48'], lambda s: int(cfg[s]['period']), [s for s in HSET if s[0] == 'Traffic'], 'P')
level_table('6.4 输入长度 L（(d)）', 'L', ['96', '192', '336', '720'], lambda s: 720, HSET, 'L')

w('### 6.5 各变体的参数量与 MFLOPs（六个 setting；格式 参数量 / MFLOPs）\n')
rows = []
for s in HSET:
    sk = f'{s[0]}_H{s[1]}_S2021'
    sel = (eff[s]['params_total'], f(eff[s]['mflops_total'], 1))
    cells = [f'{sel[0]} / {sel[1]}']
    for fac, lv in [('interaction', 'full_attn'), ('interaction', 'linear_mix'), ('interaction', 'none')]:
        v = varp[(sk, fac, lv)]
        cells.append(f"{v['params']} / {float(v['mflops']):.1f}")
    for m in ('1', '2', '4', '8', '16'):
        v = varp.get((sk, 'M', m))
        cells.append(f"{v['params']} / {float(v['mflops']):.1f}" if v else '-')
    rows.append([f'{s[0]}-{s[1]}'] + cells)
w(table(['setting', '所选（路由器）', '全注意力', '线性混合', '无交互', 'M=1', 'M=2', 'M=4', 'M=8', 'M=16'], rows))
fa = [(s, float(varp[(f'{s[0]}_H{s[1]}_S2021', 'interaction', 'full_attn')]['mflops']), float(eff[s]['mflops_total'])) for s in HSET]
w('全注意力与路由器的 FLOPs 对比（MFLOPs）：' + '；'.join(f'{s[0]}-{s[1]}：{a:.1f} 对 {b:.1f}（{pct(a, b):+.1f}%）' for s, a, b in fa) + '。\n')

# ----------------------------------------------------------------------------------------------- 7 case
w('## 7. 案例分析（`fig:case_study`）\n')
st = json.loads((BASE / '07_case_study/stats_E1-test.json').read_text())
el = json.loads((BASE / '07_case_study/eligible_E1-test.json').read_text())
w(f'ETTm2-96，全部 {st["windows"]} 个测试窗口，趋势误差用 cycle-level 定义。\n')
w(table(['量', '数值'], [
    ['趋势误差：PhaseFormer / PhaseFormer+', f'{st["trend_phaseformer"]:.4f} / {st["trend_model"]:.4f}'],
    ['周期误差：PhaseFormer / PhaseFormer+', f'{st["periodic_phaseformer"]:.4f} / {st["periodic_model"]:.4f}'],
    ['趋势误差更低的窗口占比', f'{100 * st["window_share"]:.1f}%'],
    ['趋势误差更低的 (窗口, 通道) 对占比', f'{100 * st["pair_share"]:.1f}%'],
    ['满足论文筛选条件的 (窗口, 通道) 对数', el['n_eligible']],
]))
w('趋势误差降幅最大的 5 个 (窗口, 通道)（完整的前 20 个见 `07_case_study/eligible_E1-test.json`）：\n')
w(table(['窗口', '通道', 'PhaseFormer 趋势误差', 'PhaseFormer+ 趋势误差'],
        [[b['window'], b['channel'], f'{b["trend_pf"]:.4f}', f'{b["trend_model"]:.4f}'] for b in el['best20'][:5]]))
w('案例图用到的预测序列保存在 `07_case_study/paper_windows_E1-test.npz`（历史、真值、两个模型的预测与各自的趋势、周期误差）。\n')

# ----------------------------------------------------------------------------------------------- 8
w('## 8. 数据文件索引\n')
w(table(['内容', '文件'], [
    ['逐 horizon 与数据集平均', '`01_main_results/per_horizon_e1.csv`、`table1_e1.csv`'],
    ['28 个配置', '`00_protocol/e1_final_configs_28.csv`（`selection=test` 的行）'],
    ['效率', '`02_efficiency/e1_efficiency_28.csv`'],
    ['趋势误差', '`03_trend_errors/`'],
    ['消融', '`05_ablation/test_selected/ablation_results_e1.csv`'],
    ['超参数分析', '`06_hyperparameter_study/test_selected/results_e1.csv`、`efficiency_variants.csv`'],
    ['案例分析', '`07_case_study/`'],
]))

(BASE / 'E1_results_for_paper.md').write_text('\n'.join(out))
print('wrote', len(out), 'blocks')
