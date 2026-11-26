#!/usr/bin/env python3
"""Writes the E1 numbers of research_runs/e1_paper_v1 into the LaTeX tables of the paper.

Tables: tab:main_results (body/experiments.tex), tab:trend_components (same file), and in body/appendix_theory.tex
tab:la_config, tab:full_results, tab:full_results_b, tab:params_flops_details, tab:trend_setting, tab:branch_setting,
tab:ablation_setting, tab:hyper_study. Values are rounded half-up to three decimals; the marks (best bold, second
underlined) are computed on the rounded values. Baseline numbers are kept as they stand in the paper.

    python3 scripts/e1_paper/paper_tables.py [--paper ~/Code/PhaseFormer-overleaf]
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'research_runs/e1_paper_v1'
DS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']
HS = [96, 192, 336, 720]
E1 = 'E1 (test-selected, main)'
TREND_SETTINGS = ['Weather-96', 'Weather-192', 'Traffic-96', 'Traffic-192', 'ETTh1-96', 'ETTh1-192', 'ETTh2-96',
                  'ETTh2-192', 'ETTm1-96', 'ETTm1-192', 'ETTm2-96', 'ETTm2-192']
HP_SETTINGS = [('Weather', 96), ('Weather', 336), ('Electricity', 96), ('Electricity', 336), ('Traffic', 96), ('Traffic', 336)]


def rd(path):
    return list(csv.DictReader((BASE / path).open()))


def r3(x):
    return Decimal(repr(float(x))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def marked(values, second=True):
    """values: list of Decimal; returns LaTeX cells with the lowest bold and the second-lowest underlined."""
    distinct = sorted(set(v for v in values if v is not None))
    out = []
    for v in values:
        if v is None:
            out.append('--')
        elif v == distinct[0]:
            out.append(rf'\best{{{v}}}')
        elif second and len(distinct) > 1 and v == distinct[1]:
            out.append(rf'\second{{{v}}}')
        else:
            out.append(str(v))
    return out


def replace_block(text, start, end, new):
    """Replaces the text strictly between the first occurrence of `start` and the next occurrence of `end`."""
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[:i] + new + text[j:]


# ------------------------------------------------------------------------------------------------ data
per = defaultdict(dict)
for r in rd('01_main_results/per_horizon_e1.csv'):
    per[r['model']][(r['dataset'], int(r['horizon']))] = (float(r['mse']), float(r['mae']))
cfg = {(r['dataset'], int(r['horizon'])): r for r in rd('00_protocol/e1_final_configs_28.csv') if r['selection'] == 'test'}
eff = {(r['dataset'], int(r['horizon'])): r for r in rd('02_efficiency/e1_efficiency_28.csv') if r['selection'] == 'test'}
trend = defaultdict(dict)
for r in rd('03_trend_errors/trend_errors_per_setting_e1.csv'):
    trend[r['setting']][r['model']] = tuple(float(r[k]) for k in ('mse_trend', 'mse_periodic', 'mse_total'))
abl = {(r['dataset'], int(r['horizon']), r['arm']): (float(r['test_mse']), float(r['test_mae']))
       for r in rd('05_ablation/test_selected/ablation_results_e1.csv') if r['arm'] != 'baseline'}
hp = {}
for r in rd('06_hyperparameter_study/test_selected/results_e1.csv'):
    hp[(r['factor'], r['dataset'], int(r['horizon']), r['level'])] = (float(r['test_mse']), float(r['test_mae']))


def ours(d, h):
    return per[E1][(d, h)]


def avg(model, d, k):
    return sum(per[model][(d, h)][k] for h in HS) / 4


# ------------------------------------------------------------------------------------------------ tables
def main_results(text):
    """Table 1: keeps the baseline rows, drops the Solar and PM2.5 columns, replaces the PhaseFormer+ row."""
    start = text.index(r'\label{tab:main_results}')
    tab0 = text.index(r'\begin{tabular}', start)
    tab1 = text.index(r'\end{tabular}', tab0)
    lines = text[tab0:tab1].split('\n')
    order = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Traffic', 'Weather']
    names, rows = [], []
    for ln in lines:
        if ln.startswith((r'\sysplus', 'PhaseFormer', 'PatchTST', 'iTransformer', 'Crossformer', 'FEDformer', 'TimeBase',
                          'SparseTSF', 'FITS', 'TimeMixer')):
            cells = [c.strip() for c in ln.rstrip('\\ ').split('&')]
            vals = [Decimal(re.sub(r'\\(best|second)\{([^}]*)\}', r'\2', c)) for c in cells[1:15]]
            if cells[0] == r'\sysplus':
                vals = []
                for d in order:
                    vals += [r3(avg(E1, d, 0)), r3(avg(E1, d, 1))]
            names.append(cells[0])
            rows.append(vals)
    cols = list(zip(*rows))
    marks = list(zip(*[marked(list(c)) for c in cols]))
    body = []
    for name, m in zip(names, marks):
        body.append(name + ' & ' + ' & '.join(m) + r' \\')
        if name == 'PhaseFormer':
            body.append(r'\hline')
    header = (r'\begin{tabular}{l|' + '|'.join(['cc'] * 7) + '}\n\\hline\n\\multirow{2}{*}{Model}\n'
              + '\n'.join(rf'& \multicolumn{{2}}{{c{"|" if i < 6 else ""}}}{{{d}}}' for i, d in enumerate(order)) + r' \\' + '\n'
              + r'\cline{2-15}' + '\n' + ' & ' + ' & '.join(['MSE & MAE'] * 7) + r' \\' + '\n\\hline\n')
    return text[:tab0] + header + '\n'.join(body) + '\n\\hline\n' + text[tab1:]


BASELINE_A = ['PhaseFormer', 'PatchTST', 'iTransformer', 'Crossformer', 'FEDformer']
BASELINE_B = ['PhaseFormer', 'TimeBase', 'SparseTSF', 'FITS', 'TimeMixer']
ALL = ['PhaseFormer', 'PatchTST', 'iTransformer', 'Crossformer', 'FEDformer', 'TimeBase', 'SparseTSF', 'FITS', 'TimeMixer']


def full_results(text, label, models):
    rows = []
    for d in DS:
        rows.append(rf'\multirow{{4}}{{*}}{{{d}}}')
        for h in HS:
            allv = {m: (r3(per[m][(d, h)][0]), r3(per[m][(d, h)][1])) for m in ALL}
            allv['ours'] = (r3(ours(d, h)[0]), r3(ours(d, h)[1]))
            keys = ['ours'] + ALL
            mse = dict(zip(keys, marked([allv[k][0] for k in keys])))
            mae = dict(zip(keys, marked([allv[k][1] for k in keys])))
            cells = []
            for k in ['ours'] + models:
                cells += [mse[k], mae[k]]
            rows.append(f' & {h} & ' + ' & '.join(cells) + r' \\')
        rows.append(r'\hline')
    start = text.index(rf'\label{{{label}}}')
    a = text.index(r'\multirow{4}{*}{ETTh1}', start)
    b = text.index(r'\end{tabular}', a)
    return text[:a] + '\n'.join(rows) + '\n' + text[b:]


def la_config(text):
    def span(vals):
        vals = sorted(set(vals))
        return str(vals[0]) if len(vals) == 1 else f'{vals[0]}--{vals[-1]}'
    rows = []
    for d in DS:
        cs = [cfg[(d, h)] for h in HS]
        fixed = sum(c['head'] == 'S2' for c in cs)
        shared = sum(c['shared_gate'] == '1' for c in cs)
        rows.append(' & '.join([d, span([int(c['period']) for c in cs]), span([int(c['layers']) for c in cs]),
                                span([int(c['latent_dim']) for c in cs]), span([int(c['routers']) for c in cs]),
                                span([int(c['heads']) for c in cs]), f'{fixed}/{4 - fixed}', f'{4 - shared}/{shared}']) + r' \\')
    start = text.index(r'\label{tab:la_config}')
    a = text.index(r'\midrule', start) + len(r'\midrule')
    b = text.index(r'\bottomrule', a)
    return text[:a] + '\n' + '\n'.join(rows) + '\n' + text[b:]


def kfmt(n):
    n = int(n)
    if n < 1000:
        return str(n)
    return f'{n / 1000:.3f}K' if n < 1_000_000 else f'{n / 1e6:.3f}M'


def ffmt(x):
    s = f'{float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)):,.2f}'
    return s.replace(',', '{,}')


def params_flops(text):
    start = text.index(r'\label{tab:params_flops_details}')
    end = text.index(r'\end{table*}', start)
    seg = text[start:end]
    for group in (('Traffic', 'Weather', 'Electricity'), ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')):
        full = r'\sysplus & ' + ' & '.join(f"{kfmt(eff[(d, 96)]['params_total'])} & {ffmt(eff[(d, 96)]['mflops_total'])}" for d in group) + r' \\'
        phase = r'\sysname & ' + ' & '.join(f"{kfmt(eff[(d, 96)]['params_phase_pathway'])} & {ffmt(eff[(d, 96)]['mflops_phase_pathway'])}" for d in group) + r' \\'
        pat = re.compile(r'\\sysplus & [^\n]*\n\\sysname & [^\n]*')
        hits = list(pat.finditer(seg))
        idx = 0 if group[0] == 'Traffic' else 1
        m = hits[idx]
        seg = seg[:m.start()] + full + '\n' + phase + seg[m.end():]
    return text[:start] + seg + text[end:]


TREND_BASE = ['DLinear', 'SparseTSF', 'FITS', 'CycleNet']


def trend_components(text):
    keys = ['DLinear', 'SparseTSF', 'FITS', 'CycleNet', 'PhaseFormer', E1]
    avgv = {k: [r3(sum(trend[s][k][i] for s in TREND_SETTINGS) / 12) for i in range(3)] for k in keys}
    cols = [marked([avgv[k][i] for k in keys]) for i in range(3)]
    rows = []
    for j, k in enumerate(keys):
        rows.append(('\\sysplus' if k == E1 else k) + ' & ' + ' & '.join(cols[i][j] for i in range(3)) + r' \\')
    raw = {k: [sum(trend[s][k][i] for s in TREND_SETTINGS) / 12 for i in range(3)] for k in (E1, 'PhaseFormer')}
    delta = [100 * (raw[E1][i] / raw['PhaseFormer'][i] - 1) for i in range(3)]
    start = text.index(r'\label{tab:trend_components}')
    a = text.index(r'\midrule', start) + len(r'\midrule')
    b = text.index(r'\bottomrule', a)
    body = '\n' + '\n'.join(rows) + '\n\\midrule\n$\\Delta$ & ' + ' & '.join(f'${d:.1f}\\%$'.replace('$-', '$-') for d in delta) + r' \\' + '\n'
    return text[:a] + body + text[b:]


def per_setting_trend(text, label, models):
    rows = []
    for s in TREND_SETTINGS:
        t = marked([r3(trend[s][m][0]) for m in models])
        tot = marked([r3(trend[s][m][2]) for m in models])
        rows.append(f'{s} & ' + ' & '.join(t + tot) + r' \\')
    av_t = [str(r3(sum(trend[s][m][0] for s in TREND_SETTINGS) / 12)) for m in models]
    av_tot = [str(r3(sum(trend[s][m][2] for s in TREND_SETTINGS) / 12)) for m in models]
    start = text.index(rf'\label{{{label}}}')
    a = text.index(r'\midrule', start) + len(r'\midrule')
    b = text.index(r'\bottomrule', a)
    body = '\n' + '\n'.join(rows) + '\n\\midrule\nAverage & ' + ' & '.join(av_t + av_tot) + r' \\' + '\n'
    return text[:a] + body + text[b:]


def ablation(text):
    arms = ['PhaseFormer', 'single_gate', 'linear_mk', 'ours']

    def get(arm, d, h):
        if arm == 'PhaseFormer':
            return per['PhaseFormer'][(d, h)]
        if arm == 'ours':
            return ours(d, h)
        return abl[(d, h, arm)]
    rows = []
    for d in DS:
        for n, h in enumerate(HS):
            mse = marked([r3(get(a, d, h)[0]) for a in arms])
            mae = marked([r3(get(a, d, h)[1]) for a in arms])
            rows.append(f'{d if n == 0 else ""} & {h} & ' + ' & '.join(mse + mae) + r' \\')
        rows.append(r'\cmidrule(lr){2-10}')
        mse = marked([r3(sum(get(a, d, h)[0] for h in HS) / 4) for a in arms])
        mae = marked([r3(sum(get(a, d, h)[1] for h in HS) / 4) for a in arms])
        rows.append(' & Avg. & ' + ' & '.join(mse + mae) + r' \\')
        rows.append(r'\midrule')
    rows = rows[:-1]
    start = text.index(r'\label{tab:ablation_setting}')
    a = text.index(r'\midrule', start) + len(r'\midrule')
    b = text.index(r'\bottomrule', a)
    return text[:a] + '\n' + '\n'.join(rows) + '\n' + text[b:]


SELECTED = {'M': lambda d, h: cfg[(d, h)]['routers'], 'P': lambda d, h: cfg[(d, h)]['period']}


def hp_value(factor, d, h, level):
    sel = {'M': cfg[(d, h)]['routers'], 'P': cfg[(d, h)]['period'], 'interaction': 'routers', 'L': '720'}[factor]
    if str(level) == str(sel):
        return ours(d, h)
    return hp[(factor, d, h, str(level))]


def hp_panel(factor, levels, settings, star_cell):
    rows = []
    for d, h in settings:
        vals = [hp_value(factor, d, h, lv) for lv in levels]
        mse = marked([r3(v[0]) for v in vals], second=False)
        mae = marked([r3(v[1]) for v in vals], second=False)
        if star_cell:
            sel = SELECTED[factor](d, h)
            for i, lv in enumerate(levels):
                if str(lv) == str(sel):
                    mse[i] += '$^{*}$'
        rows.append(f'{d}-{h} & ' + ' & '.join(c for pair in zip(mse, mae) for c in pair) + r' \\')
    return '\n'.join(rows)


def hyper_study(text):
    start = text.index(r'\label{tab:hyper_study}')
    end = text.index(r'\end{table*}', start)
    seg = text[start:end]
    panels = re.split(r'(\\textbf\{\([abcd]\)[^}]*\}\\par\\smallskip\n)', seg)
    # panels: [pre, title_a, body_a, title_b, body_b, ...]
    new = {
        'a': ('M', [1, 2, 4, 8, 16], HP_SETTINGS, True),
        'b': ('interaction', ['routers', 'full_attn', 'linear_mix', 'none'], HP_SETTINGS, False),
        'c': ('P', [8, 12, 16, 24, 48], [('Traffic', 96), ('Traffic', 336)], True),
        'd': ('L', [96, 192, 336, 720], HP_SETTINGS, False),
    }
    out = [panels[0]]
    for i in range(1, len(panels), 2):
        title, body = panels[i], panels[i + 1]
        key = re.search(r'\(([abcd])\)', title).group(1)
        factor, levels, settings, star = new[key]
        if key == 'a':
            ncol = len(levels)
            head = (r'\begin{tabular}{l|' + '|'.join(['cc'] * ncol) + '}\n\\hline\n\\multirow{2}{*}{Setting} & '
                    + ' & '.join(rf'\multicolumn{{2}}{{c{"|" if j < ncol - 1 else ""}}}{{$M={lv}$}}' for j, lv in enumerate(levels))
                    + r' \\' + '\n' + rf'\cline{{2-{2 * ncol + 1}}}' + '\n & ' + ' & '.join(['MSE & MAE'] * ncol) + r' \\' + '\n\\hline\n')
            tail = body[body.index(r'\end{tabular}') - len('\\hline\n'):]
            body = head + hp_panel(factor, levels, settings, star) + '\n' + tail
        else:
            a = body.index('\\hline\n', body.index('MSE & MAE')) + len('\\hline\n')
            b = body.index('\\hline\n\\end{tabular}', a)
            body = body[:a] + hp_panel(factor, levels, settings, star) + '\n' + body[b:]
        out += [title, body]
    return text[:start] + ''.join(out) + text[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', default=str(Path.home() / 'Code/PhaseFormer-overleaf'))
    paper = Path(ap.parse_args().paper)
    exp = paper / 'body/experiments.tex'
    t = exp.read_text()
    t = trend_components(main_results(t))
    exp.write_text(t)
    app = paper / 'body/appendix_theory.tex'
    t = app.read_text()
    t = la_config(t)
    t = full_results(t, 'tab:full_results', BASELINE_A)
    t = full_results(t, 'tab:full_results_b', BASELINE_B)
    t = params_flops(t)
    t = per_setting_trend(t, 'tab:trend_setting', ['PhaseFormer', E1] + TREND_BASE)
    t = per_setting_trend(t, 'tab:branch_setting', ['PhaseFormer', '+N-BEATS', '+ETSformer', '+BasisFormer', E1])
    t = ablation(t)
    t = hyper_study(t)
    app.write_text(t)
    print('tables written')


if __name__ == '__main__':
    main()
