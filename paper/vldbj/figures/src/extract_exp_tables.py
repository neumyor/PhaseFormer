"""Extract the experiment numbers of the paper from its LaTeX tables into CSV files.

    python3 extract_exp_tables.py <overleaf repo> <output dir>

exp_horizon.csv: per-horizon MSE and MAE of the ten models of Table 1 (supplement Tables
full_results and full_results_b, PhaseFormer+ from Table la_full); exp_trend.csv: MSE_trend and
MSE_total on the twelve settings of Sections 5.3 and 5.4 (Tables trend_setting and branch_setting);
exp_ablation.csv: the 28 settings of Section 5.5 (Table ablation_setting).
"""
import csv, re, sys
from pathlib import Path

paper = Path(sys.argv[1]); out = Path(sys.argv[2])
num = lambda c: float(re.sub(r'\\(best|second)\{([^}]*)\}', r'\2', c).strip())
DS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']

def block(text, label):
    i = text.index(f'\\label{{{label}}}')
    return text[i:text.index('\\end{tabular}', i)]

def rows(blk):
    ds = ''
    for line in blk.split('\n'):
        line = line.strip()
        m = re.match(r'\\multirow\{4\}\{\*\}\{([A-Za-z0-9]+)\}', line)
        if m:
            ds = m.group(1)
        elif line.endswith('\\\\') and '&' in line:
            c = [x.strip() for x in line[:-2].split('&')]
            if c[0] == '' and ds and len(c) == 14:
                c[0] = '{' + ds + '}'
            yield c

sup = (paper / 'body/appendix.tex').read_text()
thy = (paper / 'body/appendix_theory.tex').read_text()

# per-horizon MSE/MAE: supplement tables (PhaseFormer and baselines), la_full (PhaseFormer+)
horizon = []
for label, models in [('tab:full_results', ['PhaseFormer', 'PatchTST', 'iTransformer', 'Crossformer', 'FEDformer', 'TimeBase']),
                      ('tab:full_results_b', ['PhaseFormer', 'SparseTSF', 'FITS', 'TimeMixer', None, None])]:
    ds = None
    for c in rows(block(sup, label)):
        if len(c) != 14 or not c[1].isdigit():
            continue
        m = re.search(r'\{([A-Za-z0-9]+)\}\s*$', c[0])
        if m: ds = m.group(1)
        for k, model in enumerate(models):
            if model and not (label == 'tab:full_results_b' and model == 'PhaseFormer'):
                horizon.append((ds, int(c[1]), model, num(c[2 + 2 * k]), num(c[3 + 2 * k])))
ds = None
for c in rows(block(thy, 'tab:la_full')):
    if len(c) == 6 and c[1].isdigit():
        ds = c[0] or ds
        horizon.append((ds, int(c[1]), 'PhaseFormer+', num(c[4]), num(c[5])))
assert len(horizon) == 28 * 10, len(horizon)
with open(out / 'exp_horizon.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['dataset', 'horizon', 'model', 'mse', 'mae']); w.writerows(horizon)

# twelve settings: MSE_trend and MSE_total
trend = []
for c in rows(block(thy, 'tab:trend_setting')):
    if '-' in c[0] and len(c) == 13:
        for k, model in enumerate(['DLinear', 'SparseTSF', 'FITS', 'CycleNet']):
            trend.append((c[0], model, num(c[3 + k]), num(c[9 + k])))
for c in rows(block(thy, 'tab:branch_setting')):
    if '-' in c[0] and len(c) == 11:
        for k, model in enumerate(['PhaseFormer', '+N-BEATS', '+ETSformer', '+BasisFormer', 'PhaseFormer+']):
            trend.append((c[0], model, num(c[1 + k]), num(c[6 + k])))
assert len(trend) == 12 * 9, len(trend)
with open(out / 'exp_trend.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['setting', 'model', 'mse_trend', 'mse_total']); w.writerows(trend)

# ablation: 28 settings
abl = []; ds = None
for c in rows(block(thy, 'tab:ablation_setting')):
    if len(c) == 10 and c[1].isdigit():
        ds = c[0] or ds
        for k, model in enumerate(['PhaseFormer', 'Single gate', 'Linear', 'PhaseFormer+']):
            abl.append((ds, int(c[1]), model, num(c[2 + k]), num(c[6 + k])))
assert len(abl) == 28 * 4, len(abl)
with open(out / 'exp_ablation.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['dataset', 'horizon', 'model', 'mse', 'mae']); w.writerows(abl)
print(len(horizon), len(trend), len(abl))
