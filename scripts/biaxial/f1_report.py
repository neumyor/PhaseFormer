"""Summarize the F1 token-design test: per setting mean test MSE over seeds, paired change to the base variant.

    python -m scripts.biaxial.f1_report runs/f1_results.jsonl
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict

VARIANTS = ('base', 'preln', 'fourier', 'both')


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    cell = defaultdict(dict)          # (dataset, horizon, seed) -> variant -> row
    for r in rows:
        cell[(r['dataset'], r['horizon'], r['seed'])][r['group'].split('/')[1]] = r
    settings = sorted({(d, h) for d, h, _ in cell})
    for metric in ('test_mse', 'val_mse'):
        print(f'\n{metric}: mean over 3 seeds (std); change = mean of per-seed relative change to base\n')
        print('| setting | ' + ' | '.join(VARIANTS) + ' |')
        print('|---|' + '---:|' * len(VARIANTS))
        total = {v: [] for v in VARIANTS}
        chg = {v: [] for v in VARIANTS}
        wins = {v: [0, 0] for v in VARIANTS}
        for d, h in settings:
            cells = []
            for v in VARIANTS:
                vals = [cell[(d, h, s)][v][metric] for s in (2021, 2022, 2023) if v in cell[(d, h, s)]]
                base = [cell[(d, h, s)]['base'][metric] for s in (2021, 2022, 2023) if v in cell[(d, h, s)]]
                if len(vals) < 3:
                    cells.append(f'n={len(vals)}')
                    continue
                rel = st.mean(a / b - 1 for a, b in zip(vals, base))
                total[v].append(st.mean(vals)); chg[v].append(rel)
                wins[v][0] += rel < 0; wins[v][1] += 1
                cells.append(f'{st.mean(vals):.4f} ({st.pstdev(vals):.4f})' + ('' if v == 'base' else f' {rel * 100:+.1f}%'))
            print(f'| {d}-{h} | ' + ' | '.join(cells) + ' |')
        print('| **mean** | ' + ' | '.join(f'{st.mean(total[v]):.4f}' + ('' if v == 'base' else f' {st.mean(chg[v]) * 100:+.2f}% ({wins[v][0]}/{wins[v][1]} better)')
                                           if total[v] else '-' for v in VARIANTS) + ' |')
    print('\nparams:', {v: next(r['params'] for r in rows if r['group'].endswith('/' + v)) for v in VARIANTS})


if __name__ == '__main__':
    main()
