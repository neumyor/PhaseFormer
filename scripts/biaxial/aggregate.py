"""Summarize a results file and apply the screening rule.

    python -m scripts.biaxial.aggregate results.jsonl --baseline E0/M0 [--groups E1/B0 E1/B2]

Per group: validation MSE per (dataset, horizon), the mean relative change against
the baseline, the number of datasets where the group is not worse than the baseline
(dataset value = mean over its horizons), parameters, FLOPs per channel and training
time. A group beats the baseline when its mean relative change is at most -1% and it
is not worse on at least 3 of the 4 datasets. Test MSE is printed for reference only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

THRESHOLD = -0.01
MIN_DATASETS = 3


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    table = defaultdict(dict)
    for r in rows:
        table[r['group']][(r['dataset'], r['horizon'], r['seed'])] = r
    return table


def summarize(table, group, baseline):
    cells = table[group]
    base = table[baseline]
    shared = sorted(set(cells) & set(base))
    rel = {k: cells[k]['val_mse'] / base[k]['val_mse'] - 1 for k in shared}
    by_dataset = defaultdict(list)
    for (dataset, _, _), value in rel.items():
        by_dataset[dataset].append(value)
    dataset_mean = {d: sum(v) / len(v) for d, v in by_dataset.items()}
    mean_rel = sum(rel.values()) / len(rel) if rel else float('nan')
    not_worse = sum(v <= 0 for v in dataset_mean.values())
    any_row = next(iter(cells.values()))
    return {
        'group': group, 'cells': len(cells), 'mean_rel': mean_rel, 'not_worse': not_worse,
        'datasets': len(dataset_mean), 'beats': mean_rel <= THRESHOLD and not_worse >= MIN_DATASETS,
        'params': any_row['params'], 'flops': any_row['flops_per_channel'],
        'train_s': sum(r['train_seconds'] for r in cells.values()) / len(cells),
        'val': sum(r['val_mse'] for r in cells.values()) / len(cells),
        'test': sum(r['test_mse'] for r in cells.values()) / len(cells),
        'rel': rel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--groups', nargs='*')
    args = parser.parse_args()
    table = load(args.results)
    groups = args.groups or sorted(g for g in table if g != args.baseline)
    print('| group | runs | mean val MSE | rel. val vs baseline | datasets not worse | beats | params | MFLOPs/ch | s/run | mean test MSE |')
    print('|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|')
    for group in [args.baseline] + groups:
        if group not in table:
            continue
        s = summarize(table, group, args.baseline)
        print(f"| {group} | {s['cells']} | {s['val']:.4f} | {s['mean_rel'] * 100:+.2f}% | "
              f"{s['not_worse']}/{s['datasets']} | {'yes' if s['beats'] else 'no'} | {s['params']} | "
              f"{s['flops'] / 1e6:.2f} | {s['train_s']:.0f} | {s['test']:.4f} |")


if __name__ == '__main__':
    main()
