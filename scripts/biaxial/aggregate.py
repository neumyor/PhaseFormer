"""Summarize a results file against the original PhaseFormer (golden) results.

    python -m scripts.biaxial.aggregate results.jsonl [--groups G ...] [--golden docs/PhaseFormer_gold_standard.md]

Per group, over the cells present in the golden table: mean validation MSE (the selection
criterion between candidates), mean test MSE, the mean relative gap of the test MSE to the
golden test MSE, parameters, FLOPs per channel and training time. Groups are sorted by
validation MSE. Only cells of the screening set (ETTh1 and ETTm1, horizons 96 and 336,
seed 2021) are used.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CELLS = {(d, h) for d in ('ETTh1', 'ETTm1') for h in (96, 336)}
DEFAULT_GOLDEN = Path(__file__).resolve().parents[2] / 'docs/PhaseFormer_gold_standard.md'


def golden(path):
    table = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', line)
        if m:
            table[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return table


def load(paths):
    table = defaultdict(dict)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if (r['dataset'], r['horizon']) in CELLS and r['seed'] == 2021:
                    table[r['group']][(r['dataset'], r['horizon'])] = r
    return table


def summarize(cells, gold):
    rows = list(cells.values())
    gap = [r['test_mse'] / gold[(r['dataset'], r['horizon'])] - 1 for r in rows]
    return {
        'cells': len(rows), 'val': sum(r['val_mse'] for r in rows) / len(rows),
        'test': sum(r['test_mse'] for r in rows) / len(rows), 'gap': sum(gap) / len(gap),
        'params': rows[0]['params'], 'flops': rows[0]['flops_per_channel'],
        'seconds': sum(r['train_seconds'] for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+')
    parser.add_argument('--groups', nargs='*')
    parser.add_argument('--golden', default=str(DEFAULT_GOLDEN))
    args = parser.parse_args()
    gold = golden(args.golden)
    table = load(args.results)
    groups = args.groups or sorted(table)
    stats = {g: summarize(table[g], gold) for g in groups if g in table}
    print('| group | cells | mean val MSE | mean test MSE | test gap to golden | params | MFLOPs/ch | s/run |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for g, s in sorted(stats.items(), key=lambda kv: kv[1]['val']):
        print(f"| {g} | {s['cells']} | {s['val']:.4f} | {s['test']:.4f} | {s['gap'] * 100:+.2f}% | "
              f"{s['params']} | {s['flops'] / 1e6:.2f} | {s['seconds']:.0f} |")


if __name__ == '__main__':
    main()
