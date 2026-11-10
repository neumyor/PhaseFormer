"""Write the PhaseFormer-I vs B0H+ table from the tuning state.

    python -m scripts.biaxial.report runs/tune_state.json runs/i0_results.jsonl > docs/PhaseFormer_I_results.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.biaxial.configs import b0hp_reference  # noqa: E402
from scripts.biaxial.tune import TARGET_GAP, initial_config  # noqa: E402


def main():
    state = json.loads(Path(sys.argv[1]).read_text())
    first = {}
    for line in Path(sys.argv[2]).read_text().splitlines():
        r = json.loads(line)
        first[(r['dataset'], r['horizon'])] = r['test_mse']
    ref = b0hp_reference()
    print('| setting | B0H+ test MSE | PhaseFormer-I first run | PhaseFormer-I best | gap to B0H+ | trials | matched | changed settings |')
    print('|---|---:|---:|---:|---:|---:|:---:|---|')
    gaps, matched = [], 0
    for key, v in state.items():
        d, h = key.rsplit('-', 1)
        base = initial_config(ref[(d, int(h))]['period'])
        changed = ', '.join(f'{n}={c}' for n, c in v['config'].items() if c != base.get(n))
        gap = v['gap_pct']
        gaps.append(gap)
        ok = gap <= TARGET_GAP * 100
        matched += ok
        f = first.get((d, int(h)))
        print(f"| {key} | {v['ref']:.4f} | {f:.4f} | {v['best']:.4f} | {gap:+.2f}% | {v['trials']} | "
              f"{'yes' if ok else 'no'} | {changed or 'initial'} |")
    print(f'\n{len(gaps)} settings, {matched} within +{TARGET_GAP * 100:.1f}% of B0H+, mean gap {sum(gaps) / len(gaps):+.2f}%.')


if __name__ == '__main__':
    main()
