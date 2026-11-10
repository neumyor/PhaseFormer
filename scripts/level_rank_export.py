#!/usr/bin/env python3
"""Dump the level rank sweep results to JSON for plotting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.level_rank_sweep import DEFAULT_SCRATCH, SETTINGS, RANKS  # noqa: E402

out = {}
for dataset, horizon in SETTINGS:
    rows = {}
    for rank in RANKS:
        hits = list((DEFAULT_SCRATCH / 'r1' / 'runs').glob(
            f'{dataset}_H{horizon}_S2021__b0__r{rank}__*/result.json'))
        if hits:
            r = json.loads(hits[0].read_text())['best']
            rows[str(rank)] = {'mse': r['test_mse'], 'mae': r['test_mae']}
    out[f'{dataset}-{horizon}'] = rows
print(json.dumps(out, indent=1))
