#!/usr/bin/env python3
"""Parameters and forward FLOPs of the selected B0H+ runs (input 720, one multivariate window).

FLOPs are counted with torch.utils.flop_counter.FlopCounterMode (2 x multiply-accumulate of every
matmul / convolution / attention product) for one window with all channels, in eval mode on CPU.

    python3 scripts/vldbj_m1/efficiency.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.flop_counter import FlopCounterMode

from scripts.run_lowrank_interaction_stage2 import forecast
from scripts.vldbj_m1.model import construct, parameter_breakdown
from src.dataset.data_factory import data_provider


def measure(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    model, args = construct(config)
    model.eval()
    dataset, _ = data_provider(args.dataset_args, 'val')
    items = [dataset[0]]
    batch = [torch.as_tensor(item).float().unsqueeze(0) for item in items[0]]
    with FlopCounterMode(display=False) as counter:
        forecast(model, batch, 'cpu')
    per_module = {name: sum(ops.values()) for name, ops in counter.get_flop_counts().items()}
    extension = sum(v for name, v in per_module.items()
                    if name.split('.')[-1] in ('lowrank_interaction_branch', 'lowrank_dynamic_fusion')
                    or name.endswith('lowrank_interaction_branch'))
    return {'run': run_dir.name, 'channels': int(batch[0].shape[-1]), 'input_shape': list(batch[0].shape),
            'parameters': parameter_breakdown(model), 'mflops': counter.get_total_flops() / 1e6,
            'mflops_trend_pathway': extension / 1e6,
            'modules': {k: round(v / 1e6, 4) for k, v in per_module.items() if v and k.count('.') <= 1}}


if __name__ == '__main__':
    for path in sys.argv[1:]:
        print(json.dumps(measure(path)), flush=True)
