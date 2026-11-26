#!/usr/bin/env python3
"""Worker of the R3 tuning jobs of PhaseFormer-E1 (scripts/e1_tune_r3.py).

It builds the E1 model exactly as ``scripts/phaseformer_e12.py`` does and adds one option:
a job with ``rank_override`` r replaces the trend state of the learned decoder head by one
of dimension r (encoder d -> r and decoder r -> H, decoder zero-initialized as before).
The option lives here because the protocol modules fix the state dimension to 3.

    python3 scripts/phaseformer_e13.py --job path/to/job.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402
import scripts.phaseformer_e12 as pe  # noqa: E402

_e1_structure = pe.apply_structure


def set_state_dimension(model, rank):
    head = model.lowrank_interaction_branch.lowrank_head
    if not hasattr(head, 'decoder') or not isinstance(head.decoder, nn.Linear):
        raise ValueError('the state dimension is free for the learned decoder head only')
    head.encoder = nn.Linear(head.encoder.in_features, rank)
    head.decoder = nn.Linear(rank, head.pred_len)
    nn.init.zeros_(head.decoder.weight)
    nn.init.zeros_(head.decoder.bias)
    head.rank = rank


def apply_structure(model, arm, job):
    _e1_structure(model, arm, job)
    if job.get('rank_override') is not None:
        set_state_dimension(model, int(job['rank_override']))


sa.apply_structure = apply_structure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--job', type=Path, required=True)
    sa.worker(parser.parse_args().job)


if __name__ == '__main__':
    main()
