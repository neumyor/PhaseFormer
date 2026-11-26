#!/usr/bin/env python3
"""Parameters and FLOPs of the selected E1 runs, measured as in experiment_data/server_scripts/efficiency.

torch.utils.flop_counter.FlopCounterMode (2 x multiply-accumulate), eval mode on CPU, one validation window with all
channels, L = 720, the horizon of the setting. The E1 model is built from the job of the run exactly as in training
(scripts/vldbj_m1/proto.build_model plus the E1 patches of scripts/phaseformer_e13.py).

    python3 scripts/e1_paper/efficiency.py            # writes research_runs/e1_paper_v1/02_efficiency/e1_efficiency_28.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import scripts.b0hp_structure_ablation as sa  # noqa: E402
import scripts.phaseformer_e13 as pe13  # noqa: E402
from scripts.run_lowrank_interaction_stage2 import forecast  # noqa: E402
from scripts.vldbj_m1.model import parameter_breakdown  # noqa: E402
from scripts.vldbj_m1.proto import build_model  # noqa: E402
from src.dataset.data_factory import data_provider  # noqa: E402

BASE = ROOT / 'research_runs/e1_paper_v1'


def measure(job_path):
    job = json.loads(Path(job_path).read_text())
    job = dict(job, smoke=True)
    job.pop('source_hashes', None)
    job.pop('code_commit', None)
    model, args, _, _ = build_model(job)
    pe13.apply_structure(model, job['ablation_arm'], job)
    model.eval()
    dataset, _ = data_provider(args.dataset_args, 'val')
    batch = [torch.as_tensor(v).float().unsqueeze(0) for v in dataset[0]]
    with FlopCounterMode(display=False) as counter:
        forecast(model, batch, 'cpu')
    total = counter.get_total_flops() / 1e6
    per = {n: sum(o.values()) / 1e6 for n, o in counter.get_flop_counts().items()}
    parts = parameter_breakdown(model)
    trend = per.get('PhaseFormer.lowrank_interaction_branch', 0.0)
    hp = job['source']['config']['hyperparams']
    return dict(channels=int(batch[0].shape[-1]), period=hp['period_len'], layers=hp['layers'], latent_dim=hp['latent_dim'],
                routers=hp['phase_num_routers'], heads=hp['phase_attn_heads'], params_total=parts['total'],
                params_phase_pathway=parts['backbone'],
                params_trend_pathway_and_gates=parts['level_branch'] + parts['fusion'] + parts['modulation'],
                mflops_total=round(total, 4), mflops_trend_pathway=round(trend, 4),
                mflops_phase_pathway=round(total - trend, 4))


def main():
    sa.register_heads()
    rows = list(csv.DictReader((BASE / '00_protocol/e1_final_configs_28.csv').open()))
    cache, out = {}, []
    for r in rows:
        key = r['run_dir']
        if key not in cache:
            cache[key] = measure(r['job'])
            print(r['dataset'], r['horizon'], r['selection'], cache[key]['params_total'], cache[key]['mflops_total'], flush=True)
        out.append(dict(dataset=r['dataset'], horizon=r['horizon'], selection=r['selection'], stage=r['stage'],
                        run_dir=key, **cache[key]))
    target = BASE / '02_efficiency/e1_efficiency_28.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print('wrote', target, len(out), 'rows,', len(cache), 'distinct runs')


if __name__ == '__main__':
    main()
