#!/usr/bin/env python3
"""Parameters and FLOPs of the hyperparameter-study variants (router count M, interaction form, phase length P).

Measured as in scripts/e1_paper/efficiency.py; the interaction variants are patched with patch_routing as in training.
    python3 scripts/e1_paper/efficiency_interaction.py   # writes 06_hyperparameter_study/test_selected/efficiency_variants.csv
"""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
import scripts.b0hp_structure_ablation as sa  # noqa: E402
import scripts.phaseformer_e13 as pe13  # noqa: E402
import scripts.e1_paper.hyperparameter_study as hs  # noqa: E402
from scripts.run_lowrank_interaction_stage2 import forecast  # noqa: E402
from scripts.vldbj_m1.model import parameter_breakdown  # noqa: E402
from scripts.vldbj_m1.proto import build_model  # noqa: E402
from src.dataset.data_factory import data_provider  # noqa: E402

OUT = ROOT / 'research_runs/e1_paper_v1/06_hyperparameter_study/test_selected'


def measure(path):
    job = dict(json.loads(Path(path).read_text()), smoke=True)
    job.pop('source_hashes', None); job.pop('code_commit', None)
    model, args, _, _ = build_model(job)
    pe13.apply_structure(model, job['ablation_arm'], job)
    if job.get('routing_variant'):
        hs.patch_routing(model, job['routing_variant'])
    model.eval()
    dataset, _ = data_provider(args.dataset_args, 'val')
    batch = [torch.as_tensor(v).float().unsqueeze(0) for v in dataset[0]]
    with FlopCounterMode(display=False) as counter:
        forecast(model, batch, 'cpu')
    return parameter_breakdown(model)['total'], counter.get_total_flops() / 1e6


def main():
    sa.register_heads()
    rows = []
    for p in sorted((OUT / 'jobs').glob('*.json')):
        factor, level = p.stem.split('__', 1)[1].split('_', 1)
        params, mf = measure(p)
        rows.append(dict(setting=p.stem.split('__')[0], factor=factor, level=level, params=params, mflops=round(mf, 4)))
        print(rows[-1], flush=True)
    with (OUT / 'efficiency_variants.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
