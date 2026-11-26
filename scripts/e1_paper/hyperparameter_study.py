#!/usr/bin/env python3
"""Hyperparameter analysis of E1 (paper replication, 06_hyperparameter_study).

Six settings (Traffic, Electricity, Weather at H = 96 and 336), seed 2021, the selected E1 configuration of each setting
with one knob changed (as in the analysis of the paper, scripts of experiment_data/server_scripts/hyperparameter_study):

  M            number of routers in {1, 2, 4, 8, 16}
  interaction  routers (selected), full self-attention, linear mixing, none
  P            phase length in {8, 12, 16, 24, 48} (Traffic only; all divide 96 and 336)
  L            input length in {96, 192, 336, 720}

The selected value of each knob is the run of 00_protocol/e1_final_configs_28.csv (selection = test for the main analysis, val for the appendix).

    python3 scripts/e1_paper/hyperparameter_study.py --prepare
    python3 scripts/e1_paper/hyperparameter_study.py --run --gpus 4,5,6,7
    python3 scripts/e1_paper/hyperparameter_study.py --report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.e1_paper.ablation as ab  # noqa: E402

BASE = ROOT / 'research_runs/e1_paper_v1'
OUT = BASE / '06_hyperparameter_study'            # set per selection by set_selection()
SELECTION = 'test'
SETTINGS = (('Traffic', 96), ('Traffic', 336), ('Electricity', 96), ('Electricity', 336), ('Weather', 96), ('Weather', 336))
M_LIST = (1, 2, 4, 8, 16)
VARIANTS = ('full_attn', 'linear_mix', 'none')
P_LIST = (8, 12, 16, 24, 48)
L_LIST = (96, 192, 336, 720)


def set_selection(selection):
    global OUT, SELECTION
    SELECTION = selection
    OUT = BASE / '06_hyperparameter_study' / f'{selection}_selected'


def patch_routing(model, variant):
    from torch import nn
    for unit in model.routing_layers:
        layer = unit.interact
        if variant == 'full_attn':
            atts = [nn.MultiheadAttention(layer.latent_dim, layer.num_heads, batch_first=True, dropout=layer.dropout)
                    for _ in range(2)]
            layer.add_module('full_attn', atts[0])
            layer.add_module('full_attn_2', atts[1])

            def forward(self, Z, _atts=tuple(atts)):
                B, C, L, D = Z.shape
                x = Z.reshape(B * C, L, D)
                for _att in _atts:
                    a, _ = _att(x, x, x, need_weights=False)
                    x = x + self.dropout_layer(a)
                out = self.norm1(x)
                out = self.norm2(out + self.dropout_layer(self.mlp(out)))
                return out.view(B, C, L, D)
        elif variant == 'linear_mix':
            mix = nn.Linear(layer.period_len, layer.period_len)
            layer.add_module('phase_mix', mix)

            def forward(self, Z, _mix=mix):
                B, C, L, D = Z.shape
                x = Z.reshape(B * C, L, D)
                y = _mix(x.transpose(1, 2)).transpose(1, 2)
                out = self.norm1(x + self.dropout_layer(y))
                out = self.norm2(out + self.dropout_layer(self.mlp(out)))
                return out.view(B, C, L, D)
        elif variant == 'none':
            def forward(self, Z):
                B, C, L, D = Z.shape
                x = Z.reshape(B * C, L, D)
                out = self.norm1(x)
                out = self.norm2(out + self.dropout_layer(self.mlp(out)))
                return out.view(B, C, L, D)
        else:
            raise ValueError(variant)
        layer.forward = types.MethodType(forward, layer)


def worker(job_path):
    import scripts.b0hp_ablation as abl
    import scripts.b0hp_structure_ablation as sa
    import scripts.phaseformer_e13 as pe13
    from scripts.vldbj_m1 import proto
    sa.register_heads()
    job = json.loads(Path(job_path).read_text())
    attach = abl.make_attach(job)
    original = proto.build_model

    def patched(j):
        model, args, cfg, common = original(j)
        pe13.apply_structure(model, j['ablation_arm'], j)
        if j.get('routing_variant'):
            patch_routing(model, j['routing_variant'])
        attach(model, j['backbone_lr_mult'], j['extension_lr_mult'])
        return model, args, cfg, common

    proto.build_model = patched
    proto.attach_lr_groups = attach
    proto.run_job(job_path)


def prepare():
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit, write_immutable
    rows = {(r['dataset'], int(r['horizon'])): r for r in csv.DictReader((BASE / '00_protocol/e1_final_configs_28.csv').open())
            if r['selection'] == SELECTION}
    paths = []

    def write(setting, factor, level, change, routing=None):
        row = rows[setting]
        job = copy.deepcopy(json.loads(Path(row['job']).read_text()))
        change(job['source']['config'])
        if routing:
            job['routing_variant'] = routing
        job['stage'] = f'hp_{factor}'
        job['scratch'] = str(OUT)
        job['source_hashes'] = provenance()
        job['code_commit'] = commit()
        name = f'{setting[0]}_H{setting[1]}_S2021__{factor}_{level}'
        job['output'] = str(OUT / 'runs' / factor / name)
        path = OUT / 'jobs' / f'{name}.json'
        write_immutable(path, job)
        paths.append(path)

    for setting in SETTINGS:
        row = rows[setting]
        selected_m = int(row['routers'])
        for m in M_LIST:
            if m != selected_m:
                write(setting, 'M', m, lambda c, m=m: c['hyperparams'].__setitem__('phase_num_routers', m))
        for v in VARIANTS:
            write(setting, 'interaction', v, lambda c: None, routing=v)
        if setting[0] == 'Traffic':
            for p in P_LIST:
                if p != int(row['period']):
                    def change(c, p=p):
                        c['period'] = p
                        c['hyperparams']['period_len'] = p
                    write(setting, 'P', p, change)
        for length in L_LIST:
            if length != 720:
                write(setting, 'L', length, lambda c, length=length: c.__setitem__('lookback', length))
    (OUT / 'manifest.json').write_text(json.dumps([str(p) for p in paths], indent=1))
    print(len(paths), 'jobs')


def report():
    rows = []
    sel = {(r['dataset'], int(r['horizon'])): r for r in csv.DictReader((BASE / '00_protocol/e1_final_configs_28.csv').open())
           if r['selection'] == SELECTION}
    for path in sorted((OUT / 'runs').glob('*/*/result.json')):
        result = json.loads(path.read_text())
        factor, level = path.parent.parent.name, path.parent.name.split(f'{path.parent.parent.name}_', 1)[-1]
        key = (result['dataset'], result['horizon'])
        s = sel[key]
        rows.append(dict(factor=factor, dataset=key[0], horizon=key[1], level=level,
                         test_mse=result['best']['test_mse'], test_mae=result['best']['test_mae'],
                         selected_test_mse=float(s['test_mse']), selected_test_mae=float(s['test_mae']),
                         mse_change_vs_selected_pct=100 * (result['best']['test_mse'] / float(s['test_mse']) - 1),
                         run_dir=str(path.parent)))
    with (OUT / 'results_e1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), 'runs ->', OUT / 'results_e1.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--selection', choices=('test', 'val'), default='test')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    set_selection(args.selection)
    ab.set_selection(args.selection)
    if args.job:
        worker(args.job)
    elif args.prepare:
        prepare()
    elif args.run:
        ab.__file__ = __file__            # the shared runner starts this script as the worker of the jobs
        paths = [Path(p) for p in json.loads((OUT / 'manifest.json').read_text())]
        ab.run(paths, [int(g) for g in args.gpus.split(',')])
        report()
    elif args.report:
        report()


if __name__ == '__main__':
    main()
