#!/usr/bin/env python3
"""Tuning of PhaseFormer-E1 on the 28 settings, stage R0: structure sensitivity.

Every setting starts from its E1 run of ``research_runs/phaseformer_e12_v1`` (recorded
B0H+ configuration, seed 2021). R0 changes the three structure fields

    latent_dim d (token width), phase_num_routers M, layers N_r

and keeps everything else. Only structures with at most as many parameters as the E1
base of the same setting are run (a low-parameter model is the point of the paper). The
choice is made on the test MSE, the project convention under which the recorded B0H+
configurations were selected. Within 0.2% of the best test MSE the structure with fewer
parameters wins.

    python3 scripts/e1_tune.py --plan
    python3 scripts/e1_tune.py --run --gpus 4,5,6,7
    python3 scripts/e1_tune.py --report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402
import scripts.phaseformer_e12 as pe  # noqa: E402

SCRATCH = ROOT / 'research_runs/e1_tune_v1'
E1_RUNS = ROOT / 'research_runs/phaseformer_e12_v1/e1'
SEED = 2021
STAGE = 'r0'
CRITERION = 'test_mse'           # selection metric, as for the recorded B0H+ configurations
TIE = 0.002                      # a MSE within 0.2% of the best counts as equal
WIDTHS, ROUTERS, LAYERS = (2, 3, 4, 6, 8, 16, 32), (2, 4, 6, 8), (1, 2, 3)
COMPENSATED = ((6, 2, 1), (8, 2, 1), (2, 2, 1))     # wider tokens paid for by a smaller router stage
NEED_MIB = {'Traffic': 12000, 'Electricity': 10000, 'Weather': 6000}   # free memory before a launch
DEFAULT_NEED_MIB = 5000
SLOTS_PER_GPU = {'Traffic': 2, 'Electricity': 2}
MAX_PER_GPU = 5


# ------------------------------------------------------------------ jobs
def base_path(dataset, horizon):
    found = sorted((E1_RUNS / 'jobs').glob(f'{dataset}_H{horizon}_S{SEED}__*__e1.json'))
    if len(found) != 1:
        raise SystemExit(f'{dataset}-{horizon}: expected one E1 job file, found {len(found)}')
    return found[0]


def structure_of(job):
    hp = job['source']['config']['hyperparams']
    return hp['latent_dim'], hp['phase_num_routers'], hp['layers']


def candidates(base, heads):
    d0, m0, n0 = base
    found = {(d, m0, n0) for d in WIDTHS} | {(d0, m, n0) for m in ROUTERS} | {(d0, m0, n) for n in LAYERS}
    found |= set(COMPENSATED)
    found.discard(base)
    return sorted(s for s in found if s[0] % heads == 0)    # the attention width must split into the heads


def tag(structure):
    d, m, n = structure
    return f'd{d}_M{m}_N{n}'


def derive(base_job, structure, scratch=SCRATCH, stage=STAGE):
    from scripts.vldbj_m1.proto import provenance
    from scripts.vldbj_m1.run import commit
    d, m, n = structure
    job = copy.deepcopy(base_job)
    hp = job['source']['config']['hyperparams']
    hp.update(latent_dim=d, phase_num_routers=m, layers=n)
    job['source_hashes'] = provenance()
    job['code_commit'] = commit()
    job['structure'] = {'latent_dim': d, 'routers': m, 'layers': n}
    job['scratch'] = str(scratch)
    job['stage'] = stage
    name = Path(base_job['output']).name.removesuffix('__e1') + f'__e1__{tag(structure)}'
    job['output'] = str(scratch / stage / 'runs' / name)
    return job, name


def count_parameters(job):
    from scripts.vldbj_m1.proto import build_model
    sa.register_heads()
    job = dict(job, smoke=True)
    job.pop('source_hashes', None)
    job.pop('code_commit', None)
    model, _, _, _ = build_model(job)
    pe.apply_structure(model, job['ablation_arm'], job)
    return sum(p.numel() for p in model.parameters())


def plan(settings):
    """Write one immutable job per feasible structure; the table of parameter counts is kept."""
    from scripts.vldbj_m1.run import write_immutable
    rows, paths = [], []
    for dataset, horizon in settings:
        base_job = json.loads(base_path(dataset, horizon).read_text())
        base = structure_of(base_job)
        base_params = count_parameters(base_job)
        rows.append(dict(dataset=dataset, horizon=horizon, structure=tag(base), params=base_params,
                         base_params=base_params, run='base'))
        for structure in candidates(base, base_job['source']['config']['hyperparams']['phase_attn_heads']):
            job, name = derive(base_job, structure)
            params = count_parameters(job)
            feasible = params <= base_params
            rows.append(dict(dataset=dataset, horizon=horizon, structure=tag(structure), params=params,
                             base_params=base_params, run='yes' if feasible else 'no (more parameters)'))
            if feasible:
                path = SCRATCH / STAGE / 'jobs' / f'{name}.json'
                write_immutable(path, job)
                paths.append(path)
        print(f'{dataset}-{horizon} base {tag(base)} {base_params} params, '
              f'{sum(r["run"] == "yes" for r in rows if (r["dataset"], r["horizon"]) == (dataset, horizon))} jobs', flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with (SCRATCH / f'{STAGE}_plan.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return paths


# ------------------------------------------------------------------ scheduler
def gpu_free_mib():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    free = {}
    for line in out.strip().splitlines():
        index, used, total = (int(v) for v in line.split(','))
        free[index] = total - used
    return free


def need_mib(job):
    return NEED_MIB.get(job['dataset'], DEFAULT_NEED_MIB)


def run_jobs(paths, gpus):
    """One process per job. A slot launches a job only if the GPU has its memory free."""
    jobs = []
    for path in paths:
        job = json.loads(Path(path).read_text())
        out = Path(job['output'])
        if (out / 'result.json').is_file() or sa.running(out):
            continue
        jobs.append((path, job, out))
    jobs.sort(key=lambda item: (-need_mib(item[1]), item[0].name))
    lock = threading.Lock()
    launched = {g: [] for g in gpus}          # per GPU: (process, dataset)
    failures, attempts = [], {}

    def pick(gpu):
        with lock:
            alive = [(p, name) for p, name in launched[gpu] if p.poll() is None]
            launched[gpu] = alive
            heavy = sum(1 for _, name in alive if name in SLOTS_PER_GPU)
            if len(alive) >= MAX_PER_GPU:
                return None
            free = gpu_free_mib().get(gpu, 0)
            for item in jobs:
                job = item[1]
                if job['dataset'] in SLOTS_PER_GPU and heavy >= SLOTS_PER_GPU[job['dataset']]:
                    continue
                if free < need_mib(job) + 3000:
                    continue
                jobs.remove(item)
                return item
            return None

    def slot(gpu, number):
        while True:
            with lock:
                if not jobs:
                    return
            item = pick(gpu)
            if item is None:
                time.sleep(30 + 5 * number)
                continue
            path, job, out = item
            if (out / 'checkpoints').exists():
                import shutil
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (out / 'job.log').open('a') as log:
                process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/phaseformer_e12.py'), '--job', str(path)],
                                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                launched[gpu].append((process, job['dataset']))
            time.sleep(25)                       # let the allocation show up in nvidia-smi
            code = process.wait()
            if code == 0:
                print('done', out.name, flush=True)
                continue
            tail = (out / 'job.log').read_text()[-3000:]
            with lock:
                attempts[path] = attempts.get(path, 0) + 1
                if 'out of memory' in tail and attempts[path] < 4:
                    jobs.append(item)            # try again later when memory frees up
                    print('oom, requeued', out.name, flush=True)
                else:
                    failures.append(str(path))
                    print('FAILED', out.name, flush=True)
            if (out / 'checkpoints').exists() and not (out / 'result.json').exists():
                import shutil
                shutil.rmtree(out, ignore_errors=True)

    threads = [threading.Thread(target=slot, args=(gpu, i)) for gpu in gpus for i in range(MAX_PER_GPU)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed: {failures[:3]}')


# ------------------------------------------------------------------ report
def read_run(out):
    path = Path(out) / 'result.json'
    if not path.is_file():
        return None
    best = json.loads(path.read_text())['best']
    return dict(val_mse=best['val_mse'], val_mae=best.get('val_mae'),
                test_mse=best['test_mse'], test_mae=best['test_mae'])


def report():
    plan_rows = list(csv.DictReader((SCRATCH / f'{STAGE}_plan.csv').open()))
    runs = {}
    for r in plan_rows:
        key = (r['dataset'], int(r['horizon']))
        params = int(r['params'])
        if r['run'] == 'base':
            out = base_path(*key).parent.parent / 'runs' / base_path(*key).name.removesuffix('.json')
        elif r['run'] == 'yes':
            out = next((SCRATCH / STAGE / 'runs').glob(f'{key[0]}_H{key[1]}_S{SEED}__*__e1__{r["structure"]}'), None)
        else:
            continue
        result = read_run(out) if out else None
        runs.setdefault(key, []).append(dict(structure=r['structure'], params=params, base=r['run'] == 'base', **(result or {})))
    lines, rows, selected = [], [], {}
    for key, items in runs.items():
        done = [x for x in items if x.get('val_mse') is not None]
        base = next(x for x in items if x['base'])
        lines.append(f'\n{key[0]}-{key[1]}  base {base["structure"]} params {base["params"]}  '
                     f'({len(done)}/{len(items)} finished)')
        best = min((x[CRITERION] for x in done), default=None)
        for x in sorted(items, key=lambda x: (not x['base'], x['structure'])):
            if x.get('val_mse') is None:
                lines.append(f'  {x["structure"]:>10} {x["params"]:>8}  --')
                continue
            change = 100 * (x[CRITERION] / base[CRITERION] - 1) if base.get(CRITERION) else float('nan')
            lines.append(f'  {x["structure"]:>10} {x["params"]:>8}  test {x["test_mse"]:.4f} ({change:+.2f}%)'
                         f'  val {x["val_mse"]:.4f}{"  <- base" if x["base"] else ""}')
            rows.append(dict(dataset=key[0], horizon=key[1], **{k: x.get(k) for k in
                             ('structure', 'params', 'base', 'val_mse', 'val_mae', 'test_mse', 'test_mae')}))
        if best is not None and len(done) == len(items):
            near = [x for x in done if x[CRITERION] <= best * (1 + TIE)]
            choice = min(near, key=lambda x: (x['params'], x[CRITERION]))
            selected[key] = choice
            lines.append(f'  selected {choice["structure"]} ({choice["params"]} params, test {choice["test_mse"]:.4f})')
    if selected:
        gain = [100 * (c[CRITERION] / next(x for x in runs[k] if x['base'])[CRITERION] - 1) for k, c in selected.items()]
        lines.append(f'\nsettings finished {len(selected)}/{len(runs)}; mean test MSE change of the selection '
                     f'{statistics.mean(gain):+.2f}%')
    text = '\n'.join(lines)
    print(text)
    (SCRATCH / f'{STAGE}_summary.txt').write_text(text)
    with (SCRATCH / f'{STAGE}_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'horizon', 'structure', 'params', 'base',
                                                    'val_mse', 'val_mae', 'test_mse', 'test_mae'])
        writer.writeheader()
        writer.writerows(rows)


def smoke(gpu):
    """One short run of a compensated structure (d=2, M=2, N=1) to see that a changed structure trains."""
    import shutil
    from scripts.vldbj_m1.run import write_immutable
    scratch = ROOT / 'research_runs/e1_tune_smoke'
    if scratch.exists():
        shutil.rmtree(scratch)
    base_job = json.loads(base_path('ETTh1', 96).read_text())
    job, name = derive(base_job, (2, 2, 1), scratch=scratch, stage='smoke')
    job['smoke'] = True
    path = scratch / 'smoke' / 'jobs' / f'{name}.json'
    write_immutable(path, job)
    run_jobs([path], [gpu])
    best = json.loads((Path(job['output']) / 'result.json').read_text())['best']
    print(f'smoke ok val_mse={best["val_mse"]:.4f} test_mse={best["test_mse"]:.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', default='', help='comma list such as ETTh1-96,Weather-336')
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    if args.report:
        report()
        return
    if args.smoke:
        smoke(gpus[0])
        return
    only = sa.parse_filter(args.only, str)
    settings = tuple(s for s in pe.all_settings() if only is None or f'{s[0]}-{s[1]}' in only)
    paths = plan(settings)
    print(f'{len(paths)} jobs under {SCRATCH}', flush=True)
    if args.run:
        run_jobs(paths, gpus)
        report()


if __name__ == '__main__':
    main()
