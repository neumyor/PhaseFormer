"""Detach a live matrix controller while preserving its active training workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def main(args):
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from scripts.extend_interaction_test import save, run_tasks
    scratch = args.scratch.resolve()
    manifest = json.loads((scratch/'manifest.json').read_text())
    command_path = Path('/proc')/str(args.controller)/'cmdline'
    command = command_path.read_bytes().replace(b'\0', b' ').decode()
    assert str(repo) in command and 'runpy.run_path' in command and 'extend_interaction_test' in command
    os.kill(args.controller, signal.SIGSTOP)
    workers = []
    try:
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                parts = path.read_bytes().split(b'\0')
                if b'--worker' not in parts or str(repo/'scripts/extend_interaction_test.py').encode() not in parts:
                    continue
                key = Path(parts[parts.index(b'--worker')+1].decode()).stem
                workers.append({'pid': int(path.parent.name), 'key': key})
            except (OSError, UnicodeDecodeError):
                continue
        save(scratch/'detached_handoff.json', {'supervisor_pid': os.getpid(),
            'session_id': os.getsid(0), 'parent_pid': os.getppid(), 'old_controller': args.controller,
            'adopted_workers': workers, 'started_at': time.time(), 'stdin': os.readlink('/proc/self/fd/0'),
            'stdout': os.readlink('/proc/self/fd/1'), 'repo': str(repo)})
        # Only the coordinator is replaced. Training children keep their file-backed logs.
        os.kill(args.controller, signal.SIGKILL)
    except BaseException:
        try:
            os.kill(args.controller, signal.SIGCONT)
        except ProcessLookupError:
            pass
        raise
    print(f'Detached supervisor {os.getpid()}; preserving {len(workers)} active workers', flush=True)
    while workers:
        pending = []
        for worker in workers:
            path = Path('/proc')/str(worker['pid'])/'cmdline'
            try:
                command = path.read_bytes().replace(b'\0', b' ').decode()
            except FileNotFoundError:
                command = ''
            if worker['key'] in command and '--worker' in command:
                pending.append(worker)
        workers = pending
        if workers:
            time.sleep(5)
    new = [entry.copy() for entry in manifest['entries'] if entry['new_training']]
    pending = []
    for entry in new:
        entry['job_path'] = str(scratch/'jobs'/f"{entry['key']}.json")
        if not (Path(entry['run'])/'result.json').exists():
            if (Path(entry['run'])/'checkpoints').exists():
                raise RuntimeError(f'Adopted worker failed before completion; preserve its partial run: {entry["key"]}')
            pending.append(entry)
    run_tasks(pending, 'train', args.gpus.split(','), scratch)
    assert all((Path(entry['run'])/'result.json').exists() for entry in new)
    save(scratch/'training_complete_before_test.json', {'jobs': len(new), 'completed_at': time.time()})
    entries = [entry.copy() for entry in manifest['entries']]
    for entry in entries:
        entry['evaluation_output'] = str(scratch/'evaluations'/f"{entry['key']}.json")
    run_tasks(entries, 'evaluate', args.gpus.split(','), scratch)
    save(scratch/'complete.json', {'training_jobs': len(new), 'test_inference_runs': len(entries)})
    env = {**os.environ, 'PYTHONPATH': str(repo), 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2'}
    subprocess.run([sys.executable, str(args.summarizer.resolve()), '--scratch', str(scratch),
                    '--output', str(repo/'research_runs/interaction_test_extension_20261009')], env=env, check=True)
    save(scratch/'detached_complete.json', {'status': 'passed', 'completed_at': time.time(),
                                         'training_jobs': len(new), 'test_inference_runs': len(entries)})
    print('Full test matrix and comparison table complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--controller', type=int, required=True)
    parser.add_argument('--summarizer', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except BaseException:
        arguments.scratch.mkdir(parents=True, exist_ok=True)
        (arguments.scratch/'detached_failure.json').write_text(json.dumps({
            'status': 'failed', 'traceback': traceback.format_exc(), 'time': time.time()}, indent=2)+'\n')
        raise
