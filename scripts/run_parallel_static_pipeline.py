#!/usr/bin/env python3
"""Run the predeclared parallel/static matrix and frozen-checkpoint closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_lowrank_interaction_stage2 import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--history-audit',type=Path,required=True)
    p.add_argument('--name',default='lowrank_parallel_static');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    a=p.parse_args();root=ROOT/'research_runs';audit=root/a.name;scratch=root/(a.name+'_scratch')
    test=root/(a.name+'_test');test_scratch=root/(a.name+'_test_scratch')
    status=scratch/'pipeline_status.json'
    def run(stage,command):
        save_json(status,{'status':'running','stage':stage,'command':command})
        try:subprocess.run(command,cwd=ROOT,check=True)
        except Exception as error:
            save_json(status,{'status':'failed','stage':stage,'error':str(error),'command':command})
            raise
    def passed(folder):
        path=folder/'audit_complete.json'
        return path.exists() and json.loads(path.read_text())['status']=='passed'
    python=sys.executable
    if not passed(scratch):
        run('training_and_validation',[python,str(ROOT/'scripts/run_lowrank_six.py'),'--study','parallel',
            '--baseline-reference','components_static','--source-dir',str(a.source_dir),
            '--scratch-dir',str(scratch),'--audit-dir',str(audit),'--epochs','30','--gpus',a.gpus])
    run('freeze_validation',[python,str(ROOT/'scripts/finalize_parallel_static.py'),
        '--scratch-dir',str(scratch),'--validation-audit',str(audit),'--history-audit',str(a.history_audit),'--freeze-only'])
    if not passed(test_scratch):
        run('frozen_test',[python,str(ROOT/'scripts/evaluate_lowrank_level_test.py'),
            '--training-scratch-dir',str(scratch),'--scratch-dir',str(test_scratch),
            '--audit-dir',str(test),'--gpus',a.gpus])
    run('finalize_reports',[python,str(ROOT/'scripts/finalize_parallel_static.py'),
        '--scratch-dir',str(scratch),'--validation-audit',str(audit),'--test-audit',str(test),
        '--history-audit',str(a.history_audit)])
    save_json(status,{'status':'completed','stage':'remote_audits_complete',
        'validation_audit':str(audit),'test_audit':str(test),'local_sync_pending':True})


if __name__=='__main__':main()
