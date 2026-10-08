#!/usr/bin/env python3
"""Freeze the registered validation decision and close parallel/static audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.finalize_lowrank_level_audit import paired,md_table
from scripts.run_lowrank_interaction_stage2 import save_json,digest
from scripts.run_lowrank_six import PARALLEL_ARMS,PARALLEL_COMPARISONS

MARKER='\n## 11. Parallel Static-Fusion Decision\n'


def decision(table):
    a=table[table.config_id=='parallel_components_static'].set_index('setting').sort_index()
    b=table[table.config_id=='components_static'].set_index('setting').sort_index()
    assert a.index.equals(b.index)
    p=paired(table,'parallel_components_static','components_static')
    criteria={'mean_relative_mse_below_zero':p['relative_mse_percent']<0,
        'median_relative_mse_below_zero':p['median_relative_mse_percent']<0,
        'eight_setting_wins':p['mse_wins']>=8,
        'four_task_wins':p['task_mse_wins']>=4,
        'mean_relative_mae_nonpositive':p['relative_mae_percent']<=0}
    rows=pd.DataFrame({'dataset':a.dataset,'horizon':a.horizon,
        'relative_mse':(a.mse/b.mse-1)*100,'relative_mae':(a.mae/b.mae-1)*100,
        'relative_p95':(a.candidate_mse_p95/b.candidate_mse_p95-1)*100,
        'relative_top1':(a.candidate_mse_top1_mean/b.candidate_mse_top1_mean-1)*100})
    problems=[]
    for (dataset,horizon),g in rows.groupby(['dataset','horizon']):
        if ((g.relative_mse>1)&(g.relative_mae>1)).all():
            problems.append({'dataset':dataset,'horizon':int(horizon),'reason':'both seeds MSE and MAE >1% worse'})
        for metric in ['relative_p95','relative_top1']:
            if (g[metric]>5).all():
                problems.append({'dataset':dataset,'horizon':int(horizon),'reason':f'both seeds {metric} >5% worse'})
    passed=all(criteria.values()) and not problems
    return {'source':'validation','frozen_before_test':True,'test_set_selection':False,
        'criteria':criteria,'blocking_tasks':problems,'performance_gate_passed':passed,
        'primary_comparison':p,'default_candidate':'parallel_components_static' if passed else 'components_static',
        'retained_arms':['components_static','parallel_components_static'] if passed else ['components_static'],
        'additional_training':0,'core_training_runs':36,'test_checkpoint_evaluations':36,
        'scope':'Two-seed matched internal comparison; no significance, blind-test or Golden-superiority claim.'}


def close(audit,frozen,history):
    run=yaml.safe_load((audit/'run.yaml').read_text())
    assert run['validation']['status']=='passed'
    table=pd.read_csv(audit/'results.csv');assert set(table.config_id)==set(PARALLEL_ARMS)
    comparisons=[paired(table,a,b) for a,b in PARALLEL_COMPARISONS]
    split=str(table.setting.iloc[0]).rsplit('_',1)[1]
    diagnostics=[]
    for _,r in table.iterrows():
        row={'setting':r.setting,'config_id':r.config_id,'mse':float(r.mse),
             'phase_mse':float(r.mean_phase_mse),'trend_mse':float(r.mean_trend_mse),
             'off_relative_mse_percent':float((r.mean_off_mse/r.mse-1)*100)}
        for column in ['mean_only_pt_mse','mean_only_tp_mse']:
            row[column+'_relative_percent']=float((r[column]/r.mse-1)*100)
        diagnostics.append(row)
    matches=0
    if split=='val' and history:
        prior=pd.read_csv(history/'results.csv');base=table[table.config_id=='components_static'].set_index('setting')
        old=prior[prior.config_id=='components_static'].set_index('setting')
        np.testing.assert_allclose(base[['mse','mae']],old.loc[base.index,['mse','mae']],atol=0,rtol=0)
        matches=len(base)
    run['selection'].update(source='validation',default_candidate=frozen['default_candidate'],
        test_set_selection=False,prior_test_exposure=True,validation_decision=frozen)
    for record in run['selection']['selected_configs']:
        record['config_id']=frozen['retained_arms']
        record['search_notes']='Validation decision frozen before test; test cannot change selection.'
    table['selected']=table.config_id.isin(frozen['retained_arms']);table.to_csv(audit/'results.csv',index=False)
    patterns=[]
    with np.load(audit/'selected_cases.npz',allow_pickle=False) as z:
        seen=set()
        for i,raw in enumerate(z['metadata']):
            m=json.loads(str(raw));key=(m['setting'],m['candidate_config_id'],m['sample_id'],m['channel'])
            if key in seen:continue
            seen.add(key);y=z['truth'][i,:z['truth_length'][i]];p=z['candidate'][i,:z['candidate_length'][i]]
            patterns.append({'config_id':m['candidate_config_id'],'unique_cases':1,
                'level_error_dominant':int(m['candidate_level_mse']>m['candidate_shape_mse']),
                'std_above_truth':int(p.std()>y.std())})
    patterns=pd.DataFrame(patterns).groupby('config_id').sum().reset_index().to_dict('records')
    run['analysis']['final_decision']={'frozen_validation':frozen,'split_comparisons':comparisons,
        'fixed_weight_directional_diagnostics':diagnostics,'historical_baseline_exact_matches':matches,
        'selected_case_patterns':patterns}
    run['code']['closure_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report=(audit/'objective_error_analysis.md').read_text().split(MARKER)[0]
    lines=[MARKER,'',f"Validation-frozen default: {frozen['default_candidate']}. Upgrade gate passed: {frozen['performance_gate_passed']}. Test does not reselect.",'',
        f'Final {split} comparisons, setting-equal relative percentages (negative is better):','',
        md_table(comparisons,['candidate','baseline','relative_mse_percent','median_relative_mse_percent','relative_mae_percent','mse_wins','settings','task_mse_wins','tasks']),'',
        md_table(comparisons,['candidate','baseline','mean_candidate_level_mse_relative_percent','mean_candidate_shape_mse_relative_percent','candidate_mse_p95_relative_percent','candidate_mse_top1_mean_relative_percent']),'',
        'Gate conditions fixed before training:','',json.dumps(frozen['criteria'],indent=2),'',
        'Blocking two-seed task regressions: '+json.dumps(frozen['blocking_tasks']),'',
        'Fixed-weight interventions (off removes both interaction directions; only_pt/only_tp retain one direction). Removing post-norm interaction also removes its normalization update, so this is a path intervention, not a pure message-only or retrained single-direction comparison.','',
        md_table(diagnostics,['setting','config_id','mse','phase_mse','trend_mse','off_relative_mse_percent','mean_only_pt_mse_relative_percent','mean_only_tp_mse_relative_percent']),'',
        'Both directional modules read the original states. T->P has one key/value and broadcasts a message; P->T can select phase slots. This experiment does not add pre-norm, message scaling, null routing or dynamic output gates.','',
        'Observed patterns within unique ranked cases; this selected subset does not estimate population prevalence:','',
        md_table(patterns,['config_id','unique_cases','level_error_dominant','std_above_truth']),'',
        f'Historical B0 exact paired matches on this split: {matches}. Exact reproduction is not another independent seed.','',
        'The oracle uses future labels only for post-hoc diagnosis and is not deployable. Branch/oracle and intervention differences cannot identify the training cause or establish that future optimal coefficients are learnable.','',
        'All 36 registered training runs use random initialization and joint training; final test uses their registered validation-Huber checkpoints, with no additional training. No automatic dataset/seed expansion, test-based retuning or dataset-wise winner stitching. Prior validation/test exposure is disclosed. Golden remains the fixed original reference; matching full original training protocol is not established.','']
    (audit/'objective_error_analysis.md').write_text(report+'\n'.join(lines))
    report=(audit/'objective_error_analysis.md').read_text();refs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
    assert all(len(Path(r).parts)==2 and Path(r).parts[0]=='figures' for r in refs)
    with zipfile.ZipFile(audit/'objective_error_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for rel in ['objective_error_analysis.md',*refs]:
            assert not (audit/rel).is_symlink();z.write(audit/rel,rel)
    with zipfile.ZipFile(audit/'objective_error_analysis.zip') as z:
        assert set(z.namelist())=={'objective_error_analysis.md',*refs}
        for rel in z.namelist():assert z.read(rel)==(audit/rel).read_bytes()
    assert {p.name for p in audit.iterdir()}=={'run.yaml','results.csv','sample_errors.csv','selected_cases.npz','objective_error_analysis.md','objective_error_analysis.zip','figures'}
    assert {p.name for p in (audit/'figures').iterdir()}=={Path(r).name for r in refs}
    run['validation']['final_decision_and_archive_checked']=True
    (audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    print(json.dumps({'audit':str(audit),'comparisons':comparisons}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch-dir',type=Path,required=True)
    p.add_argument('--validation-audit',type=Path,required=True);p.add_argument('--test-audit',type=Path)
    p.add_argument('--history-audit',type=Path);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    target=a.scratch_dir/'frozen_selection.json'
    if not target.exists():
        run=yaml.safe_load((a.validation_audit/'run.yaml').read_text());assert run['validation']['status']=='passed'
        frozen=decision(pd.read_csv(a.validation_audit/'results.csv'))
        frozen['validation_results_sha256_before_closure']=digest(a.validation_audit/'results.csv')
        save_json(target,frozen)
    frozen=json.loads(target.read_text());assert frozen['frozen_before_test']
    close(a.validation_audit,frozen,a.history_audit)
    if not a.freeze_only:
        assert a.test_audit is not None
        close(a.test_audit,frozen,None)
    print(json.dumps(frozen,indent=2),flush=True)


if __name__=='__main__':main()
