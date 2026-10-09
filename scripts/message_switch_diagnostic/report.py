"""Complete the portable negative-result audit from frozen diagnostic artifacts."""
import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml
from scripts.message_switch_diagnostic.core import score,summary
from scripts.lowrank_token_interaction.activation_audit import digest,save_json
from scripts.analyze_lowrank_stage2 import markdown_table,pick_cases
from scripts.analyze_lowrank_six import unpack


def main(args):
    run=yaml.safe_load((args.audit/'run.yaml').read_text())
    freeze=json.loads((args.scratch/'selector_frozen_before_evaluation.json').read_text())
    results=pd.read_csv(args.audit/'results.csv')
    errors=pd.read_csv(args.audit/'sample_errors.csv')
    cases=unpack(args.audit/'selected_cases.npz')
    data=pd.concat([pd.read_csv(p) for p in sorted((args.scratch/'parts').glob('*.csv'))],ignore_index=True)
    discrimination=[]
    for name,selector in freeze['selectors'].items():
        for (role,setting),g in data.groupby(['role','setting']):
            s=score(g,selector);target=(g.off_mse<g.candidate_mse).to_numpy()
            n=int(target.sum());m=len(g)-n
            auc=float((rankdata(s)[target].sum()-n*(n+1)/2)/(n*m)) if n and m else None
            discrimination.append({'selector':name,'role':role,'setting':setting,'auc':auc})
    sensitivity=[]
    for name,selector in freeze['selectors'].items():
        threshold=next(a['threshold'] for a in freeze['calibration_search'][name] if a['name']=='q95')
        for setting,g in data[data.role=='eval'].groupby('setting'):
            r=summary(g,score(g,selector)>=threshold)
            r.update(selector=name,setting=setting,relative_mse=(r['mse']/r['native_mse']-1)*100)
            sensitivity.append(r)
    # q95 is a fixed sensitivity reading; it never replaces the calibration selection.
    run['analysis'].update(discrimination=discrimination,q95_sensitivity=sensitivity,
        decision_status='stop_no_from_scratch_message_gate_training',new_predictor_training_runs=0,
        archive_script_sha256=digest(__file__),case_replay=json.loads((args.scratch/'case_replay_complete.json').read_text()))
    selected=freeze['selected_selector'];main=results[results.config_id==selected]
    avg=results.groupby('config_id')[['relative_mse_native','relative_mse_b0','relative_mae_native',
                'relative_mae_b0','disabled_fraction','lost_benefit_fraction','oracle_recovery']].mean().reset_index()
    cal=[]
    for name,attempts in freeze['calibration_search'].items():
        cal.extend({'selector':name,'threshold':a['name'],'relative_mse_native':a['relative_mse'],
                    'all_protection_checks':a['valid']} for a in attempts)
    auc_table=pd.DataFrame(discrimination).groupby(['selector','role']).auc.mean().reset_index()
    sensitivity_table=pd.DataFrame(sensitivity).groupby('selector')[['relative_mse','disabled_fraction',
                                        'lost_benefit_fraction','oracle_recovery']].mean().reset_index()
    case_stats=[]
    for case in cases:
        m=case['metadata'];truth=case['truth'].astype(np.float64)
        row={k:m[k] for k in ('setting','category','rank','sample_id','channel')}
        row.update(truth_range=float(np.ptp(truth)),baseline_std=float(np.std(case['baseline'])),
                   candidate_std=float(np.std(case['candidate'])),truth_std=float(np.std(truth)),
                   baseline_mae=m['baseline_mae'],candidate_mae=m['candidate_mae'],off_mae=m['off_mae'])
        for label in ('baseline','candidate','off'):
            for i,e in enumerate(np.array_split(case[label].astype(np.float64)-truth,3)):
                row[f'{label}_third{i+1}_mse']=float(np.square(e).mean())
        case_stats.append(row)
    run['analysis']['case_descriptive_statistics']=case_stats
    figure_dir=args.audit/'figures';figure_dir.mkdir(exist_ok=True)
    chosen=[]
    for setting in sorted(errors.setting.unique()):
        group=[c for c in cases if c['metadata']['setting']==setting]
        chosen.append(max(group,key=lambda c:c['metadata']['delta_mae']))
    # Ten plots: worst ranked regressions across the setting-coverage candidates.
    chosen=sorted(chosen,key=lambda c:-c['metadata']['delta_mae'])[:10]
    refs=[]
    for case in chosen:
        m=case['metadata'];fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
        ax=axes[0]
        history=case['history'][-192:]
        ax.plot(np.arange(-len(history),0),history,color='.65',label='history')
        for name,color in [('truth','black'),('baseline','#2166ac'),('candidate','#b2182b')]:
            ax.plot(case[name],label=name,color=color,lw=1)
        ax.legend(fontsize=8);ax.set_title(f"{m['setting']} sample{m['sample_id']} c{m['channel']}; MAE delta {m['delta_mae']:+.5f}")
        axes[1].plot(case['truth'],color='black',label='truth',lw=1)
        axes[1].plot(case['off'],color='#188e66',label='off',lw=1)
        axes[1].legend(fontsize=8);axes[1].set_xlabel('Forecast step (history below zero)')
        fig.tight_layout();rel=f"figures/{m['setting']}__sample{m['sample_id']}_c{m['channel']}.png"
        fig.savefig(args.audit/rel,dpi=120);plt.close(fig);refs.append(rel)
    lines=['# Experiment and Objective Error Analysis: temporal message switch','',
        '## 1. Experiment Setup','',
        'Fixed PhaseFormer-L TP-Add and independent b0 weights. Three datasets × H96/H720 × two seeds. Original train prefix60% fits a shared linear selector; train suffix40% calibrates; validation windows with histories entirely in validation evaluate. Complete history-plus-target supports do not overlap across roles. No new predictor training or test inference.',
        '**Exposure:** original predictors trained on all train data and checkpoints selected using original full validation. This is a temporal-transfer diagnostic of fitted predictors, not blind generalization. Prior test findings informed the research focus.',
        'Highest float32 attribution/inference; original medium results are not replaced. Fit/cal sampled every24 starts; evaluation uses every legal start. Overlapping windows and shared labels across seeds are not independent replications.','',
        '## 2. Experiment Results','',
        '**Decision: stop; retain b0 architecture and do not train a new message Gate.** Both selectors chose all-on on calibration. Even without protection constraints, every tested nontrivial calibration threshold has positive mean relative MSE versus native.','']
    lines+=markdown_table(avg)
    lines+=['','Relative metrics are percent; fractions are in [0,1] except signed oracle recovery. Switching statistics on the b0 row describe the native all-on reference action and are not meaningful b0 switching results. Means weight settings equally.']
    lines+=['']+markdown_table(main[['setting','mse','mae','relative_mse_b0','relative_mae_b0','windows']])
    lines+=['','## 3. Parameter / Configuration Search','',
        'History4: recent shift, last-point innovation, period24 difference RMS/correlation. Activation8 adds normalized output message RMS, branch cycle-level difference RMS, write cosine and cancellation. No task/channel/time identifiers, future labels or oracle alignment enter features. Fit-only robust preprocessing; L2=0.01; weighted logistic loss uses absolute MSE advantage with per-setting normalization. Calibration scans19 score quantiles plus all-on/all-off; selection is frozen before evaluation metrics.',
        'Both selected thresholds are all-on. The selected diagnostic action equals native TP-Add; the retained production architecture remains independent b0. Learned selectors have5/9 coefficients, but full feature/forward latency was not benchmarked because effect screening failed; no full efficiency-pass claim.','']
    lines+=markdown_table(pd.DataFrame(cal))
    lines+=['','## 4. Error Distribution','',
        'AUC indicates unweighted ranking of whether off is better. It is descriptive only: expected MSE and misclosed compensation determine adoption.','']
    lines+=markdown_table(auc_table)
    lines+=['','Fixed q95 sensitivity below is not selected for deployment and is not tuned on evaluation.','']
    lines+=markdown_table(sensitivity_table)
    lines+=['','History q95 loses up to %.2f%% of beneficial compensation and %.2f%% of the strongest1%% on evaluation; activation q95 reduces this risk but still yields no average MSE gain.' % (
         max(r['lost_benefit_fraction'] for r in sensitivity if r['selector']=='history4')*100,
         max(r['lost_extreme_benefit_fraction'] for r in sensitivity if r['selector']=='history4')*100),
        'Activation q95 loses up to %.2f%% of beneficial compensation and %.2f%% of the strongest1%%; mean relative evaluation MSE is %+.6f%% versus native.' % (
         max(r['lost_benefit_fraction'] for r in sensitivity if r['selector']=='history_activation8')*100,
         max(r['lost_extreme_benefit_fraction'] for r in sensitivity if r['selector']=='history_activation8')*100,
         np.mean([r['relative_mse'] for r in sensitivity if r['selector']=='history_activation8'])),
        '## 5. Horizon-wise Error','',
        'Every decision controls one whole forecast per sample/channel; no per-step routing. Results retain horizon identifiers. The tables below report forecast-third MSE for plotted ranked snapshots only, not population-wide horizon error.','',
        '## 6. High-Error Selection','',
        'Per setting Top3 baseline MAE, positive regression and negative improvement; same-channel starts separated by at least horizon within category. All requests retained in run.yaml/NPZ; ten highest-regression setting representatives plotted.','',
        '## 7. Case Analysis','']
    for case,rel in zip(chosen,refs):
        m=case['metadata'];row=next(r for r in case_stats if all(r[k]==m[k] for k in ('setting','category','rank','sample_id','channel')))
        lines += [f"### {m['setting']} sample{m['sample_id']} channel{m['channel']}",'',
                  f"Program-selected {m['category']} rank{m['rank']}; target {m['time_range']}. "
                  f"b0/native/off MAE={m['baseline_mae']:.6f}/{m['native_mae']:.6f}/{m['off_mae']:.6f}. "
                  'The selected action leaves the message enabled. This snapshot compares separately trained b0 and TP-Add; it does not isolate the causal effect of joint-training adaptation.','']
        lines+=markdown_table(pd.DataFrame([{'prediction':label,**{f'third{i}_mse':row[f'{label}_third{i}_mse'] for i in (1,2,3)}} for label in ('baseline','candidate','off')]))
        lines += ['',f'![{Path(rel).stem}]({rel})','']
    lines+=['## 8. Repeated Observable Patterns','',
        f"All selected decisions leave interaction enabled, so misclosed-benefit loss is zero but oracle recovery is zero. Across the {len(cases)} category-ranked snapshots (duplicates across categories retained), candidate standard deviation exceeds truth in {sum(r['candidate_std']>r['truth_std'] for r in case_stats)}; candidate exceeds b0 standard deviation in {sum(r['candidate_std']>r['baseline_std'] for r in case_stats)}. These are selected-case descriptions, not population frequencies.",
        'Activation features improve calibration discrimination (mean AUC about0.567 versus history0.492), then weaken on later validation (about0.525 versus0.505). This supports weak temporal transfer, not proof that all historical information is insufficient.',
        '## 9. Objective Defect Summary','',
        'Oracle on/off remains useful on the evaluation block, but the tested shared linear features do not extract that benefit. Activation q95 is almost neutral; history q95 miscloses useful ETT compensation. Limitations include in-sample predictor behavior during fit/cal, original validation checkpoint selection, overlapping windows, linear features and fixed regularization. More complex Gates or joint training were not tested, and are not justified by this screen.',
        '## 10. Experiment Scope','',
        f'{len(data)} extracted sample/channel records across fit/cal/eval; {len(errors)} evaluation rows; {len(cases)} ranked cases;12 settings. Zero predictor training, zero new test inference. Independent checkpoint/state and forward replay checks passed.',
        'First case-replay run verified all12 settings but failed saving because the remote audit directory was absent. Directory creation was fixed; full replay rerun with original weights. No training/evaluation metrics changed.',
        '## 11. Frozen Rubric','', '```json',json.dumps(run['analysis']['decision'],indent=2),'```','']
    report='\n'.join(lines);(args.audit/'objective_error_analysis.md').write_text(report)
    with zipfile.ZipFile(args.audit/'objective_error_analysis.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for rel in ['objective_error_analysis.md',*refs]:archive.write(args.audit/rel,rel)
    assert set(errors.setting)==set(main.setting)=={c['metadata']['setting'] for c in cases}
    setting_sets=[{p['setting'] for p in run['experiment']['settings']},
                  {p['setting'] for p in run['execution']['settings']},
                  {p['setting'] for p in run['selection']['selected_configs']},set(results.setting)]
    assert all(s==set(errors.setting) for s in setting_sets)
    for p in freeze['source_parts']:
        assert digest(args.scratch/'parts'/(p['setting']+'.csv'))==p['csv_sha256']
        assert p['roles']['fit']['end']<p['roles']['cal']['start']
        assert p['roles']['cal']['end']<p['roles']['eval']['start']
        assert p['state_unchanged'] and p['checks']['single_forward_off_replay_max_abs']<5e-5
        assert digest(Path(__file__).with_name('core.py'))==p['core_sha256']
        assert digest(Path(__file__).with_name('extract.py'))==p['extract_sha256']
    assert freeze['evaluation_opened'] is False
    for setting,g in data[data.role=='eval'].groupby('setting'):
        for name in ('native','off','b0','oracle',*freeze['selectors']):
            if name in freeze['selectors']:
                action=score(g,freeze['selectors'][name])>=freeze['selectors'][name]['threshold']
            elif name=='oracle':action=(g.off_mse<g.candidate_mse).to_numpy()
            else:action=np.full(len(g),name=='off',bool)
            expected=summary(g,action)
            if name=='b0':expected.update(mse=g.baseline_mse.mean(),mae=g.baseline_mae.mean())
            record=results[(results.setting==setting+'_eval')&(results.config_id==name)].iloc[0]
            for metric in ('mse','mae'):
                np.testing.assert_allclose(expected[metric],record[metric],rtol=1e-10,atol=1e-10)
    for setting,g in errors.groupby('setting'):
        expected=pick_cases(g,int(g.horizon.iloc[0]),top_k=3)
        actual=[c['metadata'] for c in cases if c['metadata']['setting']==setting]
        key=lambda m:(m['category'],m['rank'],m['sample_id'],m['channel'])
        assert {key(m) for m in expected}=={key(m) for m in actual}
        record=main[main.setting==setting].iloc[0]
        for metric in ('mse','mae'):np.testing.assert_allclose(g['candidate_'+metric].mean(),record[metric],atol=1e-9)
    for case in cases:
        for name in ('baseline','candidate','off'):
            e=case[name].astype(np.float64)-case['truth'].astype(np.float64)
            for metric,v in [('mse',np.square(e).mean()),('mae',np.abs(e).mean())]:
                np.testing.assert_allclose(v,case['metadata'][name+'_'+metric],atol=5e-5,rtol=5e-4)
    with zipfile.ZipFile(args.audit/'objective_error_analysis.zip') as archive:
        assert re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)==refs
        assert {p.relative_to(args.audit).as_posix() for p in figure_dir.iterdir()}==set(refs)
        assert not any((args.audit/rel).is_symlink() for rel in refs)
        assert set(archive.namelist())=={'objective_error_analysis.md',*refs}
        for rel in archive.namelist():assert archive.read(rel)==(args.audit/rel).read_bytes()
    assert {p.name for p in args.audit.iterdir()}=={'run.yaml','results.csv','sample_errors.csv','selected_cases.npz',
            'objective_error_analysis.md','objective_error_analysis.zip','figures'}
    run['validation']={'status':'passed','results_checked':True,'ranking_and_cases_checked':True,
             'report_and_archive_checked':True,'directory_and_settings_checked':True,
             'sample_rows':len(errors),'case_records':len(cases),'source_rows':len(data)}
    (args.audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    save_json(args.scratch/'audit_complete.json',{**run['validation'],'artifact_hashes':{
        str(p.relative_to(args.audit)):digest(p) for p in args.audit.rglob('*') if p.is_file()}})
    print(json.dumps(run['validation']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    main(parser.parse_args())
