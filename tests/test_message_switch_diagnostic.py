import numpy as np
import pandas as pd
from scripts.message_switch_diagnostic.core import temporal_roles, fit_selector, score, summary, calibration_choice


def test_complete_temporal_supports_do_not_overlap():
    roles,cut=temporal_roles(8640,3600,720,720)
    assert roles['fit'][-1]+1440<=cut<=roles['cal'][0]
    assert roles['cal'][-1]+1440<=8640
    assert 8640-720+roles['eval'][0]>=8640


def test_risk_selector_uses_only_named_features():
    rng=np.random.default_rng(4)
    x=rng.normal(size=200)
    f=pd.DataFrame({'x':x,'candidate_mse':1+np.maximum(-x,0),
                    'off_mse':1+np.maximum(x,0),'baseline_mse':np.ones(200),'setting':'a'})
    fitted=fit_selector(f,['x'])
    assert score(f.iloc[[np.argmin(x)]],fitted)[0]>score(f.iloc[[np.argmax(x)]],fitted)[0]
    changed=f.copy();changed['off_mse']=100;changed['setting']='different';changed['sample_id']=999
    np.testing.assert_array_equal(score(f,fitted),score(changed,fitted))


def test_misclosing_large_compensation_is_expensive():
    f=pd.DataFrame({'candidate_mse':[1.,1.,3.],'off_mse':[101.,2.,1.],
        'candidate_mae':[1.,1.,1.7],'off_mae':[10.,1.4,1.],
        'baseline_mse':[1.,1.,1.],'baseline_mae':[1.,1.,1.]})
    bad=summary(f,np.array([True,False,True]))
    assert bad['lost_benefit_fraction']>0.99 and bad['lost_extreme_benefit_fraction']==1
    good=summary(f,np.array([False,False,True]))
    assert good['lost_benefit_fraction']==0 and good['oracle_recovery']==1


def test_calibration_can_fall_back_to_all_on():
    f=pd.DataFrame({'candidate_mse':np.ones(30),'off_mse':np.ones(30)*4,
                    'candidate_mae':np.ones(30),'off_mae':np.ones(30)*2,
                    'baseline_mse':np.ones(30),'baseline_mae':np.ones(30),'setting':'a'})
    selected,attempts=calibration_choice(f,np.linspace(0,1,30))
    assert selected['name']=='all_on' and len(attempts)==21
