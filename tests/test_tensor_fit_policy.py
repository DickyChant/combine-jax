from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.fit_policy import (
    POLICY, check_native_fit_state, combine_v11_fit_context, mass_cli_arguments,
)
from combine_jax.tensor.model import from_workspace, evaluate_model_numpy
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates, validate_parameter_info


CARD = Path(__file__).parent/'cards/counting.txt'


def native_probe(root):
    if not hasattr(root, 'combine_jax_fit_policy_test'):
        assert root.gInterpreter.Declare('''
        #include "HiggsAnalysis/CombinedLimit/interface/MultiDimFit.h"
        namespace combine_jax_fit_policy_test {
        class Probe : public MultiDimFit {
        public:
            using FitterAlgoBase::optimizeBounds;
            using FitterAlgoBase::restoreBounds;
        };
        }
        ''')
    root.runtimedef.set('OPTIMIZE_BOUNDS',1)
    root.runtimedef.set('UNBOUND_GAUSSIANS',0)
    return root.combine_jax_fit_policy_test.Probe()


def test_effective_bounds_native_nll_and_neutral_roundtrip(counting_path, tmp_path):
    root = load_caching_nll()
    with open_workspace(counting_path) as workspace:
        workspace.var('lumi').setRange(-.04,.04)
        workspace.var('r').setAttribute('optimizeBounds', True)  # not a nuisance
        model = from_workspace(workspace,CARD,fit_policy=POLICY)
        info = model_parameter_info(model); names = info.metadata['names']
        i = names.index('lumi'); r = names.index('r')
        assert workspace.var('lumi').getMin()==-.04 and workspace.var('lumi').getMax()==.04
        assert info.arrays['lower'][i]==-np.inf and info.arrays['upper'][i]==np.inf
        assert info.arrays['stored_lower'][i]==-.04 and info.arrays['stored_upper'][i]==.04
        assert info.arrays['lower'][r]==workspace.var('r').getMin()
        check_native_fit_state(workspace,model)
        probe = native_probe(root);config=workspace.obj('ModelConfig')
        probe.optimizeBounds(workspace,config)
        for j,n in enumerate(names):
            assert (workspace.var(n).getMin(),workspace.var(n).getMax())==(info.arrays['lower'][j],info.arrays['upper'][j])
        native=root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('data_obs'),config.GetNuisanceParameters())
        native.clearConstantZeroPoint();native.clearZeroPoint()
        errors=[]
        for value in (-.5,0.,.3):
            point=info.arrays['initial'].copy();point[i]=value
            for n,v in zip(names,point,strict=True):workspace.var(n).setVal(float(v))
            errors.append(abs(evaluate_model_numpy(model,point)-native.getVal()))
            fit_coordinates(info,initial=point)  # points outside stored bounds are allowed
        assert max(errors)<1e-10
        del native
        workspace.var('lumi').setVal(0.)
        probe.restoreBounds(workspace,config)
        model.save(tmp_path/'model')
        loaded=TensorBundle.load(tmp_path/'model');check_native_fit_state(workspace,loaded)
        script='''import sys
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates
m=TensorBundle.load(sys.argv[1]);fit_coordinates(model_parameter_info(m))
assert not any(n in sys.modules for n in ('ROOT','jax','torch'))
'''
        subprocess.run([sys.executable,'-c',script,str(tmp_path/'model')],check=True,capture_output=True)
        print('FIT_POLICY_NATIVE_NLL_MAX_ERROR',max(errors),flush=True)


def test_restore_on_failure_and_frozen_nuisance(counting_path):
    load_caching_nll()
    with open_workspace(counting_path) as workspace:
        nuisance=workspace.var('bkg');nuisance.setConstant(True)
        before=(nuisance.getMin(),nuisance.getMax())
        with pytest.raises(FileNotFoundError):from_workspace(workspace,'missing-card.txt',fit_policy=POLICY)
        assert before==(nuisance.getMin(),nuisance.getMax()) and nuisance.isConstant()
        model=from_workspace(workspace,CARD,fit_policy=POLICY)
        state=check_native_fit_state(workspace,model)
        assert state['fixed_real_values']['bkg']==0.
        nuisance.setVal(.2)
        with pytest.raises(ValueError,match='fit state differs'):check_native_fit_state(workspace,model)


def test_preflight_rejects_wrong_state_or_metadata(counting_path):
    load_caching_nll()
    with open_workspace(counting_path) as workspace:
        legacy=from_workspace(workspace,CARD)
        with pytest.raises(ValueError,match='lacks explicit'):check_native_fit_state(workspace,legacy)
        model=from_workspace(workspace,CARD,fit_policy=POLICY)
        info=model_parameter_info(model);i=info.metadata['names'].index('lumi')
        lower=info.arrays['lower'].copy();lower[i]=info.arrays['stored_lower'][i]
        with pytest.raises(ValueError,match='Effective bounds'):validate_parameter_info(TensorBundle(info.metadata,{**info.arrays,'lower':lower}))
        workspace.var('lumi').setRange(-.1,.1)
        with pytest.raises(ValueError,match='stored bounds'):check_native_fit_state(workspace,model)
        workspace.var('lumi').setAttribute('optimizeBounds',False)
        with pytest.raises(ValueError,match='fit state differs'):check_native_fit_state(workspace,model)


def test_explicit_mass_argument():
    assert mass_cli_arguments({'mass_argument':125.})==['--mass','125.0']
    assert mass_cli_arguments({'mass_argument':None})==[]
    for mass in (float('nan'),float('inf'),125.0000001):
        with pytest.raises(ValueError,match='not exactly representable'):mass_cli_arguments({'mass_argument':mass})


def test_marked_zero_width_nuisance_becomes_free(counting_path):
    load_caching_nll()
    with open_workspace(counting_path) as workspace:
        workspace.var('bkg').setRange(0.,0.)
        assert workspace.var('bkg').getAttribute('optimizeBounds') and not workspace.var('bkg').isConstant()
        model=from_workspace(workspace,CARD,fit_policy=POLICY)
        info=model_parameter_info(model);index=info.metadata['names'].index('bkg')
        assert info.arrays['stored_lower'][index]==info.arrays['stored_upper'][index]==0.
        assert info.arrays['lower'][index]==-np.inf and info.arrays['upper'][index]==np.inf
        assert index in fit_coordinates(info)['free_indices']
        check_native_fit_state(workspace,model)
