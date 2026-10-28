import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_asimov_native_reference import verified_refinements
from native_fd_refinement import REFINEMENT_FRACTIONS,REFERENCE_SPREAD,select_reference
from tensor_benchmark_common import sha


@pytest.fixture
def reference(tmp_path):
    native=dict(workspace_sha256='workspace',generation_report_sha256='generation',nll_arrays_sha256='values')
    probes=[dict(fraction=h,step=h,derivative=1.,stencil='central',direction=1.,status='valid') for h in REFINEMENT_FRACTIONS]
    case=dict(coordinate=0.,probes=probes,**select_reference(probes))
    report=dict(status='refinement_complete_pending_audit',root_only=True,dataset='asimov_prefit',
        workspace_sha256='workspace',generation_report_sha256='generation',native_values_sha256='values',
        root_version='6.40.00',combine_version='11.0.0',
        script_sha256=sha('scripts/refine_asimov_native_reference.py'),helper_sha256=sha('scripts/native_fd_refinement.py'),
        base_fd_sha256=sha('scripts/native_fd_scan.py'),reference_spread=REFERENCE_SPREAD,fractions=list(REFINEMENT_FRACTIONS),
        cases=[dict(point=0,index=0,name='p',reference=case,canonical_reference_error=0.,canonical_nll=10.)])
    return tmp_path/'refinement.json',report,native


def verify(fixture,report):
    path,_,native=fixture;path.write_text(json.dumps(report))
    return verified_refinements([path],native,np.array(['p']),np.array([[0.]]),np.array([10.]))


def test_refinement_binds_native_point_and_preserves_independent_selection(reference):
    result=verify(reference,reference[1]);assert result[(0,0)]['case']['status']=='converged'


@pytest.mark.parametrize('field',['dataset','workspace_sha256','generation_report_sha256','native_values_sha256','helper_sha256'])
def test_wrong_dataset_or_evidence_cannot_replace_reference(reference,field):
    report=copy.deepcopy(reference[1]);report[field]='different'
    with pytest.raises(AssertionError):verify(reference,report)


def test_ad_selected_step_or_different_parameter_cannot_replace_reference(reference):
    report=copy.deepcopy(reference[1]);report['cases'][0]['reference']['selected_first_probe_index']=1
    with pytest.raises(AssertionError,match='without AD'):verify(reference,report)
    report=copy.deepcopy(reference[1]);report['cases'][0]['name']='other'
    with pytest.raises(AssertionError):verify(reference,report)
