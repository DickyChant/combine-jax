import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from combine_jax.methods.asymptotic import QUANTILES, expected_limits
from combine_jax.methods.root_output import audit_crossings, write_limit_tree
from combine_jax.runtime.options import add_limits_command, request


def parse(*extra):
    parser = argparse.ArgumentParser()
    add_limits_command(parser.add_subparsers(dest='command'))
    return parser.parse_args(['limits','--workspace','workspace.root','--poi','r',*extra])


def test_public_cli_blinding_and_combine_options():
    config = request(parse('--run','expected','--noFitAsimov','--setParameters','MX=2000,MX=4000',
        '--setParameterRanges','r=0,20:MX=1000,6000','--freezeParameters','MX','--mass','125'))
    assert config['parameters'] == {'MX':4000.}
    assert config['ranges'] == {'r':[0.,20.],'MX':[1000.,6000.]}
    assert config['no_fit_asimov'] and config['signal_strength']==0.
    assert config['float64'] and config['static_guard_gib']==8.
    with pytest.raises(ValueError,match='pre-fit only'):
        request(parse('--run','expected'))
    with pytest.raises(SystemExit):
        parse('--run','observed')
    with pytest.raises(ValueError,match='exactly one POI'):
        request(parse('--poi','r,r2'))


def test_root_output_is_standard_expected_ttree_and_unresolved_is_nan(tmp_path):
    import uproot
    result = expected_limits(lambda mu:.5*mu*mu,0.,10.)
    rows = audit_crossings(result)
    assert all(row['status']=='resolved' for row in rows)
    # Simulate the known missing-lower-bracket case. The numerical midpoint is
    # diagnostic only and must never become a reported limit.
    result['trace'] = [r for r in result['trace'] if r['quantile'] != .025]
    rows = audit_crossings(result)
    assert rows[0]['limit'] is None and rows[0]['status']=='unresolved_crossing'
    output = tmp_path/'higgsCombineTest.AsymptoticLimits.mH125.0.root'
    write_limit_tree(output,rows,mass=125,seed=4,wall_seconds=2.)
    with output.open('rb') as stream, uproot.open(stream) as handle:
        tree = handle['limit']
        assert tree.classname=='TTree'
        arrays = tree.arrays(library='np')
        assert len(arrays['limit']) == 5
        assert np.isnan(arrays['limit'][0])
        np.testing.assert_array_equal(arrays['quantileExpected'],np.array(QUANTILES,np.float32))
        assert all(arrays['iSeed']==4) and all(arrays['mh']==125)
    with pytest.raises(FileExistsError):
        write_limit_tree(output,rows)


def test_workspace_only_import_matches_card_checked_import(counting_path):
    import jax
    from combine_jax.root_io import open_workspace
    from combine_jax.tensor.model import from_workspace, make_model_function
    with open_workspace(counting_path) as workspace:
        checked = from_workspace(workspace,'tests/cards/counting.txt')
        standalone = from_workspace(workspace)
    assert standalone.metadata['provenance']['channel_source']=='workspace_modelconfig'
    assert standalone.metadata['parameter_names']==checked.metadata['parameter_names']
    for name in checked.arrays:
        np.testing.assert_array_equal(standalone.arrays[name],checked.arrays[name])
    point = standalone.arrays['parameter_info__initial']
    values = [jax.value_and_grad(make_model_function(model))(point,jax.device_put(dict(model.arrays)))
              for model in (checked,standalone)]
    np.testing.assert_array_equal(values[0][0],values[1][0])
    np.testing.assert_array_equal(values[0][1],values[1][1])


def test_gpu_check_rejects_condor_mig_allocation_before_cuda(monkeypatch):
    from combine_jax.runtime.fit import verify
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','MIG-b05acb82-d929-50be-bab0-285bc5b18f63')
    with pytest.raises(RuntimeError,match='MIG allocations are unsupported'):
        verify('gpu')
