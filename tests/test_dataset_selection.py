import pytest

from combine_jax.datasets import dataset_selection


@pytest.mark.parametrize('analysis', ['hhvv', 'blind'])
def test_hig_authorization_does_not_unblind_other_analyses(analysis):
    with pytest.raises(ValueError, match='prohibited'):
        dataset_selection('data_obs', analysis=analysis, kind='observed')


def test_named_hhvv_model_cannot_bypass_blinding():
    with pytest.raises(ValueError, match='prohibited'):
        dataset_selection('data_obs', analysis='hig23006', kind='observed', parameter_names=['r_hhvv'])


def test_observed_cannot_be_relabelled_asimov():
    with pytest.raises(ValueError, match='labelled'):
        dataset_selection('data_obs', analysis='hig23006', kind='asimov')


@pytest.mark.parametrize('analysis', ['hig23006', 'synthetic'])
def test_observed_provenance(analysis):
    selected = dataset_selection('data_obs', analysis=analysis, kind='observed')
    assert selected['observed_data_accessed'] and selected['name'] == 'data_obs'
    assert selected['analysis'] == analysis


def test_hhvv_asimov_retained():
    assert not dataset_selection('asimov_prefit', analysis='hhvv', kind='asimov')['observed_data_accessed']
