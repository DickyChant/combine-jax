"""Explicit dataset selection and per-analysis blinding policy.

This policy is independent of ROOT and the numerical backend. Synthetic data
may exercise either path; authorizing HIG observed data never authorizes HHVV.
"""


def dataset_selection(dataset, *, analysis, kind, parameter_names=()):
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError('An explicit dataset name is required')
    if kind not in ('asimov', 'observed'):
        raise ValueError('Dataset kind must be asimov or observed')
    if analysis not in ('hig23006', 'hhvv', 'blind', 'synthetic'):
        raise ValueError('An explicit supported analysis policy is required')
    if dataset == 'data_obs' and kind != 'observed':
        raise ValueError('data_obs cannot be labelled Asimov')
    if kind == 'observed' and (analysis in ('hhvv', 'blind') or 'r_hhvv' in parameter_names):
        raise ValueError('Observed data are prohibited for this blind analysis')
    return dict(name=dataset, kind=kind, analysis=analysis,
                observed_data_accessed=kind == 'observed',
                authorization='SQ HIG-23-006 only' if analysis == 'hig23006' and kind == 'observed'
                else 'synthetic fixture' if analysis == 'synthetic' else 'blinded Asimov')
