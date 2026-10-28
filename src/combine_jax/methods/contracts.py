"""Input contracts for the first, continuous expected-limit method."""


def require_continuous_model(metadata):
    choices = metadata.get('provenance', {}).get('discrete_choices', {})
    if choices:
        raise NotImplementedError('AsymptoticLimits discrete RooMultiPdf profiling is not validated: '
                                  + ', '.join(sorted(choices)))


def validate_expected_model(model, poi, *, run=None, no_fit_asimov=False, discrete_policy='profile'):
    if discrete_policy not in ('profile', 'fixed'):
        raise ValueError('Unknown expected-limit discrete policy')
    choices = model.metadata.get('provenance', {}).get('discrete_choices', {})
    record = model.metadata.get('asymptotic_asimov', {})
    if choices and discrete_policy == 'fixed':
        if (record.get('discrete_policy') != 'fixed'
                or record.get('fixed_discrete_choices') != choices):
            raise ValueError('Fixed-discrete limit requires an explicit matching preparation record')
    else:
        require_continuous_model(model.metadata)
    if model.metadata.get('parameter_info', {}).get('fit_policy', {}).get('nuisance_bounds_removed'):
        raise ValueError('AsymptoticLimits preserves workspace nuisance bounds; reimport this FitterAlgoBase-policy artifact')
    if record.get('signal_strength') != 0. or record.get('poi') != poi:
        raise ValueError('Limit input must be an explicitly prepared background-only Asimov model')
    mode = record.get('mode')
    if mode not in ('prefit', 'conditional_background_fit'):
        raise ValueError('Unknown background-Asimov preparation mode')
    dataset = model.metadata.get('provenance', {}).get('dataset', {})
    if not record.get('dataset_identity') or record['dataset_identity'] != dataset.get('identity'):
        raise ValueError('Limit Asimov identity mismatch')
    if record.get('dataset') != dataset.get('name'):
        raise ValueError('Limit Asimov dataset name mismatch')
    if run not in (None, 'blind', 'expected'):
        raise ValueError('Only blind/expected limits are supported')
    if run is not None or no_fit_asimov:
        wanted = 'prefit' if run == 'blind' or no_fit_asimov else 'conditional_background_fit'
        if mode != wanted:
            raise ValueError(f'Prepared model is {mode}; requested Asimov mode is {wanted}. Prepare a new model.')
    return record
