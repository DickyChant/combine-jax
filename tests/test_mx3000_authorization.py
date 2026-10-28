"""The explicit exception must never authorize other masses or subchannels."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('authorization', Path('scripts/hhvv_mx3000_authorization.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record():
    return dict(mass=3000, run='observed', combination='HHVV_v4_final',
                workspace_sha256=module.WORKSPACE_SHA256, card_sha256=module.CARD_SHA256,
                fixed_categories=module.CATEGORIES.copy(), template_convention='v10_density',
                float64=True, static_guard_gib=8.)


def test_exact_exception_is_accepted():
    module.authorize(record())


@pytest.mark.parametrize('key,value', [('mass',2000),('mass',4000),('combination','VV4q'),
    ('workspace_sha256','different'),('card_sha256','different'),('float64',False),
    ('template_convention','native'),('static_guard_gib',16.),('fixed_categories',{})])
def test_other_inputs_stay_blind(key, value):
    r = record(); r[key] = value
    with pytest.raises(ValueError, match='Only the explicitly authorized'):
        module.authorize(r)
