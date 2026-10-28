"""Narrow, hash-bound authorization for the user's combined-MX3000 unblinding."""
import hashlib
from pathlib import Path
import socket

WORKSPACE_SHA256 = '9be1f1e00341774842dabb6d1fdfa8fa9b86a01f846bd0e0b6ce2d81c2902868'
CARD_SHA256 = '72280b67dd90c5ddb110e0362f5cc535ca48a5cf5f1fefa868432355358cc943'
CATEGORIES = {'pdfindex_boosted_cat0_13TeV': 2, 'pdfindex_boosted_cat1_13TeV': 1,
              'pdfindex_resolved_cat0_13TeV': 3}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def authorize(manifest, workspace=None, card=None):
    if socket.gethostname().split('.')[0] == 'lxplus905':
        raise RuntimeError('Forbidden execution node')
    if (manifest.get('mass') != 3000 or manifest.get('run') != 'observed'
            or manifest.get('combination') != 'HHVV_v4_final'
            or manifest.get('workspace_sha256') != WORKSPACE_SHA256
            or manifest.get('card_sha256') != CARD_SHA256
            or manifest.get('fixed_categories') != CATEGORIES
            or manifest.get('template_convention') != 'v10_density'
            or manifest.get('float64') is not True or manifest.get('static_guard_gib') != 8.):
        raise ValueError('Only the explicitly authorized combined HHVV MX3000 workspace may be unblinded')
    if workspace is not None and sha(workspace) != WORKSPACE_SHA256:
        raise ValueError('Unauthorized observed workspace bytes')
    if card is not None and sha(card) != CARD_SHA256:
        raise ValueError('Unauthorized observed card bytes')


def check_workspace(ws):
    assert ws.var('MX').getVal() == 3000. and ws.var('MX').isConstant()
    assert ws.var('MH').getVal() == 125. and ws.var('MH').isConstant()
    assert [v.GetName() for v in ws.obj('ModelConfig').GetParametersOfInterest()] == ['r_hhvv']
    assert ws.obj('ModelConfig').GetPdf().indexCat().numTypes() == 459
    for name, index in CATEGORIES.items():
        assert ws.cat(name).getIndex() == index
