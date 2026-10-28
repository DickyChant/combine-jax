"""Native import boundary for scalar tensor batches; not a full PDF importer yet."""
from pathlib import Path
import hashlib
from importlib.metadata import version

from .bundle import TensorBundle
from .scalar import SCALAR_CLASSES, dependency_flags, import_scalars


def from_workspace(workspace, datacard):
    """Extract once with the certified registry, then lower to scalar array batches.

    ROOT/JAX dependencies are confined to this import boundary. The resulting
    TensorBundle and its serialization need neither framework to inspect or load.
    Non-scalar likelihood terms are inventoried explicitly, never omitted from a
    claimed complete objective; this artifact has kind scalar_program.
    """
    from optparse import OptionParser
    from HiggsAnalysis.CombinedLimit.DatacardParser import parseCard, addDatacardParserOptions
    from ..graph import walk
    from ..likelihood import from_workspace as oracle_export

    parser = OptionParser()
    addDatacardParserOptions(parser)
    options, _ = parser.parse_args([])
    datacard = Path(datacard)
    with datacard.open() as stream:
        card = parseCard(stream, options)
    exported = oracle_export(workspace)
    if set(card.bins) != {c.name for c in exported.channels}:
        raise ValueError('Datacard/workspace channel sets differ')
    nodes = {name:(node.class_name, node.inputs, dict(node.attrs)) for name,node in exported.nodes.items()}
    observables = {name for channel in exported.channels for name in channel.observables}
    profiles = {name for profile in exported.profiles for name in profile.parameters}
    raw = walk(workspace.obj('ModelConfig').GetPdf())
    flags = dependency_flags(nodes, exported.parameters, observables=observables, profiles=profiles)
    roots = sorted(n for n in nodes if nodes[n][0] in SCALAR_CLASSES and not any(flags[n][1:]))
    bundle = import_scalars(nodes, exported.parameters, roots,
        fixed_value=lambda n:float(raw[n].getVal()), observables=observables, profiles=profiles,
        provenance=dict(card=str(datacard), card_sha256=hashlib.sha256(datacard.read_bytes()).hexdigest(),
            combine_version=version('higgs-combine-tool'), root_version=str(__import__('ROOT').gROOT.GetVersion()),
            scalar_extraction='Existing certified registry; fixed scalars evaluated natively at import',
            channels=[c.name for c in exported.channels],
            pending_channel_pdf_roots=[c.pdf for c in exported.channels],
            pending_constraint_roots=list(exported.constraints),
            profiled_parameters=sorted(profiles), discrete_choices=exported.discrete_choices,
            mask_configuration=exported.mask_configuration))
    return bundle
