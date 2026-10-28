"""Read-only HHVV workspace/card census; no likelihood tracing or minimization.

ROOT server trees are the primary structural definition. The existing trusted
numeric export supplies exact data rows, template geometry and active parameters.
Its source and configuration are checked against the workspace and manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import pickle
import resource
import subprocess
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def tree_fingerprints(nodes, labels=None):
    """Ordered class trees, with DAG reuse unfolded; names never enter the hash."""
    result, visiting = {}, set()

    def visit(name):
        if name in visiting:
            raise ValueError(f'Cycle at {name}')
        if name not in result:
            visiting.add(name)
            cls, children = nodes[name][:2]
            label = labels[name] if labels is not None else cls
            result[name] = fingerprint([label, [visit(child) for child in children]])
            visiting.remove(name)
        return result[name]

    for name in nodes:
        visit(name)
    return result


def descendants(nodes, root):
    seen, pending = set(), [root]
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(nodes[name][1])
    return seen


def leaf_alias_fingerprint(nodes, root, leaf_names):
    """Record repeated parameter identity separately from the unfolded class tree."""
    aliases, cache = {}, {}

    def visit(name):
        if name in leaf_names:
            if name not in aliases:
                aliases[name] = len(aliases)
            return fingerprint(['leaf', nodes[name][0], aliases[name]])
        if name not in cache:
            cache[name] = fingerprint([nodes[name][0], [visit(c) for c in nodes[name][1]]])
        return cache[name]

    return visit(root)


def semantic_labels(nodes):
    """Conservative refinements, not a certificate that batched evaluation is exact.

    Ignore numeric buffers/leaf values, retain formula literals and mode settings.
    Resolve named direct-input roles to positions. Full alias patterns are separate.
    """
    labels = {}
    flags = {'algorithm', 'conditional', 'logarithmic', 'vertical_type', 'extended',
             'no_rounding', 'protect_negative', 'sign', 'region', 'smooth_region',
             'pdf_floor', 'integral_floor'}
    references = {'pdfs', 'coefs', 'functions', 'symmetric', 'asymmetric', 'other',
                  'morphs', 'axes', 'fields', 'observable_name', 'bin_parameters'}
    for name, (cls, children, attrs) in nodes.items():
        def ref(value):
            if isinstance(value, (tuple, list)):
                return [ref(v) for v in value]
            if value is None:
                return None
            if value in children:
                return ['input', children.index(value)]
            # E.g. a conditional normalization observable deeper in the graph.
            if value in nodes:
                return ['descendant_class', nodes[value][0]]
            raise ValueError(f'Unresolved semantic reference {name}: {value}')

        modes = {k: attrs[k] for k in flags & attrs.keys()}
        modes.update({k: ref(attrs[k]) for k in references & attrs.keys()})
        if cls == 'RooFormulaVar':
            modes['formula_ast_with_literals'] = attrs['ast']
        if cls == 'RooRealVar':
            modes.update(constant=attrs['constant'], observable=attrs.get('observable', False))
        if cls == 'RooProdPdf':
            modes['normalisations'] = [(kind, ref(axes)) for kind, axes in attrs['normalisations']]
        if cls == 'CMSHistErrorPropagator':
            modes['bin_types'] = attrs['bin_types']
        if cls == 'RooMultiPdf':
            modes['selection'] = attrs['index']
        labels[name] = [cls, modes]
    return labels


def groups_for(channels, key, prefix):
    partitions = defaultdict(list)
    for channel in channels:
        partitions[channel[key]].append(channel)
    result = []
    ordered = sorted(partitions.items(), key=lambda item: (-len(item[1]), item[1][0]['name']))
    for i, (signature, members) in enumerate(ordered):
        identifier = f'{prefix}{i:03d}'
        for member in members:
            member[prefix+'_group'] = identifier
        rows = [c['data']['rows'] for c in members]
        result.append(dict(id=identifier, signature=signature, count=len(members),
            channels=[c['name'] for c in members],
            analyses=dict(sorted(Counter(c['analysis'] for c in members).items())),
            observable_dimensions=dict(sorted(Counter(c['observable_dimensions'] for c in members).items())),
            data_rows=dict(total=sum(rows), minimum=min(rows), maximum=max(rows),
                           histogram=dict(sorted(Counter(rows).items()))),
            sum_weights=sum(c['data']['sum_weights'] for c in members),
            parameter_counts=dict(sorted(Counter(len(c['parameter_indices']) for c in members).items())),
            semantic_variants=len({c['semantic_signature'] for c in members}),
            leaf_alias_variants=len({c['leaf_alias_signature'] for c in members})))
    assert sum(g['count'] for g in result) == len(channels)
    return result


def parse_card(path):
    from HiggsAnalysis.CombinedLimit.DatacardParser import parseCard, addDatacardParserOptions
    from optparse import OptionParser
    parser = OptionParser()
    addDatacardParserOptions(parser)
    options, _ = parser.parse_args([])
    with path.open() as stream:
        card = parseCard(stream, options)
    types = defaultdict(set)
    rows = []
    for name, nofloat, kind, args, effects in card.systs:
        types[name].add(kind)
        affected = sorted(channel for channel, procs in effects.items()
                          if any(value != 0 for value in procs.values())) if isinstance(effects, dict) else []
        rows.append(dict(name=name, type=kind, nofloat=nofloat, channels=affected))
    for name in card.flatParamNuisances:
        types[name].add('flatParam')
    rate_names = {spec[0][0] for specs in card.rateParams.values() for spec in specs}
    for name in rate_names:
        types[name].add('rateParam')
    for name in card.discretes:
        types[name].add('discrete')
    with path.open() as stream:
        first = stream.readline()
    analyses = [token.split('=', 1)[0] for token in first.split() if '=' in token]
    return card, types, dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size,
        parser='Combine v11 HiggsAnalysis.CombinedLimit.DatacardParser.parseCard, default options',
        channels=len(card.bins), process_columns=len(card.keyline), processes=len(card.processes),
        nuisance_rows=len(rows), nuisance_rows_by_type=dict(sorted(Counter(r['type'] for r in rows).items())),
        unique_systematic_names=len({r['name'] for r in rows}),
        flat_parameters=sorted(card.flatParamNuisances), rate_parameter_names=sorted(rate_names),
        rate_parameter_channel_process_bindings=len(card.rateParams), discrete=card.discretes,
        auto_mc_stats={k: list(v) for k,v in sorted(card.binParFlags.items())},
        nuisance_rows_detail=rows, analysis_prefixes=analyses)


def census(args):
    from combine_jax.graph import walk, names
    from combine_jax.root_io import open_workspace
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_bytes())
    model = manifest['models']['hhvv']
    # This is an existing, locally generated and hash-verified trusted pickle.
    export_hash = digest(args.export)
    prior_export = json.loads(args.export_reference.read_bytes())
    assert export_hash == prior_export['export_sha256']
    assert model['source'] == prior_export['source']
    with args.export.open('rb') as stream:
        payload = pickle.load(stream)
    assert payload['source'] == model['source']
    assert digest(args.workspace) == model['source']['sha256']
    assert version('higgs-combine-tool') == manifest['combine_version']
    card, card_types, card_report = parse_card(args.card)
    nodes = payload['nodes']
    translated = tree_fingerprints(nodes)
    semantic = tree_fingerprints(nodes, semantic_labels(nodes))
    parameters = payload['parameters']
    pindex = {p.name: i for i,p in enumerate(parameters)}
    profile_names = {n for profile in payload['profiles'] for n in profile.parameters}
    parameter_leaves = set(pindex) | profile_names
    classes = dict(sorted(Counter(n[0] for n in nodes.values()).items()))
    usage, constraint_usage = defaultdict(list), defaultdict(list)
    constraint_rows = []
    for name in payload['constraints']:
        deps = descendants(nodes, name)
        for par in deps & parameter_leaves:
            constraint_usage[par].append(name)
        constraint_rows.append(dict(name=name, class_name=nodes[name][0],
            parameters=sorted(deps & parameter_leaves),
            normalization_axes=list(payload['constraint_axes'].get(name, ())),
            inputs=list(nodes[name][1])))
    channels = []
    with open_workspace(args.workspace) as workspace:
        for name, settings in model['source']['configuration'].items():
            workspace.var(name).setVal(settings['value'])
            workspace.var(name).setConstant(settings['constant'])
        config = workspace.obj('ModelConfig')
        sim = config.GetPdf()
        raw_objects = walk(sim)
        raw_nodes = {n:(str(v.ClassName()), tuple(str(c.GetName()) for c in v.servers()))
                     for n,v in raw_objects.items()}
        raw_tree = tree_fingerprints(raw_nodes)
        raw_nuisances = names(config.GetNuisanceParameters())
        global_observables = names(config.GetGlobalObservables())
        root_observables = {}
        for n in sorted(names(config.GetObservables())):
            obj = workspace.var(n)
            if obj:
                bins = obj.getBinning()
                root_observables[n] = dict(class_name=str(obj.ClassName()), bins=int(obj.getBins()),
                    minimum=float(obj.getMin()), maximum=float(obj.getMax()),
                    edges=[float(bins.binLow(i)) for i in range(bins.numBins())]+[float(bins.highBound())])
        for name, pdf, axes, coordinates, weights, storage, masked in payload['channels']:
            assert pdf in raw_nodes and sim.getPdf(name), (name, pdf)
            assert raw_nodes[pdf][0] == nodes[pdf][0]
            full_pdf = str(sim.getPdf(name).GetName())
            assert pdf in descendants(raw_nodes, full_pdf)
            deps = descendants(nodes, pdf)
            local_parameters = deps & parameter_leaves
            for par in local_parameters:
                usage[par].append(name)
            indices = sorted(pindex[n] for n in local_parameters if n in pindex)
            grid_sizes = [len(np.unique(coordinates[a])) for a in axes]
            nrows = len(weights)
            unique_rows = len(np.unique(np.column_stack([coordinates[a] for a in axes]), axis=0)) if axes else int(nrows > 0)
            templates = Counter()
            for n in deps:
                cls, _, attrs = nodes[n]
                if 'edges' in attrs:
                    edges = attrs['edges']
                    bins = (len(edges)-1,) if isinstance(edges, np.ndarray) else tuple(len(e)-1 for e in edges)
                    templates[(cls, bins, attrs.get('active'))] += 1
            direct = nodes[pdf][2].get('pdfs', nodes[pdf][2].get('functions', ()))
            # Only an architectural family sketch; never an exact equivalence claim.
            shape_classes = Counter()
            def shapes(n):
                cls, inputs, attrs = nodes[n]
                if cls in ('RooAddPdf', 'RooRealSumPdf', 'RooProdPdf'):
                    for child in attrs.get('pdfs', attrs.get('functions', inputs)):
                        shapes(child)
                elif cls == 'RooMultiPdf':
                    shapes(attrs['selected'])
                else:
                    shape_classes[cls] += 1
            for n in direct:
                shapes(n)
            family_signature = fingerprint([len(axes), sorted(shape_classes)])
            analysis = next((a for a in sorted(card_report['analysis_prefixes'], key=len, reverse=True)
                             if name.startswith(a+'_')), 'unmatched')
            card_nuisances = Counter(t for p in local_parameters for t in card_types.get(p, ()))
            constraint_types = Counter(nodes[c][0] for p in local_parameters for c in constraint_usage[p])
            channels.append(dict(name=name, analysis=analysis, pdf=pdf, full_root_pdf=full_pdf,
                root_class=nodes[pdf][0], root_class_tree_signature=raw_tree[pdf],
                full_root_class_tree_signature=raw_tree[full_pdf],
                translated_class_tree_signature=translated[pdf], semantic_signature=semantic[pdf],
                leaf_alias_signature=leaf_alias_fingerprint(nodes, pdf, parameter_leaves),
                operator_family_signature=family_signature, operator_family_classes=dict(sorted(shape_classes.items())),
                observable_dimensions=len(axes), observables=list(axes), masked=bool(masked),
                data=dict(storage_class=storage, rows=nrows, nonzero_weight_rows=int(np.count_nonzero(weights)),
                    zero_weight_rows=int(np.count_nonzero(weights == 0)), sum_weights=float(np.sum(weights)),
                    nonunit_weight_rows=int(np.count_nonzero(weights != 1)),
                    fractional_weight_rows=int(np.count_nonzero(weights != np.round(weights))),
                    unique_coordinates_per_axis=grid_sizes, unique_coordinate_rows=unique_rows,
                    observed_cartesian_grid_cells=math.prod(grid_sizes),
                    covers_observed_cartesian_grid=unique_rows == math.prod(grid_sizes),
                    declared_observable_grid_cells=math.prod(root_observables[a]['bins'] for a in axes)),
                template_geometry=[dict(class_name=k[0], bins=list(k[1]), active=k[2], nodes=v)
                                   for k,v in sorted(templates.items(), key=lambda item:str(item[0]))],
                unique_dependency_nodes=len(deps),
                class_counts=dict(sorted(Counter(nodes[n][0] for n in deps).items())),
                process_count=len(card.exp[name]), process_names=sorted(card.exp[name]),
                parameter_indices=indices, profiled_parameters=sorted(local_parameters & profile_names),
                nuisance_card_types=dict(sorted(card_nuisances.items())),
                nuisance_constraint_types=dict(sorted(constraint_types.items()))))
        original_data = workspace.data('data_obs')
        original = dict(root_version=str(__import__('ROOT').gROOT.GetVersion()),
            simultaneous_class=str(sim.ClassName()), graph_nodes=len(raw_nodes),
            graph_classes=dict(sorted(Counter(n[0] for n in raw_nodes.values()).items())),
            nuisance_count=len(raw_nuisances), nuisance_names=sorted(raw_nuisances),
            globals=sorted(global_observables), pois=sorted(names(config.GetParametersOfInterest())),
            data_rows=int(original_data.numEntries()), data_sum_weights=float(original_data.sumEntries()),
            observables=root_observables)
    assert set(card.bins) == {c['name'] for c in channels}
    assert sum(c['data']['rows'] for c in channels) == original['data_rows']
    assert abs(sum(c['data']['sum_weights'] for c in channels)-original['data_sum_weights']) < 1e-6
    root_groups = groups_for(channels, 'root_class_tree_signature', 'root')
    full_groups = groups_for(channels, 'full_root_class_tree_signature', 'full_root')
    export_groups = groups_for(channels, 'translated_class_tree_signature', 'translated')
    semantic_groups = groups_for(channels, 'semantic_signature', 'semantic')
    family_groups = groups_for(channels, 'operator_family_signature', 'family')
    for c in channels:
        c['shared_parameter_count'] = sum(len(usage[parameters[i].name]) > 1 for i in c['parameter_indices'])
    by_analysis = {}
    for analysis in sorted({c['analysis'] for c in channels}):
        members = [c for c in channels if c['analysis'] == analysis]
        by_analysis[analysis] = dict(channels=len(members), root_structures=len({c['root_group'] for c in members}),
            semantic_structures=len({c['semantic_group'] for c in members}),
            operator_families=len({c['family_group'] for c in members}),
            observable_dimensions=dict(sorted(Counter(c['observable_dimensions'] for c in members).items())),
            data_rows=sum(c['data']['rows'] for c in members),
            nonzero_weight_rows=sum(c['data']['nonzero_weight_rows'] for c in members),
            sum_weights=sum(c['data']['sum_weights'] for c in members))
    channel_analysis = {c['name']: c['analysis'] for c in channels}
    parameter_report = []
    for i, par in enumerate(parameters):
        constraint_classes = Counter(nodes[n][0] for n in constraint_usage[par.name])
        parameter_report.append(dict(index=i, name=par.name, role=par.role,
            fixed_bounds=par.minimum == par.maximum,
            channels=usage[par.name], channel_count=len(usage[par.name]),
            analyses=sorted({channel_analysis[n] for n in usage[par.name]}),
            constraints=constraint_usage[par.name], constraint_classes=dict(sorted(constraint_classes.items())),
            card_types=sorted(card_types.get(par.name, ())),
            in_model_config_nuisances=par.name in raw_nuisances))
    sharing = dict(active_parameters=len(parameters), roles=dict(Counter(p.role for p in parameters)),
        fixed_bound_parameters=[p.name for p in parameters if p.minimum == p.maximum],
        channel_degree_histogram=dict(sorted(Counter(p['channel_count'] for p in parameter_report).items())),
        local_to_one_channel=sum(p['channel_count'] == 1 for p in parameter_report),
        shared_across_channels=sum(p['channel_count'] > 1 for p in parameter_report),
        shared_across_analyses=sum(len(p['analyses']) > 1 for p in parameter_report),
        constraint_only=sum(p['channel_count'] == 0 and bool(p['constraints']) for p in parameter_report),
        unconstrained_active_parameters=sum(not p['constraints'] for p in parameter_report),
        without_card_name_match=sum(not p['card_types'] for p in parameter_report),
        profiled_parameters=[dict(name=n, channels=usage[n], constraints=constraint_usage[n]) for n in sorted(profile_names)],
        most_shared=sorted(parameter_report, key=lambda p:(-p['channel_count'], p['name']))[:25],
        parameters=parameter_report)
    geometry = dict(dimensions=dict(sorted(Counter(c['observable_dimensions'] for c in channels).items())),
        data_rows=sum(c['data']['rows'] for c in channels),
        nonzero_weight_rows=sum(c['data']['nonzero_weight_rows'] for c in channels),
        zero_weight_rows=sum(c['data']['zero_weight_rows'] for c in channels),
        fractional_weight_rows=sum(c['data']['fractional_weight_rows'] for c in channels),
        sum_weights=sum(c['data']['sum_weights'] for c in channels),
        row_count_histogram=dict(sorted(Counter(c['data']['rows'] for c in channels).items())),
        dense_observed_cartesian_channels=sum(c['data']['covers_observed_cartesian_grid'] for c in channels),
        interpretation='Rows are actual weighted RooDataSet entries, often bin centres. Sum of weights is the observed event yield; declared observable bins and template bins are reported separately. No binned/unbinned assumption is inferred from RooDataSet alone.')
    formula_asts = Counter(fingerprint(n[2]['ast']) for n in nodes.values() if n[0] == 'RooFormulaVar')
    all_buffers = {}
    def buffers(value):
        if isinstance(value, np.ndarray):
            all_buffers[id(value)] = value.nbytes
        elif isinstance(value, dict):
            for v in value.values(): buffers(v)
        elif isinstance(value, (tuple, list)):
            for v in value: buffers(v)
    for _,_,attrs in nodes.values(): buffers(attrs)
    assert original['root_version'] == manifest['root_version']
    ranked_analyses = sorted(by_analysis, key=lambda a:-by_analysis[a]['data_rows'])
    dominant = ranked_analyses[:2]
    report = dict(status='census_complete', created_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        script_sha256=digest(__file__), workspace=dict(path=str(args.workspace), **model['source']),
        export=dict(path=str(args.export), sha256=export_hash, manifest=str(args.manifest),
            manifest_sha256=digest(args.manifest), combine_version=manifest['combine_version'],
            root_version=manifest['root_version'], reference=str(args.export_reference),
            reference_sha256=digest(args.export_reference)),
        runtime=dict(combine_version=version('higgs-combine-tool'), root_version=original['root_version'],
            jax_version=version('jax'), numerical_jax_calls=0),
        definition=dict(primary='Ordered RooFit direct-server class trees rooted at the factored channel PDF. Names, parameter values, bin contents and buffer dimensions are ignored; child order and arity are retained. DAG reuse is unfolded for the tree signature.',
            constraints='Shared constraints are listed once and kept out of the channel shape structure. Full original channel roots are also counted.',
            translated='Existing export input trees can differ from raw RooFit servers because translators flatten cached templates and select MultiPdf alternatives.',
            semantic='Conservative exported-tree refinement retains formula AST including numeric literals, input roles, leaf constancy/observable status and evaluation flags. Ignores numeric buffers and geometry; alias patterns are reported separately. This is not an exact-batching certificate.',
            family='Coarse density operator-class sets plus observable dimension, ignoring multiplicity, rates and formulas. Feasibility sketch only; these groups are NOT identical structures.'),
        summary=dict(channels=len(channels), root_structures=len(root_groups),
            root_singletons=sum(g['count'] == 1 for g in root_groups),
            full_channel_root_structures=len(full_groups), translated_structures=len(export_groups),
            semantic_structures=len(semantic_groups), coarse_operator_families=len(family_groups),
            root_group_size_histogram=dict(sorted(Counter(g['count'] for g in root_groups).items())),
            exported_nodes=len(nodes), exported_classes=classes, formulas=classes['RooFormulaVar'],
            distinct_formula_asts_with_literals=len(formula_asts),
            numeric_node_buffer_bytes=sum(all_buffers.values())),
        original_workspace=original, datacard=card_report, data=geometry, analyses=by_analysis,
        groups=root_groups, full_root_groups=full_groups, translated_groups=export_groups,
        semantic_groups=semantic_groups, coarse_operator_families=family_groups,
        constraints=dict(count=len(constraint_rows), by_class=dict(sorted(Counter(c['class_name'] for c in constraint_rows).items())),
                         entries=constraint_rows), sharing=sharing, channels=channels,
        feasibility=dict(
            conclusion='A tensor importer is worth prototyping, but exact whole-channel grouping alone does not produce a handful of kernels. Coarse density families identify operator batching opportunities; formula/rate lowering, exact normalization and shared-parameter gathers still need implementation and validation.',
            exact_grouping_channels_per_structure=len(channels)/len(root_groups),
            largest_two_data_analyses=dominant,
            largest_two_data_row_fraction=sum(by_analysis[a]['data_rows'] for a in dominant)/geometry['data_rows'],
            zero_weight_row_fraction=geometry['zero_weight_rows']/geometry['data_rows'],
            zero_weight_caveat='These rows need no log-density contribution but may still be needed for exact normalization and bin-statistics/profile conventions. This census does not drop them.',
            repeated_shapes_caveat='Class trees do not determine formulas, normalization modes, grids, parameter aliasing or active MultiPdf selection; all are separate metadata.',
            exact_integration_grid_status='Not established by this structural census. CACHE_HISTOGRAM_GRID / VV_FACTORIZED_GRID semantics must be matched to the configured native evaluator before precomputing analytic densities.',
            implementation_started=False, new_backend_performance_measured=False),
        validation=dict(channel_card_set_equal=True, original_data_row_count_equal=True,
            data_sum_weights_absolute_error=abs(geometry['sum_weights']-original['data_sum_weights']),
            every_group_partitions_all_channels=True, export_hash_and_workspace_source_verified=True,
            no_likelihood_or_fit_agreement_claimed=True),
        elapsed_seconds=time.perf_counter()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report['summary'].items() if k != 'exported_classes'}, indent=2))
    print('DATA', json.dumps(geometry))
    print('ANALYSES', json.dumps(by_analysis))
    print('SHARING', json.dumps({k:v for k,v in sharing.items() if k not in ('parameters','most_shared','profiled_parameters')}))
    print('CONSTRAINTS', report['constraints']['by_class'])
    print('ELAPSED', report['elapsed_seconds'], 'PEAK_RSS_MIB', report['peak_rss_mib'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('inputs/workspace_MX2000.root'))
    parser.add_argument('--card', type=Path, default=Path('/eos/cms/store/group/phys_b2g/sqian/SHHVV/xinyue/HHVV/HHVV_combCard_singlet_v4_final/datacard_mass_2000.txt'))
    parser.add_argument('--export', type=Path, default=Path('build/gpu_exports/hhvv.pkl'))
    parser.add_argument('--manifest', type=Path, default=Path('build/gpu_exports/manifest_random.json'))
    parser.add_argument('--export-reference', type=Path, default=Path('reports/gpu_local_hhvv_hessian_no_fusion.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/structure_census.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing census; use a fresh output or explicitly archive it first')
    census(args)


if __name__ == '__main__':
    main()
