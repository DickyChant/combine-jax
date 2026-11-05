"""Sparse, framework-neutral storage of exact CMSHistFunc morph coefficients.

Each active (histogram, nuisance, source bin) is stored once, including zero
coefficients. Only padding is removed. Nuisance and bin order are retained;
source-bin projections and tails keep their original float64 values.
"""
import numpy as np


def pack_sparse_group(group, output_bins):
    count = len(group)
    width = max(h['source_active'] for h in group)
    fields = dict(base=np.zeros((count, width)), region=np.ones(count),
        source_mask=np.zeros((count, width), dtype=bool), nominal_sum=np.zeros(count),
        projection=np.zeros((count, output_bins, width)), tail=np.zeros((count, output_bins)))
    inputs, histograms, morphs, bins, diffs, sums = [], [], [], [], [], []
    first = 0
    for i, h in enumerate(group):
        f, m, b = h['source_active'], len(h['indices']), h['output_size']
        fields['base'][i, :f] = h['base']
        fields['source_mask'][i, :f] = True
        fields['region'][i] = h['region']
        fields['nominal_sum'][i] = h['nominal_sum']
        fields['projection'][i, :b, :f] = h['projection']
        fields['tail'][i, :b] = h['tail']
        inputs.extend(h['indices'])
        histograms.extend([i]*m)
        # Sorted by destination bin, then original nuisance position. Repeated
        # scalar bindings remain separate terms, exactly as in the dense form.
        morphs.append(np.tile(np.arange(first, first+m, dtype=np.int32), f))
        bins.append(np.repeat(i*width+np.arange(f, dtype=np.int32), m))
        diffs.append(h['diff'].T.reshape(-1))
        sums.append(h['sum'].T.reshape(-1))
        first += m
    fields.update(morph_histograms=np.asarray(histograms, dtype=np.int32),
        entry_morphs=np.concatenate(morphs), entry_bins=np.concatenate(bins),
        diff=np.concatenate(diffs), sum=np.concatenate(sums))
    return np.asarray(inputs, dtype=np.int32), fields


def numpy_segment_sum(values, indices, size):
    result = np.zeros(size, dtype=np.float64)
    np.add.at(result, indices, values)
    return result


def validate_sparse_indices(buffers, histogram_count, morph_count):
    """Check exact active-cell coverage without reconstructing dense padding."""
    histograms = buffers['morph_histograms']
    morphs, bins = buffers['entry_morphs'], buffers['entry_bins']
    width = buffers['base'].shape[1]
    if (histograms.shape != (morph_count,) or histograms.dtype != np.int32
            or morphs.ndim != 1 or bins.shape != morphs.shape
            or morphs.dtype != np.int32 or bins.dtype != np.int32
            or np.any(histograms < 0) or np.any(histograms >= histogram_count)
            or np.any(morphs < 0) or np.any(morphs >= morph_count)
            or np.any(bins < 0) or np.any(bins >= histogram_count*width)):
        raise ValueError('Invalid sparse morph indices')
    if (np.any(histograms[morphs] != bins//width)
            or np.any(~buffers['source_mask'].reshape(-1)[bins])
            or np.any((bins[1:] < bins[:-1])
                      | ((bins[1:] == bins[:-1]) & (morphs[1:] <= morphs[:-1])))):
        raise ValueError('Sparse morph entries must be unique, ordered and in their histogram')
    expected = np.sum(buffers['source_mask'], axis=1)[histograms]
    if not np.array_equal(np.bincount(morphs, minlength=morph_count), expected):
        raise ValueError('Sparse morph source-bin coverage differs')
