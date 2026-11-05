"""Framework-neutral normalization geometry of Combine's simple product."""


def simple_product_axes(inputs, settings, axes, depends):
    """Native createIntegral mutates each conditional set in factor order."""
    selected = [{a for a in axes if depends(name, a)} for name in inputs]
    for i, flag in enumerate(settings):
        if flag == 1:
            for j in range(len(inputs)):
                if i != j:
                    selected[i].difference_update(selected[j])
    return [tuple(sorted(s)) for s in selected]
