from mbi import CliqueVector, Dataset
from mbi import junction_tree
import pandas as pd
import numpy as np


def from_marginals(
    marginals: CliqueVector, rows: int, method: str = "round"
) -> Dataset:
    """Generate synthetic tabular data from the distribution.
    Valid options for method are 'round' and 'sample'."""
    total = max(1, int(rows))
    domain = marginals.domain  # Maybe should pass this in (?)
    cols = domain.attrs
    data = np.zeros((total, len(cols)), dtype=int)
    df = pd.DataFrame(data, columns=cols)
    cliques = [set(cl) for cl in marginals.cliques]
    jtree, elimination_order = junction_tree.make_junction_tree(domain, cliques)

    def synthetic_col(counts, total):
        if method == "sample":
            probas = counts / counts.sum()
            return np.random.choice(counts.size, total, True, probas)
        counts *= total / counts.sum()
        frac, integ = np.modf(counts)
        integ = integ.astype(int)
        extra = total - integ.sum()
        if extra > 0:
            idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
            integ[idx] += 1
        vals = np.repeat(np.arange(counts.size), integ)
        np.random.shuffle(vals)
        return vals

    order = elimination_order[::-1]
    col = order[0]
    marg = marginals.project([col]).datavector(flatten=False)
    df.loc[:, col] = synthetic_col(marg, total)
    used = {col}

    for col in order[1:]:
        relevant = [cl for cl in cliques if col in cl]
        relevant = used.intersection(set.union(*relevant))
        proj = tuple(relevant)
        used.add(col)
        # Will this work without having the maximal cliques of the junction tree?
        marg = marginals.project(proj + (col,)).datavector(flatten=False)

        def foo(group):
            idx = group.name
            vals = synthetic_col(marg[idx], group.shape[0])
            group[col] = vals
            return group

        if len(proj) >= 1:
            df = df.groupby(list(proj), group_keys=False).apply(foo)
        else:
            df[col] = synthetic_col(marg, df.shape[0])

    return Dataset(df, domain)
