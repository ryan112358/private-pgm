from collections.abc import Sequence
import attr
import chex
import numpy as np
import pandas as pd

from . import junction_tree, marginal_oracles
from .clique_vector import CliqueVector
from .dataset import Dataset
from .factor import Factor


@attr.dataclass(frozen=True)
class MarkovRandomField:
    """Represents a learned graphical model, storing potentials, marginals, and the total count."""
    potentials: CliqueVector
    marginals: CliqueVector
    total: chex.Numeric = 1

    def project(self, attrs: tuple[str, ...]) -> Factor:
        if self.marginals.supports(attrs):
            return self.marginals.project(attrs)
        return marginal_oracles.variable_elimination(
            self.potentials, attrs, self.total
        )

    def supports(self, attrs: str | Sequence[str]) -> bool:
        return self.marginals.domain.supports(attrs)

    def synthetic_data(self, rows: int | None = None, method: str = "round"):
        """Generates synthetic data based on the learned model's marginals."""
        total = max(1, int(rows or self.total))
        domain = self.domain
        cols = domain.attrs
        data = np.zeros((total, len(cols)), dtype=int)
        df = pd.DataFrame(data, columns=cols)
        cliques = [set(cl) for cl in self.cliques]
        jtree, elimination_order = junction_tree.make_junction_tree(domain, cliques)

        def synthetic_col(counts, total):
            """Generates a synthetic column by sampling or rounding based on counts and total."""
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
        marg = self.project((col,)).datavector(flatten=False)
        df.loc[:, col] = synthetic_col(marg, total)
        used = {col}

        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set().union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            # Will this work without having the maximal cliques of the junction tree?
            marg = self.project(proj + (col,)).datavector(flatten=False)

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


    @property
    def domain(self):
        """Returns the Domain object associated with this graphical model."""
        return self.potentials.domain

    @property
    def cliques(self):
        """Returns the list of cliques the model's potentials are defined over."""
        return self.potentials.cliques
