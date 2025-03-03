import attr
from typing import Any, Callable, TypeAlias, Protocol, Mapping
from mbi import Factor, CliqueVector

import jax
import jax.numpy as jnp
import functools
import chex

Clique: TypeAlias = tuple[str, ...]


def maximal_subset(cliques: list[Clique]) -> list[Clique]:
    """Given a list of cliques, finds a maximal subset of non-nested cliques.

    A clique is considered nested in another if all its vertices are a subset
    of the other's vertices.

    Example Usage:
    >>> maximal_subset([('A', 'B'), ('B',), ('C',), ('B', 'A')])
    [('A', 'B'), ('C',)]

    Args:
      cliques: A list of cliques.

    Returns:
      A new list containing a maximal subset of non-nested cliques.
    """
    cliques = sorted(cliques, key=len, reverse=True)
    result = []
    for cl in cliques:
        if not any(set(cl) <= set(cl2) for cl2 in result):
            result.append(cl)
    return result


def clique_mapping(
    maximal_cliques: list[Clique], all_cliques: list[Clique]
) -> dict[Clique, Clique]:
    """Creates a mapping from cliques to their corresponding maximal clique.

    Example Usage:
    >>> maximal_cliques = [('A', 'B'), ('B', 'C')]
    >>> all_cliques = [('B', 'A'), ('B',), ('C',), ('B', 'C')]
    >>> mapping = clique_mapping(maximal_cliques, all_cliques)
    >>> print(mapping)
    {('B', 'A'): ('A', 'B'), ('B',): ('A', 'B'), ('C',): ('B', 'C'), ('B', 'C'): ('B', 'C')}

    Args:
      maximal_cliques: A list of maximal cliques.
      all_cliques: A list of all cliques.

    Returns:
      A mapping from cliques to their maximal clique.

    """
    mapping = {}
    for cl in all_cliques:
        for cl2 in maximal_cliques:
            if set(cl) <= set(cl2):
                mapping[cl] = cl2
                break
    return mapping

def identity_fn(x: jax.Array) -> jax.Array:
    return x

@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["clique", "stddev", "query"],
    data_fields=["values"],
)
@attr.dataclass(frozen=True)
class LinearMeasurement:
    """A class for representing a private linear measurement of a marginal."""

    noisy_measurement: jax.Array = attr.field(converter=jnp.array)
    clique: Clique = attr.field(converter=tuple)
    stddev: float = 1.0
    query: Callable[[jax.Array], jax.Array] = identity_fn


@attr.dataclass(frozen=True)
class MarginalLossFn:
    """A Loss function over the concatenated vector of marginals."""

    cliques: list[Clique]
    loss_fn: Callable[[CliqueVector], chex.Numeric]

    def __call__(self, marginals: CliqueVector) -> chex.Numeric:
        return self.loss_fn(marginals)


def from_linear_measurements(
    measurements: list[LinearMeasurement], norm: str = "l2", normalize: bool = False
) -> MarginalLossFn:
    """Construct a MarginalLossFn from a list of LinearMeasurements.

    Args:
        measurements: A list of LinearMeasurements.
        norm: Either "l1" or "l2".
        normalize: Flag determining if the loss function should be normalized
            by the length of linear measurements and estimated total.

    Returns:
        The MarginalLossFn L(mu) = sum_{c} || Q_c mu_c - y_c || (possibly squared or normalized).
    """
    if norm not in ["l1", "l2"]:
        raise ValueError(f"Unknown norm {norm}.")
    cliques = [m.clique for m in measurements]
    maximal_cliques = maximal_subset(cliques)

    def loss_fn(marginals: CliqueVector) -> chex.Numeric:
        loss = 0.0
        for M in measurements:
            mu = marginals.project(M.clique).datavector()
            diff = M.query(mu) - M.noisy_measurement
            if norm == "l2":
                loss += (diff @ diff) / (2 * M.stddev)
            elif norm == "l1":
                loss += jnp.sum(jnp.abs(diff)) / M.stddev

        if normalize:
            total = marginals.project([]).datavector(flatten=False)
            loss = loss / len(measurements) / total
            if norm == "l2":
                loss = jnp.sqrt(loss)
        return loss

    return MarginalLossFn(maximal_cliques, loss_fn)


def primal_feasibility(mu: CliqueVector) -> chex.Numeric:
    ans = 0
    count = 0
    for r in mu.cliques:
        for s in mu.cliques:
            if r == s:
                break
            d = tuple(set(r) & set(s))
            if len(d) > 0:
                x = mu[r].project(d).datavector()
                y = mu[s].project(d).datavector()
                denom = 0.5 * x.sum() + 0.5 * y.sum()
                err = jnp.linalg.norm(x - y, 1) / denom
                ans += err
                count += 1
    try:
        return ans / count
    except:
        return 0
