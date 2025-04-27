"""Defines loss functions based on linear measurements of marginals.

This module provides structures and functions for defining and calculating loss
based on potentially noisy linear measurements of marginal distributions. Key
components include the `LinearMeasurement` class to represent individual
measurements and the `MarginalLossFn` class to define loss functions over
`CliqueVector` objects, enabling the evaluation of model fit against observed
or noisy data. Utilities for clique manipulation and feasibility checks are also
included.
"""
import functools
from typing import Any, Callable, Mapping, Protocol, TypeAlias

import attr
import chex
import jax
import jax.numpy as jnp

from .clique_utils import Clique
from .clique_utils import clique_mapping
from .clique_utils import maximal_subset
from .clique_vector import CliqueVector
from .factor import Factor


def identity_fn(x: jax.Array) -> jax.Array:
    """Simple identity function: returns the input `x` unchanged."""
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
    """Calculates the average L1 distance between overlapping marginals in `mu` (consistency)."""
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
