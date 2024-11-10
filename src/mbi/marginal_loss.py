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
  {('B', 'A'): ('A', 'B'), ('B',): ('A', 'B'), ('C',): ('A', 'B'), ('B', 'C'): ('B', 'C')}
    
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


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=['clique', 'stddev', 'query', 'loss'],
    data_fields=['values']
)
@attr.dataclass(frozen=True)
class LinearMeasurement:
  """A class for representing a private linear measurement of a marginal."""
  noisy_measurement: jax.Array = attr.field(converter=jnp.array)
  clique: Clique = attr.field(converter=tuple)
  stddev: float = 1.0
  query: Callable[[jax.Array], jax.Array] = lambda x: x
  loss: str = 'l2'

  def loss_fn(self, mu: jax.Array) -> jax.Array:
    diff = self.query(mu) - self.noisy_measurement
    if self.loss == 'l2':
      return (diff @ diff) / (2 * self.stddev)
    elif self.loss == 'l1':
      return jnp.sum(jnp.abs(diff)) / self.stddev
    else:
      raise ValueError('Unknown loss function.')


@attr.dataclass(frozen=True)
class MarginalLossFn:
  """A Loss function over the concatenated vector of marginals."""
  cliques: list[Clique]
  loss_fn: Callable[[CliqueVector], chex.Numeric]

  def __call__(self, marginals: CliqueVector) -> chex.Numeric:
    return self.loss_fn(marginals)

def from_linear_measurements(measurements: list[LinearMeasurement]) -> 'MarginalLossFn':
  cliques = [m.clique for m in measurements]
  maximal_cliques = maximal_subset(cliques)
  mapping = clique_mapping(maximal_cliques, cliques)

  def loss_fn(marginals: CliqueVector) -> chex.Numeric:
    loss = 0.0
    for measurement in measurements:
      cl = measurement.clique
      mu = marginals[mapping[cl]].project(cl).datavector(flatten=True)
      loss += measurement.loss_fn(mu)
    return loss

  return MarginalLossFn(maximal_cliques, loss_fn)

