import numpy as np
from mbi import Domain, Factor
import jax
import jax.numpy as jnp
import functools
import attr
from typing import TypeAlias
import operator
import chex

Clique: TypeAlias = tuple[str, ...]

@functools.partial(
  jax.tree_util.register_dataclass,
  meta_fields=['domain', 'cliques'],
  data_fields=['arrays']
)
@attr.dataclass(frozen=True)
class CliqueVector:
  """ This is a convenience class for simplifying arithmetic over the 
    concatenated vector of marginals and potentials.

    These vectors are represented as a dictionary mapping cliques (subsets of attributes)
    to marginals/potentials (Factor objects)
  """
  domain: Domain
  cliques: list[Clique]
  arrays: dict[Clique, Factor]

  def __post_init__(self):
    if set(cliques) != set(arrays):
      raise ValueError('Cliques must be equal to keys of array.')
    if len(cliques) != len(set(cliques)):
      raise ValueError('Cliques must be unique.')

  @classmethod
  def zeros(cls, domain: Domain, cliques: list[Clique]) -> 'CliqueVector':
    arrays = {cl: Factor.zeros(domain.project(cl)) for cl in cliques}
    return cls(domain, cliques, arrays)

  @classmethod
  def ones(cls, domain: Domain, cliques: list[Clique]) -> 'CliqueVector':
    arrays = {cl: Factor.ones(domain.project(cl)) for cl in cliques}
    return cls(domain, cliques, arrays)

  @classmethod
  def uniform(cls, domain: Domain, cliques: list[Clique]) -> 'CliqueVector':
    arrays = {cl: Factor.uniform(domain.project(cl)) for cl in cliques}
    return cls(domain, cliques, arrays)

  @classmethod
  def random(cls, domain: Domain, cliques: list[Clique]):
    arrays = {cl: Factor.random(domain.project(cl)) for cl in cliques}
    return cls(domain, cliques, arrays)

  @classmethod
  def from_projectable(cls, data, cliques: list[Clique]):
    arrays = { data.project(cl) for cl in cliques }
    return cls(data.domain, cliques, arrays)

  @functools.cached_property
  def active_domain(self):
    domains = [self.domain.project(cl) for cl in self.cliques]
    return functools.reduce(Domain.merge, domains)

  # @functools.lru_cache(maxsize=None)
  def parent(self, clique: Clique) -> Clique | None:
    for result in self.cliques:
      if set(clique) <= set(result):
        return result

  def supports(self, clique: Clique) -> bool:
    return self.parent(clique) is not None

  def project(self, clique: Clique) -> Factor:
    if self.supports(clique):
      return self[self.parent(clique)].project(clique)
    raise ValueError(f'Cannot project onto unsupported {clique}.')

  def combine(self, other: 'CliqueVector') -> 'CliqueVector':
    # combines this CliqueVector with other, even if they do not share the same set of factors
    # used for warm-starting optimization
    # Important note: if other contains factors not defined within this CliqueVector, they
    # are ignored and *not* combined into this CliqueVector
    for cl in other:
      for cl2 in self:
        if set(cl) <= set(cl2):
          self[cl2] += other[cl]
          break

  def normalize(self, total: float=1, log: bool=True):
    is_leaf = lambda node: isinstance(node, Factor)
    return jax.tree.map(lambda f: f.normalize(total, log), self, is_leaf=is_leaf)

  def __mul__(self, const: chex.Numeric) -> 'CliqueVector':
    return jax.tree.map(lambda f: f*const, self)

  def __rmul__(self, const: chex.Numeric) -> 'CliqueVector':
    return self.__mul__(const)

  def __add__(self, other: chex.Numeric | 'CliqueVector') -> 'CliqueVector':
    is_leaf = lambda node: isinstance(node, Factor)
    if isinstance(other, CliqueVector):
      return jax.tree.map(Factor.__add__, self, other, is_leaf=is_leaf)
    return jax.tree.map(lambda f: f*other, self, is_leaf=is_leaf)

  def __sub__(self, other: chex.Numeric | 'CliqueVector') -> 'CliqueVector':
    return self + -1 * other

  def exp(self) -> 'CliqueVector':
    return jax.tree.map(jnp.exp, self)

  def log(self) -> 'CliqueVector':
    return jax.tree.map(jnp.log, self)

  def dot(self, other: 'CliqueVector') -> chex.Numeric:
    dots = jax.tree.map(Factor.dot, self, other)
    return jax.tree.reduce(operator.add, dots)

  def size(self):
    return sum(self.domain.size(cl) for cl in self.cliques)

  def __getitem__(self, clique: Clique) -> Factor:
    return self.arrays[clique]
