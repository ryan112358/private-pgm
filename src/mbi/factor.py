from typing import TypeAlias, Collection, Callable

import chex
import jax
import jax.numpy as jnp
import attr
import functools
from mbi import Domain
import numpy as np

jax.config.update("jax_enable_x64", True)

@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=['domain'], 
    data_fields=['values']
)
@attr.dataclass(frozen=True)
class Factor:
  """A factor over a domain."""
  domain: Domain
  values: jax.Array = attr.field(converter=jnp.array)

  def __post_init__(self):
    if self.values.shape != self.domain.shape:
      raise ValueError('values must be same shape as domain.')

  # Constructors
  @classmethod
  def zeros(cls, domain: Domain) -> 'Factor':
    return cls(domain, jnp.zeros(domain.shape))

  @classmethod
  def ones(cls, domain: Domain) -> 'Factor':
    return cls(domain, jnp.ones(domain.shape))

  @classmethod
  def random(cls, domain: Domain) -> 'Factor':
    return cls(domain, np.random.rand(*domain.shape))

  # Reshaping operations
  def transpose(self, attrs: Collection[str]) -> 'Factor':
    if set(attrs) != set(self.domain.attrs):
      raise ValueError("attrs must be same as domain attributes")
    newdom = self.domain.project(attrs)
    ax = newdom.axes(self.domain.attrs)
    values = jnp.moveaxis(self.values, range(len(ax)), ax)
    return Factor(newdom, values)

  def expand(self, domain):
    if not domain.contains(self.domain):
      raise ValueError('Expanded domain must contain domain.')
    dims = len(domain) - len(self.domain)
    values = self.values.reshape(self.domain.shape + tuple([1] * dims))
    ax = domain.axes(self.domain.attrs)
    values = jnp.moveaxis(values, range(len(ax)), ax)
    values = jnp.broadcast_to(values, domain.shape)
    return Factor(domain, values)


  # Functions that aggregate along some subset of axes
  def _aggregate(self, fn: Callable, attrs: Collection[str] | None = None) -> 'Factor':
    attrs = self.domain.attrs if attrs is None else attrs
    axes = self.domain.axes(attrs)
    values = fn(self.values, axis=axes)
    newdom = self.domain.marginalize(attrs)
    return Factor(newdom, values)

  def max(self, attrs: Collection[str] | None = None) -> 'Factor':
    return self._aggregate(jnp.max, attrs)

  def sum(self, attrs: Collection[str] | None = None) -> 'Factor':
    return self._aggregate(jnp.sum, attrs)

  def logsumexp(self, attrs: Collection[str] | None = None) -> 'Factor':
    return self._aggregate(jax.scipy.special.logsumexp, attrs)

  def project(self, attrs: str | tuple[str, ...], log: bool = False) -> 'Factor':
    if isinstance(attrs, str):
      attrs = (attrs,)
    marginalized = self.domain.marginalize(attrs).attrs
    result = self.logsumexp(marginalized) if log else self.sum(marginalized)
    return result.transpose(attrs)


  # Functions that operate element-wise
  def exp(self, out = None) -> 'Factor':
    return Factor(self.domain, jnp.exp(self.values))

  def log(self, out = None) -> 'Factor':
    return Factor(self.domain, jnp.log(self.values))

  def normalize(self, total: float = 1.0, log: bool = False) -> 'Factor':
    if log:
      return self + jnp.log(total) - self.logsumexp()
    return self * total / self.sum()

  def copy(self) -> 'Factor':
    return self

  def __float__(self):
    if len(self.domain) > 0:
      raise ValueError('Domain must be empty to convert to float.')
    return float(self.values)

  # Binary operations between two factors
  def _binaryop(self, fn: Callable, other: 'Factor' | chex.Numeric) -> 'Factor':
    if isinstance(other, chex.Numeric) and jnp.ndim(other) == 0:
      other = Factor(Domain([], []), other)
    newdom = self.domain.merge(other.domain)
    factor1 = self.expand(newdom)
    factor2 = other.expand(newdom)
    return Factor(newdom, fn(factor1.values, factor2.values))

  def __sub__(self, other: 'Factor' | chex.Numeric) -> 'Factor':
    return self._binaryop(jnp.subtract, other)

  def __truediv__(self, other: 'Factor' | chex.Numeric) -> 'Factor':
    return self._binaryop(jnp.divide, other)

  def __mul__(self, other: 'Factor' | chex.Numeric) -> 'Factor':
    """Multiply two factors together.

    Example Usage:
    >>> f1 = Factor.ones(Domain(['a','b'], [2,3]))
    >>> f2 = Factor.ones(Domain(['b','c'], [3,4]))
    >>> f3 = f1 * f2
    >>> print(f3.domain)
    Domain(a: 2, b: 3, c: 4)

    Args:
      other: the other factor to multiply

    Returns:
      the product of the two factors
    """
    return self._binaryop(jnp.multiply, other)

  def __add__(self, other: 'Factor' | chex.Numeric) -> 'Factor':
    return self._binaryop(jnp.add, other)

  def __radd__(self, other: chex.Numeric) -> 'Factor':
    return self + other

  def __rsub__(self, other: chex.Numeric) -> 'Factor':
    return self + (-1*other)

  def __rmul__(self, other: chex.Numeric) -> 'Factor':
    return self * other

  def dot(self, other: 'Factor') -> 'Factor':
    if self.domain != other.domain:
        raise ValueError(f'Domains do not match {self.domain} != {other.domain}')
    return jnp.sum(self.values * other.values)

  def datavector(self, flatten: bool=True) -> jax.Array:
    return self.values.flatten() if flatten else self.values


