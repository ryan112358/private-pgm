from typing import TypeAlias, Sequence, Callable

import chex
import jax
import jax.numpy as jnp
import attr
import functools
from mbi import Domain
import numpy as np

jax.config.update("jax_enable_x64", True)

def _try_convert(values):
    try:
        return jnp.array(values)
    except:
        return values  # useful if values is a Jax tracer object

@functools.partial(
    jax.tree_util.register_dataclass, meta_fields=["domain"], data_fields=["values"]
)
@attr.dataclass(frozen=True)
class Factor:
    """Represents a factor (or potential function) over a discrete domain.

    A factor maps each possible configuration of variables in its domain to a
    non-negative value. It is essentially a multi-dimensional array where the
    dimensions are defined by a `Domain` object. Factors are fundamental
    components in probabilistic graphical models and are used extensively
    within this library for representing marginal distributions or intermediate
    results during inference.

    This implementation uses JAX arrays (`jax.Array`) for the underlying numerical
    storage, leveraging JAX's capabilities for performance and automatic
    differentiation.

    Attributes:
        domain: A `Domain` object specifying the variables included in the factor
                and their respective cardinalities (number of states). This defines
                the scope and shape of the factor.
        values: A `jax.Array` containing the numerical values of the factor.
                The shape of this array must strictly match the `shape` attribute
                of the `domain` object (e.g., if domain is Domain(['A', 'B'], (2, 3)),
                values must have shape (2, 3)).

    Relationship with Domain:
        A `Factor` is always defined with respect to a `Domain`. The `Domain`
        provides the semantic meaning to the dimensions of the `values` array.
        Operations between factors often involve merging or aligning their domains.

    Key Operations:
        - Creation: `Factor.zeros(domain)`, `Factor.ones(domain)`, `Factor.random(domain)`
          allow easy instantiation of factors with specific initial values.
        - Reshaping: `transpose(attrs)` reorders the dimensions according to the
          provided attribute order. `expand(domain)` broadcasts the factor to a
          larger domain.
        - Aggregation/Marginalization: `sum(attrs)`, `logsumexp(attrs)`, `max(attrs)`
          aggregate the factor's values along specified dimensions (attributes).
          `project(attrs)` marginalizes the factor down to a subset of attributes.
        - Element-wise: `exp()`, `log()`, `normalize()` apply element-wise
          mathematical functions. `normalize()` scales the factor values to sum
          to a specific total (default is 1.0).
        - Binary Ops: Standard arithmetic operators (`+`, `-`, `*`, `/`) are overloaded
          for operations between two factors or between a factor and a scalar.
          These operations automatically handle domain merging and expansion
          to ensure compatibility.

    Example Usage:
        >>> # Create domains
        >>> domain_ab = Domain.fromdict({'A': 2, 'B': 3})
        >>> domain_bc = Domain.fromdict({'B': 3, 'C': 4})

        >>> # Create factors
        >>> factor_a = Factor.ones(domain_ab)
        >>> print(factor_a.domain)
        Domain(A: 2, B: 3)
        >>> print(factor_a.values.shape)
        (2, 3)

        >>> factor_b = Factor.random(domain_bc)
        >>> print(factor_b.domain)
        Domain(B: 3, C: 4)

        >>> # Factor multiplication automatically handles domain merging and expansion
        >>> factor_c = factor_a * factor_b
        >>> print(factor_c.domain)
        Domain(A: 2, B: 3, C: 4)
        >>> print(factor_c.values.shape)
        (2, 3, 4)

        >>> # Marginalization using sum (summing out 'A' and 'C')
        >>> factor_marg_b = factor_c.sum(['A', 'C'])
        >>> print(factor_marg_b.domain)
        Domain(B: 3)
        >>> print(factor_marg_b.values.shape)
        (3,)

        >>> # Projection is equivalent to marginalization
        >>> factor_proj_b = factor_c.project(['B'])
        >>> print(factor_proj_b.domain)
        Domain(B: 3)
        >>> jnp.allclose(factor_marg_b.values, factor_proj_b.values)
        True

        >>> # Element-wise operation
        >>> factor_exp = factor_a.exp()
        >>> print(factor_exp.values)
        [[2.71828183 2.71828183 2.71828183]
         [2.71828183 2.71828183 2.71828183]]

        >>> # Normalization
        >>> factor_norm = factor_a.normalize()
        >>> print(factor_norm.sum().values) # Should sum to 1.0
        1.0
    """
    domain: Domain
    values: jax.Array = attr.field(converter=_try_convert)

    def __post_init__(self):
        if self.values.shape != self.domain.shape:
            raise ValueError("values must be same shape as domain.")

    # Constructors
    @classmethod
    def zeros(cls, domain: Domain) -> "Factor":
        return cls(domain, jnp.zeros(domain.shape))

    @classmethod
    def ones(cls, domain: Domain) -> "Factor":
        return cls(domain, jnp.ones(domain.shape))

    @classmethod
    def random(cls, domain: Domain) -> "Factor":
        return cls(domain, np.random.rand(*domain.shape))

    # Reshaping operations
    def transpose(self, attrs: Sequence[str]) -> "Factor":
        if set(attrs) != set(self.domain.attrs):
            raise ValueError("attrs must be same as domain attributes")
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        values = jnp.moveaxis(self.values, range(len(ax)), ax)
        return Factor(newdom, values)

    def expand(self, domain):
        if not domain.contains(self.domain):
            raise ValueError("Expanded domain must contain domain.")
        dims = len(domain) - len(self.domain)
        values = self.values.reshape(self.domain.shape + tuple([1] * dims))
        ax = domain.axes(self.domain.attrs)
        values = jnp.moveaxis(values, range(len(ax)), ax)
        values = jnp.broadcast_to(values, domain.shape)
        return Factor(domain, values)

    # Functions that aggregate along some subset of axes
    def _aggregate(
        self, fn: Callable, attrs: Sequence[str] | None = None
    ) -> "Factor":
        attrs = self.domain.attrs if attrs is None else attrs
        axes = self.domain.axes(attrs)
        values = fn(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def max(self, attrs: Sequence[str] | None = None) -> "Factor":
        return self._aggregate(jnp.max, attrs)

    def sum(self, attrs: Sequence[str] | None = None) -> "Factor":
        return self._aggregate(jnp.sum, attrs)

    def logsumexp(self, attrs: Sequence[str] | None = None) -> "Factor":
        return self._aggregate(jax.scipy.special.logsumexp, attrs)

    def project(self, attrs: str | tuple[str, ...], log: bool = False) -> "Factor":
        if isinstance(attrs, str):
            attrs = (attrs,)
        marginalized = self.domain.marginalize(attrs).attrs
        result = self.logsumexp(marginalized) if log else self.sum(marginalized)
        return result.transpose(attrs)

    # Functions that operate element-wise
    def exp(self, out=None) -> "Factor":
        return Factor(self.domain, jnp.exp(self.values))

    def log(self, out=None) -> "Factor":
        return Factor(self.domain, jnp.log(self.values))

    def normalize(self, total: float = 1.0, log: bool = False) -> "Factor":
        if log:
            return self + jnp.log(total) - self.logsumexp()
        return self * total / self.sum()

    def copy(self) -> "Factor":
        return self

    def __float__(self):
        if len(self.domain) > 0:
            raise ValueError("Domain must be empty to convert to float.")
        return float(self.values)

    # Binary operations between two factors
    def _binaryop(self, fn: Callable, other: "Factor" | chex.Numeric) -> "Factor":
        if isinstance(other, chex.Numeric) and jnp.ndim(other) == 0:
            other = Factor(Domain([], []), other)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, fn(factor1.values, factor2.values))

    def __sub__(self, other: "Factor" | chex.Numeric) -> "Factor":
        return self._binaryop(jnp.subtract, other)

    def __truediv__(self, other: "Factor" | chex.Numeric) -> "Factor":
        return self._binaryop(jnp.divide, other)

    def __mul__(self, other: "Factor" | chex.Numeric) -> "Factor":
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

    def __add__(self, other: "Factor" | chex.Numeric) -> "Factor":
        return self._binaryop(jnp.add, other)

    def __radd__(self, other: chex.Numeric) -> "Factor":
        return self + other

    def __rsub__(self, other: chex.Numeric) -> "Factor":
        return self + (-1 * other)

    def __rmul__(self, other: chex.Numeric) -> "Factor":
        return self * other

    def dot(self, other: "Factor") -> "Factor":
        if self.domain != other.domain:
            raise ValueError(f"Domains do not match {self.domain} != {other.domain}")
        return jnp.sum(self.values * other.values)

    def datavector(self, flatten: bool = True) -> jax.Array:
        return self.values.flatten() if flatten else self.values
