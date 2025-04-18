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

    A factor maps assignments of variables in its domain to non-negative real numbers.
    It is often used to represent marginal distributions, conditional probability
    tables, or intermediate results in probabilistic graphical models.

    The factor is associated with a `Domain` object that specifies the variables
    and their cardinalities. The actual numerical values are stored in a JAX array.

    Attributes:
        domain: A `Domain` object specifying the variables (attributes) and their
            shapes (cardinalities) covered by this factor.
        values: A `jax.Array` containing the numerical values of the factor.
            The shape of this array must match the shape specified by the `domain`.
    """

    domain: Domain
    values: jax.Array = attr.field(converter=_try_convert)

    def __post_init__(self):
        """Validates the Factor object after initialization.

        Ensures that the shape of the `values` array matches the shape defined
        by the `domain`.

        Raises:
            ValueError: If the shape of `values` does not match `domain.shape`.
        """
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
        """Rearranges the axes (attributes) of the factor according to a new order.

        This operation changes the order of attributes in the factor's domain and
        correspondingly permutes the axes of the underlying `values` array. It does
        not change the numerical values themselves, only their arrangement.

        Args:
            attrs: A sequence of attribute names (str) representing the desired
                new order of attributes. Must contain the same attributes as the
                factor's current domain.

        Returns:
            A new Factor object with the transposed domain and values.

        Raises:
            ValueError: If `attrs` does not contain exactly the same set of
                attributes as the factor's domain.
        """
        if set(attrs) != set(self.domain.attrs):
            raise ValueError("attrs must be same as domain attributes")
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        values = jnp.moveaxis(self.values, range(len(ax)), ax)
        return Factor(newdom, values)

    def expand(self, domain: Domain) -> "Factor":
        """Expands the factor's domain to include additional attributes.

        The factor's values are broadcasted to the new, larger domain. The original
        values are repeated along the new dimensions introduced by the expanded domain.
        This is useful for aligning factors before binary operations like multiplication
        or addition.

        Args:
            domain: The target `Domain` object to expand to. This domain must
                contain all attributes of the factor's current domain.

        Returns:
            A new Factor object defined over the expanded `domain`.

        Raises:
            ValueError: If the target `domain` does not contain all attributes
                of the factor's current domain.
        """
        # Note: Added type hint for domain argument
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
        """Projects the factor onto a subset of attributes by marginalization.

        This operation aggregates the factor's values over the attributes *not*
        included in the `attrs` argument. By default, it performs summation.
        If `log=True`, it performs log-sum-exp aggregation, suitable for factors
        representing log-probabilities. The resulting factor's domain contains
        only the specified `attrs`, in their canonical order from the original domain.

        Args:
            attrs: A single attribute name (str) or a tuple of attribute names
                to project onto.
            log: If True, performs aggregation using log-sum-exp instead of sum.
                Defaults to False.

        Returns:
            A new Factor object representing the marginal distribution (or
            log-marginal) over the specified attributes.
        """
        if isinstance(attrs, str):
            attrs = (attrs,)
        marginalized = self.domain.marginalize(attrs).attrs
        result = self.logsumexp(marginalized) if log else self.sum(marginalized)
        return result.transpose(attrs)

    # Functions that operate element-wise
    def exp(self, out=None) -> "Factor":
        # TODO(docstring): Add docstring for exp
        return Factor(self.domain, jnp.exp(self.values))

    def log(self, out=None) -> "Factor":
        # TODO(docstring): Add docstring for log
        return Factor(self.domain, jnp.log(self.values))

    def normalize(self, total: float = 1.0, log: bool = False) -> "Factor":
        """Normalizes the factor so that its values sum to a specified total.

        Scales the factor's values proportionally so that their sum equals `total`.
        If `log=True`, it assumes the factor represents log-probabilities and performs
        normalization in log-space (i.e., adds a constant such that the
        log-sum-exp of the values equals `log(total)`).

        Args:
            total: The target sum for the normalized factor's values. Defaults to 1.0.
            log: If True, performs normalization in log-space. Defaults to False.

        Returns:
            A new Factor object with normalized values.
        """
        if log:
            return self + jnp.log(total) - self.logsumexp()
        return self * total / self.sum()

    def copy(self) -> "Factor":
        # TODO(docstring): Add docstring for copy
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
        """Multiplies this factor by another factor or a scalar.

        If multiplying by another Factor (`other`), the domains are merged. Both factors
        are expanded to this merged domain, and then their values are multiplied
        element-wise.
        If multiplying by a scalar (`other`), the scalar is broadcast and multiplied
        with each element of this factor's `values`.

        Example Usage:
            >>> domain1 = Domain(('a', 'b'), (2, 3))
            >>> domain2 = Domain(('b', 'c'), (3, 4))
            >>> f1 = Factor(domain1, jnp.ones((2, 3)))
            >>> f2 = Factor(domain2, jnp.ones((3, 4)) * 2)
            >>> f3 = f1 * f2
            >>> print(f3.domain)
            Domain(a: 2, b: 3, c: 4)
            >>> print(f3.values.shape)
            (2, 3, 4)
            >>> print(f3.values[0, 0, 0])
            2.0
            >>> f4 = f1 * 3.0
            >>> print(f4.domain)
            Domain(a: 2, b: 3)
            >>> print(f4.values[0, 0])
            3.0

        Args:
            other: The Factor or scalar (chex.Numeric) to multiply by.

        Returns:
            A new Factor representing the element-wise product. The domain of the
            new factor is the merge of the input domains if `other` is a Factor,
            or the same as this factor's domain if `other` is a scalar.
        """
        return self._binaryop(jnp.multiply, other)

    def __add__(self, other: "Factor" | chex.Numeric) -> "Factor":
        """Adds this factor to another factor or a scalar.

        If adding another Factor (`other`), the domains are merged. Both factors
        are expanded to this merged domain, and then their values are added
        element-wise.
        If adding a scalar (`other`), the scalar is broadcast and added
        to each element of this factor's `values`.

        Args:
            other: The Factor or scalar (chex.Numeric) to add.

        Returns:
            A new Factor representing the element-wise sum. The domain of the
            new factor is the merge of the input domains if `other` is a Factor,
            or the same as this factor's domain if `other` is a scalar.
        """
        return self._binaryop(jnp.add, other)

    def __radd__(self, other: chex.Numeric) -> "Factor":
        # TODO(docstring): Add docstring for radd
        return self + other

    def __rsub__(self, other: chex.Numeric) -> "Factor":
        # TODO(docstring): Add docstring for rsub
        return self + (-1 * other)

    def __rmul__(self, other: chex.Numeric) -> "Factor":
        # TODO(docstring): Add docstring for rmul
        return self * other

    def dot(self, other: "Factor") -> float:
        # TODO(docstring): Add docstring for dot
        # Note: Changed return type hint to float based on implementation
        if self.domain != other.domain:
            raise ValueError(f"Domains do not match {self.domain} != {other.domain}")
        return jnp.sum(self.values * other.values)

    def datavector(self, flatten: bool = True) -> jax.Array:
        # TODO(docstring): Add docstring for datavector
        return self.values.flatten() if flatten else self.values
