"""Defines the CliqueVector class for managing collections of factors over cliques.

This module introduces the `CliqueVector`, a data structure designed to hold
and manipulate sets of `Factor` objects, each associated with a specific clique
(a subset of attributes) within a domain. It facilitates operations common in
graphical models, such as projecting onto sub-cliques, expanding to larger cliques,
and performing arithmetic operations on these collections.
"""
import functools
import operator
from typing import TypeAlias

import attr
import chex
import jax
import jax.numpy as jnp
import numpy as np

from .clique_utils import Clique
from .clique_utils import reverse_clique_mapping
from .domain import Domain
from .factor import Factor


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["domain", "cliques"],
    data_fields=["arrays"],
)
@attr.dataclass
class CliqueVector:
    """This is a convenience class for simplifying arithmetic over the
    concatenated vector of marginals and potentials.

    These vectors are represented as a dictionary mapping cliques (subsets of attributes)
    to marginals/potentials (Factor objects)
    """

    domain: Domain
    cliques: list[Clique]
    arrays: dict[Clique, Factor]

    def __attrs_post_init__(self):
        if set(self.cliques) != set(self.arrays):
            raise ValueError("Cliques must be equal to keys of array.")
        if len(self.cliques) != len(set(self.cliques)):
            raise ValueError("Cliques must be unique.")

    @classmethod
    def zeros(cls, domain: Domain, cliques: list[Clique]) -> "CliqueVector":
        """Creates a CliqueVector initialized with zero factors for each clique."""
        cliques = [tuple(cl) for cl in cliques]
        arrays = {cl: Factor.zeros(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def ones(cls, domain: Domain, cliques: list[Clique]) -> "CliqueVector":
        """Creates a CliqueVector initialized with one factors for each clique."""
        cliques = [tuple(cl) for cl in cliques]
        arrays = {cl: Factor.ones(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def uniform(cls, domain: Domain, cliques: list[Clique]) -> "CliqueVector":
        """Creates a CliqueVector initialized with uniform factors for each clique."""
        cliques = [tuple(cl) for cl in cliques]
        arrays = {cl: Factor.uniform(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def random(cls, domain: Domain, cliques: list[Clique]):
        """Creates a CliqueVector initialized with random factors for each clique."""
        cliques = [tuple(cl) for cl in cliques]
        arrays = {cl: Factor.random(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def from_projectable(cls, data, cliques: list[Clique]):
        """Creates a CliqueVector by projecting a data source onto the specified cliques."""
        cliques = [tuple(cl) for cl in cliques]
        arrays = {cl: data.project(cl) for cl in cliques}
        return cls(data.domain, cliques, arrays)

    @functools.cached_property
    def active_domain(self):
        """Returns the merged domain encompassing all attributes across all cliques."""
        domains = [self.domain.project(cl) for cl in self.cliques]
        return functools.reduce(lambda a, b: a.merge(b), domains, Domain([], []))

    # @functools.lru_cache(maxsize=None)
    def parent(self, clique: Clique) -> Clique | None:
        """Finds a clique in this vector that is a superset of the given clique."""
        for result in self.cliques:
            if set(clique) <= set(result):
                return result

    def supports(self, clique: Clique) -> bool:
        """Checks if the given clique is supported (is a subset of any clique in the vector)."""
        return self.parent(clique) is not None

    def project(self, clique: Clique, log: bool = False) -> Factor:
        if self.supports(clique):
            return self[self.parent(clique)].project(clique, log=log)
        raise ValueError(f"Cannot project onto unsupported clique {clique}.")

    def expand(self, cliques: list[Clique]) -> "CliqueVector":
        """Re-expresses this CliqueVector over an expanded set of cliques.

        If the original CliqueVector represents the potentials of a Graphical Model,
        the given cliques support the cliques in the original CliqueVector, then
        the distribution represented by the new CliqueVector will be identical.

        Args:
            cliques: The new cliques the clique vector will be defined over.

        Returns:
            An expanded CliqueVector defined over the given set of cliques.
        """
        mapping = reverse_clique_mapping(cliques, self.cliques)
        arrays = {}
        for cl in cliques:
            dom = self.domain.project(cl)
            if len(mapping[cl]) == 0:
                arrays[cl] = Factor.zeros(dom)
            else:
                arrays[cl] = sum(self[cl2] for cl2 in mapping[cl]).expand(dom)
        return CliqueVector(self.domain, cliques, arrays)

    def contract(self, cliques: list[Clique], log: bool = False) -> "CliqueVector":
        """Computes a new CliqueVector by projecting this one onto a smaller set of cliques."""
        arrays = {cl: self.project(cl, log=log) for cl in cliques}
        return CliqueVector(self.domain, cliques, arrays)

    def normalize(self, total: float = 1, log: bool = True):
        """Normalizes each factor within the CliqueVector."""
        is_leaf = lambda node: isinstance(node, Factor)
        return jax.tree.map(lambda f: f.normalize(total, log), self, is_leaf=is_leaf)

    def __mul__(self, const: chex.Numeric) -> "CliqueVector":
        """Multiplies each factor in the vector by a constant."""
        return jax.tree.map(lambda f: f * const, self)

    def __rmul__(self, const: chex.Numeric) -> "CliqueVector":
        """Right-multiplies each factor in the vector by a constant."""
        return self.__mul__(const)

    def __truediv__(self, const: chex.Numeric) -> "CliqueVector":
        """Divides each factor in the vector by a constant."""
        return self.__mul__(1 / const)

    def __add__(self, other: chex.Numeric | "CliqueVector") -> "CliqueVector":
        """Adds another CliqueVector or a constant to this vector elementwise."""
        if isinstance(other, CliqueVector):
            return jax.tree.map(jnp.add, self, other)
        return jax.tree.map(lambda f: f + other, self)

    def __sub__(self, other: chex.Numeric | "CliqueVector") -> "CliqueVector":
        """Subtracts another CliqueVector or a constant from this vector elementwise."""
        return self + -1 * other

    def exp(self) -> "CliqueVector":
        """Applies elementwise exponentiation (jnp.exp) to each factor."""
        return jax.tree.map(jnp.exp, self)

    def log(self) -> "CliqueVector":
        """Applies elementwise logarithm (jnp.log) to each factor."""
        return jax.tree.map(jnp.log, self)

    def dot(self, other: "CliqueVector") -> chex.Numeric:
        """Computes the dot product between this CliqueVector and another."""
        is_leaf = lambda node: isinstance(node, Factor)
        dots = jax.tree.map(Factor.dot, self, other, is_leaf=is_leaf)
        return jax.tree.reduce(operator.add, dots, 0)

    def size(self):
        """Calculates the total number of parameters across all factors in the vector."""
        return sum(self.domain.size(cl) for cl in self.cliques)

    def __getitem__(self, clique: Clique) -> Factor:
        """Retrieves the factor associated with the given clique."""
        return self.arrays[clique]

    def __setitem__(self, clique: Clique, value: Factor):
        """Sets the factor for a given clique, replacing the existing one if present."""
        if clique in self.cliques:
            self.arrays[clique] = value
        else:
            raise ValueError(f"Clique {clique} not in CliqueVector.")
