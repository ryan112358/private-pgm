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


def reverse_clique_mapping(
    maximal_cliques: list[Clique], all_cliques: list[Clique]
) -> dict[Clique, list[Clique]]:
    """Creates a mapping from maximal cliques to a list of cliques they contain.

    Args:
      maximal_cliques: A list of maximal cliques.
      all_cliques: A list of all cliques.

    Returns:
      A mapping from maximal cliques to cliques they contain.
    """
    mapping = {cl: [] for cl in maximal_cliques}
    for cl in all_cliques:
        for cl2 in maximal_cliques:
            if set(cl) <= set(cl2):
                mapping[cl2].append(cl)
                break
    return mapping


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
        arrays = {cl: Factor.zeros(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def ones(cls, domain: Domain, cliques: list[Clique]) -> "CliqueVector":
        arrays = {cl: Factor.ones(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def uniform(cls, domain: Domain, cliques: list[Clique]) -> "CliqueVector":
        arrays = {cl: Factor.uniform(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def random(cls, domain: Domain, cliques: list[Clique]):
        arrays = {cl: Factor.random(domain.project(cl)) for cl in cliques}
        return cls(domain, cliques, arrays)

    @classmethod
    def from_projectable(cls, data, cliques: list[Clique]):
        arrays = {cl: data.project(cl) for cl in cliques}
        return cls(data.domain, cliques, arrays)

    @functools.cached_property
    def active_domain(self):
        domains = [self.domain.project(cl) for cl in self.cliques]
        return functools.reduce(lambda a, b: a.merge(b), domains)

    # @functools.lru_cache(maxsize=None)
    def parent(self, clique: Clique) -> Clique | None:
        for result in self.cliques:
            if set(clique) <= set(result):
                return result

    def supports(self, clique: Clique) -> bool:
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
        """Compute a supported CliqueVector from this."""
        arrays = {cl: self.project(cl, log=log) for cl in cliques}
        return CliqueVector(self.domain, cliques, arrays)

    def normalize(self, total: float = 1, log: bool = True):
        is_leaf = lambda node: isinstance(node, Factor)
        return jax.tree.map(lambda f: f.normalize(total, log), self, is_leaf=is_leaf)

    def __mul__(self, const: chex.Numeric) -> "CliqueVector":
        return jax.tree.map(lambda f: f * const, self)

    def __rmul__(self, const: chex.Numeric) -> "CliqueVector":
        return self.__mul__(const)

    def __add__(self, other: chex.Numeric | "CliqueVector") -> "CliqueVector":
        if isinstance(other, CliqueVector):
            return jax.tree.map(jnp.add, self, other)
        return jax.tree.map(lambda f: f + other, self)

    def __sub__(self, other: chex.Numeric | "CliqueVector") -> "CliqueVector":
        return self + -1 * other

    def exp(self) -> "CliqueVector":
        return jax.tree.map(jnp.exp, self)

    def log(self) -> "CliqueVector":
        return jax.tree.map(jnp.log, self)

    def dot(self, other: "CliqueVector") -> chex.Numeric:
        is_leaf = lambda node: isinstance(node, Factor)
        dots = jax.tree.map(Factor.dot, self, other, is_leaf=is_leaf)
        return jax.tree.reduce(operator.add, dots)

    def size(self):
        return sum(self.domain.size(cl) for cl in self.cliques)

    def __getitem__(self, clique: Clique) -> Factor:
        return self.arrays[clique]

    def __setitem__(self, clique: Clique, value: Factor):
        if clique in self.cliques:
            self.arrays[clique] = value
        else:
            raise ValueError(f"Clique {clique} not in CliqueVector.")
