import unittest

import jax
import jax.numpy as jnp

from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi import marginal_oracles
import numpy as np
from parameterized import parameterized
import itertools


def _variable_elimination_oracle(
    potentials: CliqueVector, total: float = 1, mesh: jax.sharding.Mesh | None = None
) -> CliqueVector:
    domain, cliques = potentials.domain, potentials.cliques
    mu = {
        cl: marginal_oracles.variable_elimination(potentials, cl, total, mesh=mesh)
        for cl in cliques
    }
    return CliqueVector(domain, cliques, mu)

def _calculate_many_oracle(potentials: CliqueVector, total: float = 1, mesh: jax.sharding.Mesh | None = None):
    return marginal_oracles.calculate_many_marginals(
        potentials, potentials.cliques, total, mesh=mesh
    )


_ORACLES = [
    marginal_oracles.brute_force_marginals,
    marginal_oracles.einsum_marginals,
    marginal_oracles.message_passing_stable,
    marginal_oracles.message_passing_fast,
    _variable_elimination_oracle,
    _calculate_many_oracle
]

_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])

_CLIQUE_SETS = [
    [("a", "b"), ("b", "c"), ("c", "d")],  # tree
    [("a",), ("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")],  # cyclic
    [("a", "b"), ("d", "a")],  # missing c
    [("a", "b", "c", "d")],  # full materialization
    [("d",)],  # singleton
    [("a", "b", "c"), ("c", "b", "a"), ("b", "d")],  # (permuted) duplicates
    # [],  empty is currently not supported
]

_ALL_CLIQUES = itertools.chain.from_iterable(
    itertools.combinations(_DOMAIN.attrs, r) for r in range(5)
)


# Define mesh configurations for testing
_MESH = None
# Only create mesh if multiple devices are available for actual sharding
# This avoids potential issues on single-device setups while still testing mesh path
if jax.device_count() > 1:
    _MESH = jax.sharding.Mesh(jax.devices(), ('dp',))

# Create the list of configurations to test (None and the mesh if created)
# Ensures _MESH_CONFIGS is [None] if device_count <= 1
# and [None, Mesh(...)] if device_count > 1
_MESH_CONFIGS = [config for config in [None, _MESH] if config is not _MESH or jax.device_count() > 1]


class TestMarginalOracles(unittest.TestCase):

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS, _MESH_CONFIGS))
    def test_shapes(self, oracle, cliques, mesh):
        zeros = CliqueVector.zeros(_DOMAIN, cliques)
        marginals = oracle(zeros, mesh=mesh)
        self.assertEqual(marginals.domain, _DOMAIN)
        self.assertEqual(marginals.cliques, cliques)
        self.assertEqual(set(zeros.arrays.keys()), set(marginals.arrays.keys()))
        for cl in cliques:
            self.assertEqual(marginals[cl].domain.attrs, cl)

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS, [1, 100], _MESH_CONFIGS))
    def test_uniform(self, oracle, cliques, total, mesh):
        zeros = CliqueVector.zeros(_DOMAIN, cliques)
        marginals_no_mesh = oracle(zeros, total, mesh=None)
        marginals_mesh = oracle(zeros, total, mesh=mesh)
        for cl in cliques:
            m1 = marginals_no_mesh[cl]
            m2 = marginals_mesh[cl]
            np.testing.assert_allclose(
                m1.values,
                m2.values,
                atol=1e-6,
                rtol=1e-6,
                err_msg=f"Mismatch for clique {cl} with mesh={mesh}",
            )
            expected = total / _DOMAIN.size(cl)
            np.testing.assert_allclose(marginals_no_mesh[cl].datavector(), expected)

    @parameterized.expand(itertools.product(_ORACLES, _CLIQUE_SETS, _MESH_CONFIGS))
    def test_matches_brute_force(self, oracle, cliques, mesh, total=10):
        theta = CliqueVector.random(_DOMAIN, cliques)
        mu1_mesh = oracle(theta, total, mesh=mesh)
        mu1_no_mesh = oracle(theta, total, mesh=None)
        mu2_bf = marginal_oracles.brute_force_marginals(theta, total, mesh=None)
        for cl in cliques:
            np.testing.assert_allclose(
                mu1_no_mesh[cl].values,
                mu1_mesh[cl].values,
                atol=1e-6,
                rtol=1e-6,
                err_msg=f"Mismatch for clique {cl} between mesh=None and mesh={mesh}",
            )
            np.testing.assert_allclose(
                mu1_mesh[cl].datavector(), mu2_bf[cl].datavector()
            )

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _ALL_CLIQUES, _MESH_CONFIGS))
    def test_variable_elimination(self, model_cliques, query_clique, mesh):
        theta = CliqueVector.random(_DOMAIN, model_cliques)
        ans_mesh = marginal_oracles.variable_elimination(
            theta, query_clique, mesh=mesh
        )
        ans_no_mesh = marginal_oracles.variable_elimination(
            theta, query_clique, mesh=None
        )
        np.testing.assert_allclose(
            ans_no_mesh.values,
            ans_mesh.values,
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"Mismatch for VE query {query_clique} with mesh={mesh}",
        )
        self.assertEqual(ans_mesh.domain.attributes, query_clique)
