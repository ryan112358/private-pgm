import unittest

import jax
import jax.numpy as jnp

from mbi.domain import Domain
from mbi.factor import Factor
from mbi.clique_vector import CliqueVector
from mbi import marginal_loss, estimation
import numpy as np
from parameterized import parameterized
import itertools

np.random.seed(0)  # Avoid flaky tests

_DOMAIN = Domain(["a", "b", "c", "d"], [2, 3, 4, 5])

_CLIQUE_SETS = [
    [("a", "b"), ("b", "c"), ("c", "d")],  # tree
    [("a",), ("a", "b"), ("b", "c"), ("a", "c"), ("b", "d")],  # cyclic
    [("a", "b"), ("d", "a")],  # missing c
    [("a", "b", "c", "d")],  # full materialization
    [("d",)],  # singleton
    [("a", "b", "c"), ("c", "b", "a"), ("b", "d")],  # (permuted) duplicates
]


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


def fake_measurements(cliques):
    P = Factor.random(_DOMAIN)
    P = P / P.sum()
    measurements = []
    for cl in cliques:
        y = P.project(cl).datavector()
        measurements.append(marginal_loss.LinearMeasurement(y, cl))
    return measurements


class TestEstimation(unittest.TestCase):

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_total_estimator(self, cliques):
        measurements = fake_measurements(cliques)
        total = estimation.minimum_variance_unbiased_total(measurements)
        np.testing.assert_allclose(total, 1.0)

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _MESH_CONFIGS))
    def test_mirror_descent(self, cliques, mesh):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        model_mesh = estimation.mirror_descent(
            _DOMAIN, loss_fn, known_total=1.0, iters=250, mesh=mesh
        )
        model_no_mesh = estimation.mirror_descent(
            _DOMAIN, loss_fn, known_total=1.0, iters=250, mesh=None
        )

        self.assertTrue(model_no_mesh.potentials.allclose(model_mesh.potentials, atol=1e-5, rtol=1e-5),
                        f"Potentials mismatch for mesh={mesh}")
        self.assertTrue(model_no_mesh.marginals.allclose(model_mesh.marginals, atol=1e-5, rtol=1e-5),
                        f"Marginals mismatch for mesh={mesh}")
        self.assertEqual(model_no_mesh.total, model_mesh.total)

        for M in measurements:
            expected = M.noisy_measurement
            actual = model_mesh.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _MESH_CONFIGS))
    def test_universal_accelerated_method(self, cliques, mesh):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        model_mesh = estimation.universal_accelerated_method(
                _DOMAIN, loss_fn, known_total=1.0, iters=250, mesh=mesh
        )
        model_no_mesh = estimation.universal_accelerated_method(
                _DOMAIN, loss_fn, known_total=1.0, iters=250, mesh=None
        )

        self.assertTrue(model_no_mesh.potentials.allclose(model_mesh.potentials, atol=1e-5, rtol=1e-5),
                        f"Potentials mismatch for mesh={mesh}")
        self.assertTrue(model_no_mesh.marginals.allclose(model_mesh.marginals, atol=1e-5, rtol=1e-5),
                        f"Marginals mismatch for mesh={mesh}")
        self.assertEqual(model_no_mesh.total, model_mesh.total)

        for M in measurements:
            expected = M.noisy_measurement
            actual = model_mesh.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _MESH_CONFIGS))
    def test_mirror_descent_l1(self, cliques, mesh):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements, norm="l1")

        model_mesh = estimation.mirror_descent(
            _DOMAIN, loss_fn, known_total=1.0, iters=250, stepsize=0.01, mesh=mesh
        )
        model_no_mesh = estimation.mirror_descent(
            _DOMAIN, loss_fn, known_total=1.0, iters=250, stepsize=0.01, mesh=None
        )

        self.assertTrue(model_no_mesh.potentials.allclose(model_mesh.potentials, atol=1e-5, rtol=1e-5),
                        f"Potentials mismatch for mesh={mesh}")
        self.assertTrue(model_no_mesh.marginals.allclose(model_mesh.marginals, atol=1e-5, rtol=1e-5),
                        f"Marginals mismatch for mesh={mesh}")
        self.assertEqual(model_no_mesh.total, model_mesh.total)

        for M in measurements:
            expected = M.noisy_measurement
            actual = model_mesh.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS))
    def test_multiplicative_weights(self, cliques):
        measurements = fake_measurements(cliques)
        potentials = CliqueVector.zeros(_DOMAIN, [_DOMAIN.attributes])
        loss_fn = marginal_loss.from_linear_measurements(measurements)
        joint_distribution = estimation.mirror_descent(
            _DOMAIN, loss_fn, known_total=1.0, potentials=potentials, iters=250
        )

        for M in measurements:
            expected = M.noisy_measurement
            actual = joint_distribution.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    # Skipping test_multiplicative_weights for mesh parameterization for now,
    # as it doesn't directly call an estimator function that takes `mesh`.

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _MESH_CONFIGS))
    def test_primal_optimization(self, cliques, mesh):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        model_mesh = estimation.lbfgs(
            _DOMAIN, loss_fn, known_total=1.0, iters=250, mesh=mesh
        )
        model_no_mesh = estimation.lbfgs(
            _DOMAIN, loss_fn, known_total=1.0, iters=250, mesh=None
        )

        self.assertTrue(model_no_mesh.potentials.allclose(model_mesh.potentials, atol=1e-5, rtol=1e-5),
                        f"Potentials mismatch for mesh={mesh}")
        self.assertTrue(model_no_mesh.marginals.allclose(model_mesh.marginals, atol=1e-5, rtol=1e-5),
                        f"Marginals mismatch for mesh={mesh}")
        self.assertEqual(model_no_mesh.total, model_mesh.total)

        for M in measurements:
            expected = M.noisy_measurement
            actual = model_mesh.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _MESH_CONFIGS))
    def test_dual_averaging(self, cliques, mesh):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        L = 0.001  # Currently treating Lipschitz constant as a hyperparameter
        model_mesh = estimation.dual_averaging(
            _DOMAIN, loss_fn, lipschitz=L, known_total=1.0, iters=250, mesh=mesh
        )
        model_no_mesh = estimation.dual_averaging(
            _DOMAIN, loss_fn, lipschitz=L, known_total=1.0, iters=250, mesh=None
        )

        self.assertTrue(model_no_mesh.potentials.allclose(model_mesh.potentials, atol=1e-5, rtol=1e-5),
                        f"Potentials mismatch for mesh={mesh}")
        self.assertTrue(model_no_mesh.marginals.allclose(model_mesh.marginals, atol=1e-5, rtol=1e-5),
                        f"Marginals mismatch for mesh={mesh}")
        self.assertEqual(model_no_mesh.total, model_mesh.total)

        for M in measurements:
            expected = M.noisy_measurement
            actual = model_mesh.project(M.clique).datavector()
            np.testing.assert_allclose(actual, expected, atol=1e-2)

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _MESH_CONFIGS))
    def test_interior_gradient(self, cliques, mesh):
        measurements = fake_measurements(cliques)
        loss_fn = marginal_loss.from_linear_measurements(measurements)

        L = 0.01  # Currently treating Lipschitz constant as a hyperparameter
        model_mesh = estimation.interior_gradient(
            _DOMAIN, loss_fn, lipschitz=L, known_total=1.0, iters=250, mesh=mesh
        )
        model_no_mesh = estimation.interior_gradient(
            _DOMAIN, loss_fn, lipschitz=L, known_total=1.0, iters=250, mesh=None
        )

        self.assertTrue(model_no_mesh.potentials.allclose(model_mesh.potentials, atol=1e-5, rtol=1e-5),
                        f"Potentials mismatch for mesh={mesh}")
        self.assertTrue(model_no_mesh.marginals.allclose(model_mesh.marginals, atol=1e-5, rtol=1e-5),
                        f"Marginals mismatch for mesh={mesh}")
        self.assertEqual(model_no_mesh.total, model_mesh.total)

        for M in measurements:
            expected = M.noisy_measurement
            actual = model_mesh.project(M.clique).datavector()
            # Original test didn't have an assertion here, so just project.
            # np.testing.assert_allclose(actual, expected, atol=1e-2) # If assertion needed

    @parameterized.expand(itertools.product(_CLIQUE_SETS, _MESH_CONFIGS))
    def test_mle(self, cliques, mesh):
        P = Factor.random(_DOMAIN) * 10
        total = float(P.sum())
        mu = CliqueVector(_DOMAIN, cliques, {cl: P.project(cl) for cl in cliques})

        model_mesh = estimation.mle_from_marginals(mu, known_total=total, mesh=mesh)
        model_no_mesh = estimation.mle_from_marginals(mu, known_total=total, mesh=None)

        self.assertTrue(model_no_mesh.potentials.allclose(model_mesh.potentials, atol=1e-5, rtol=1e-5),
                        f"Potentials mismatch for mesh={mesh}")
        self.assertTrue(model_no_mesh.marginals.allclose(model_mesh.marginals, atol=1e-5, rtol=1e-5),
                        f"Marginals mismatch for mesh={mesh}")
        self.assertEqual(model_no_mesh.total, model_mesh.total)

        for cl in cliques:
            expected = mu.project(cl).datavector()
            actual = model_mesh.project(cl).datavector()
            np.testing.assert_allclose(actual, expected, atol=100 / total)
