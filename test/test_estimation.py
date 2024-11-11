import unittest
from mbi import Domain, Factor, CliqueVector
from mbi import marginal_loss, estimation
import numpy as np
from parameterized import parameterized
import itertools

_DOMAIN = Domain(['a', 'b', 'c', 'd'], [2, 3, 4, 5])

_CLIQUE_SETS = [
  [('a', 'b'), ('b', 'c'), ('c', 'd')],  # tree
  [('a',), ('a', 'b'), ('b', 'c'), ('a', 'c'), ('b', 'd')], # cyclic
  [('a','b'), ('d','a')], # missing c
  [('a', 'b', 'c', 'd')], # full materialization
  [('d',)], # singleton
  [('a', 'b', 'c'), ('c', 'b', 'a'), ('b', 'd')] # (permuted) duplicates
]

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

  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_mirror_descent(self, cliques):
    measurements = fake_measurements(cliques)
    loss_fn = marginal_loss.from_linear_measurements(measurements)

    model = estimation.mirror_descent(_DOMAIN, loss_fn, known_total=1.0, iters=500)
    for M in measurements:
      expected = M.noisy_measurement
      actual = model.project(M.clique).datavector() 
      np.testing.assert_allclose(actual, expected, atol=1e-2)


  @parameterized.expand(itertools.product(_CLIQUE_SETS))
  def test_multiplicative_weights(self, cliques):
    measurements = fake_measurements(cliques)
    potentials = CliqueVector.zeros(_DOMAIN, [_DOMAIN.attributes])
    loss_fn = marginal_loss.from_linear_measurements(measurements)
    joint_distribution = estimation.mirror_descent(_DOMAIN, loss_fn, known_total=1.0, potentials=potentials)

    for M in measurements:
      expected = M.noisy_measurement
      actual = joint_distribution.project(M.clique).datavector() 
      np.testing.assert_allclose(actual, expected, atol=1e-2)



