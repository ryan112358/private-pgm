import numpy as np
from mbi import Domain, callbacks, CliqueVector, Factor, LinearMeasurement
from mbi import marginal_oracles, marginal_loss, synthetic_data
from typing import Callable
import jax
import chex
import attr
import optax

_DEFAULT_CALLBACK = lambda t, loss: print(loss) if t % 50 == 0 else None

# API may change, we'll see
@attr.dataclass(frozen=True)
class GraphicalModel:
  potentials: CliqueVector
  marginals: CliqueVector
  total: chex.Numeric = 1

  def project(self, attrs: tuple[str, ...]) -> Factor:
    try:
      return self.marginals.project(attrs)
    except:
      return marginal_oracles.variable_elimination(self.potentials, attrs, self.total)

  def synthetic_data(self, rows: int | None = None):
    return synthetic_data.from_marginals(self, rows or self.total)

  @property
  def domain(self):
    return self.potentials.domain
  
  @property
  def cliques(self):
    return self.potentials.cliques


def minimum_variance_unbiased_total(measurements: list[LinearMeasurement]) -> float:
  # find the minimum variance estimate of the total given the measurements
  estimates, variances = [], []
  for M in measurements:
    y = M.noisy_measurement
    try:
      # TODO: generalize to support any linear measurement that supports total query
      if np.allclose(M.query(y), y):  # query = Identity
        estimates.append(y.sum())
        variances.append(M.stddev**2 * y.size)
    except:
      continue
  estimates, variances = np.array(estimates), np.array(variances)
  if len(estimates) == 0:
    return 1
  else:
    variance = 1.0 / np.sum(1.0 / variances)
    estimate = variance * np.sum(estimates / variances)
    return max(1, estimate)

def mirror_descent(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    known_total: float | None = None,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.message_passing,
    iters: int = 1000,
    stepsize: float | None = None,
    callback_fn: Callable[[int, chex.Numeric], None] = _DEFAULT_CALLBACK
):
  if isinstance(loss_fn, list):
    if known_total is None:
      known_total = minimum_variance_unbiased_total(loss_fn)
    loss_fn = marginal_loss.from_linear_measurements(loss_fn)
  elif known_total is None:
    raise ValueError('Must set known_total is giving a custom MarginalLossFn')

  if potentials is None:
    potentials = CliqueVector.zeros(domain, loss_fn.cliques)

  if stepsize is None:
    stepsize = 2.0 #2.0 / known_total  # what should this be?

  loss_and_grad = jax.value_and_grad(loss_fn)

  @jax.jit
  def update(theta, alpha):

    mu = marginal_oracle(theta, known_total)
    loss, dL = loss_and_grad(mu)

    theta2 = theta - alpha*dL
    mu2 = marginal_oracle(theta2, known_total)
    loss2 = loss_fn(mu2)
    
    sufficient_decrease = loss - loss2 >= 0.5 * alpha * dL.dot(mu - mu2)
    alpha = jax.lax.select(sufficient_decrease, alpha, 0.5*alpha)
    theta = jax.lax.cond(sufficient_decrease, lambda: theta2, lambda: theta)
    loss = jax.lax.select(sufficient_decrease, loss2, loss)

    return theta, loss, alpha

  for t in range(iters):
    potentials, loss, stepsize = update(potentials, stepsize)
    #if t%50 == 0: print('Stepsize', stepsize)
    callback_fn(t, loss)
   
  marginals = marginal_oracle(potentials, known_total)
  return GraphicalModel(potentials, marginals, known_total)
