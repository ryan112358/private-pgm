import numpy as np
from mbi import Domain, callbacks, CliqueVector, Factor
from mbi import marginal_oracles, marginal_loss
from typing import Callable
import jax
import chex

def minimum_variance_unbiased_total(measurements: list[marginal_loss.LinearMeasurement]) -> float:
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
    loss_fn: marginal_loss.MarginalLossFn,
    known_total: float,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.message_passing,
    iters: int = 1000,
    stepsize: float | None = None,
    callback_fn: Callable[[chex.Numeric], None] = print
):
  if potentials is None:
    potentials = CliqueVector.zeros(domain, loss_fn.cliques)

  if stepsize is None:
    stepsize = 2.0 / known_total**2

  loss_and_grad = jax.value_and_grad(loss_fn)

  @jax.jit
  def update(theta: CliqueVector, alpha: chex.Numeric) -> tuple[CliqueVector, chex.Numeric]:
    mu = marginal_oracle(theta, known_total)
    loss, dL = loss_and_grad(mu)
    return theta - alpha*dL, loss

  for _ in range(iters):
    potentials, loss = update(potentials, stepsize)
    callback_fn(loss)
    
  return marginal_oracle(potentials, known_total)
