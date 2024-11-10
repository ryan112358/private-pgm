import numpy as np
from mbi import Domain, GraphicalModel, callbacks, CliqueVector, Factor
from mbi import marginal_oracles, marginal_loss
from scipy.sparse.linalg import LinearOperator, eigsh, lsmr, aslinearoperator
from scipy import optimize, sparse
from functools import partial
from collections import defaultdict
from typing import Callable
import jax
import functools
import chex


def mirror_descent(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn,
    known_total: float,
    potentials: CliqueVector | None = None,
    marginal_oracle=marginal_oracles.brute_force_marginals,
    iters: int = 1000,
    stepsize: float | None = None:,
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
