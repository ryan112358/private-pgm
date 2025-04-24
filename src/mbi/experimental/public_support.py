from ..dataset import Dataset
from ..factor import Factor
from ..clique_vector import CliqueVector
from .. import estimation
from ..domain import Domain
from .. import marginal_loss
from ..marginal_loss import LinearMeasurement
from scipy.optimize import minimize
from collections import defaultdict
import numpy as np
from scipy.sparse.linalg import lsmr
from scipy.special import logsumexp
import jax

""" This file is experimental.  
It is an attempt to re-implement and generalize the technique used in PMW^{Pub}.
https://arxiv.org/pdf/2102.08598.pdf. This implementation is not optimized
at all after the refactoring in 2024. Pull requestes are welcome on this file. 

Notable differences:
- Shares the same interface as other estimators in this repo.
- Supports unbounded differential privacy, with automatic estimate of total
- Supports arbitrary measurements over the data marginals, or more generally any MarginalLossFn.
"""


def entropic_mirror_descent(loss_and_grad, x0, total, iters=250):
    logP = np.log(x0 + np.nextafter(0, 1)) + np.log(total) - np.log(x0.sum())
    P = np.exp(logP)
    P = x0 * total / x0.sum()
    loss, dL = loss_and_grad(P)
    alpha = 1.0
    begun = False

    for _ in range(iters):
        logQ = logP - alpha * dL
        logQ += np.log(total) - logsumexp(logQ)
        Q = np.exp(logQ)
        # Q = P * np.exp(-alpha*dL)
        # Q *= total / Q.sum()
        new_loss, new_dL = loss_and_grad(Q)

        if loss - new_loss >= 0.5 * alpha * dL.dot(P - Q):
            # print(alpha, loss)
            logP = logQ
            loss, dL = new_loss, new_dL
            # increase step size if we haven't already decreased it at least once
            if not begun:
                alpha *= 2
        else:
            alpha *= 0.5
            begun = True

    return np.exp(logP)

def _to_clique_vector(data, cliques):
    arrays = {}
    for cl in cliques:
        dom = data.domain.project(cl)
        vals = data.project(cl).datavector(flatten=False)
        arrays[cl] = Factor(dom, vals)
    return CliqueVector(dom, cliques, arrays)


def public_support(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    public_data: Dataset,
    known_total=None
) -> Dataset:

    loss_fn, known_total, _ = estimation._initialize(domain, loss_fn, known_total, None)
    loss_and_grad_mu = jax.value_and_grad(loss_and_grad_mu)

    cliques = loss_fn.cliques

    def loss_and_grad(weights):
        est = Dataset(public_data.df, public_data.domain, weights)
        mu = _to_clique_vector(est, cliques)
        loss, dL = loss_and_grad_mu(mu)
        dweights = np.zeros(weights.size)
        for cl in dL.cliques:
            idx = est.project(cl).df.values
            dweights += np.array(dL[cl].values[tuple(idx.T)])
        return loss, dweights

    weights = np.ones(public_data.records)
    weights = entropic_mirror_descent(loss_and_grad, weights, known_total)
    return Dataset(public_data.df, public_data.domain, weights)
