
import jax.nn
import jax.numpy as jnp
import pandas as pd

from mbi import (CliqueVector, Dataset, Domain, Factor, LinearMeasurement,
                 estimation, marginal_loss)

"""Implements an experimental Mixture of Products model for synthetic data generation.

This module provides an implementation of the Mixture of Products model, which
approximates methods like RAP (https://arxiv.org/abs/2103.06641) and
RAP^{softmax} (https://arxiv.org/abs/2106.07153). It aims to generate synthetic
data by learning a mixture of simpler product distributions that match given
marginal constraints.

Notable differences from some related works include:
- Adherence to the common interface used by other methods like Private-PGM.
- Use of the name "MixtureOfProducts" reflecting the model structure.
- Support for unbounded differential privacy with automatic total estimation.

Note: This implementation is experimental and may not be fully optimized.
Contributions are welcome.
"""


def adam(loss_and_grad, x0, iters=250):
    """Implements the Adam optimization algorithm (based on Kingma & Ba, 2014)."""
    # TODO: Rewrite using optax
    a = 1.0
    b1, b2 = 0.9, 0.999
    eps = 10e-8

    x = x0
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    for t in range(1, iters + 1):
        # l is unused
        _, g = loss_and_grad(x)
        # print(l)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        x = x - a * mhat / (jnp.sqrt(vhat) + eps)
    return x
def synthetic_col(counts, total):
    """Generates a synthetic column by rounding fractional counts based on total."""
    counts *= total / counts.sum()
    frac, integ = np.modf(counts)
    integ = integ.astype(int)
    extra = total - integ.sum()
    if extra > 0:
        idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
        integ[idx] += 1
    vals = np.repeat(np.arange(counts.size), integ)
    np.random.shuffle(vals)
    return vals
class MixtureOfProducts:
    """Represents a probability distribution as a mixture of product distributions.

    Stores the component product distributions (probabilities for each attribute
    within each component) and the overall total count/mass.
    """
    def __init__(self, products, domain, total):
        """Initializes the MixtureOfProducts object with components, domain, and total."""
        self.products = products
        self.domain = domain
        self.total = total
        self.num_components = next(iter(products.values())).shape[0]

    def project(self, cols):
        """Projects the mixture model onto a subset of specified columns."""
        products = {col: self.products[col] for col in cols}
        domain = self.domain.project(cols)
        return MixtureOfProducts(products, domain, self.total)

    def datavector(self, flatten=True):
        """Computes the data vector (histogram) representation of the mixture distribution."""
        d = len(self.domain)
        letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[:d]
        formula = ",".join(["a%s" % l for l in letters]) + "->" + "".join(letters)
        components = [self.products[col] for col in self.domain]
        ans = jnp.einsum(formula, *components) * self.total / self.num_components
        return ans.flatten() if flatten else ans

    def synthetic_data(self, rows=None):
        """Generates synthetic tabular data samples from the mixture model."""
        total = rows or int(self.total)
        subtotal = total // self.num_components + 1

        dfs = []
        for i in range(self.num_components):
            df = pd.DataFrame()
            for col in self.products:
                counts = self.products[col][i]
                df[col] = synthetic_col(counts, subtotal)
            dfs.append(df)

        df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)[:total]
        return Dataset(df, self.domain)


def mixture_of_products(
    domain: Domain,
    loss_fn: marginal_loss.MarginalLossFn | list[LinearMeasurement],
    *,
    known_total: int | None = None,
    mixture_components: int = 100,
    iters: int = 2500,
    # alpha: float = 0.1 # Unused parameter
) -> MixtureOfProducts:

    loss_fn,known_total, _ = estimation._initialize(domain, loss_fn, known_total, None)

    one_hot_features = sum(domain.shape)
    params = np.random.normal(
        loc=0, scale=0.25, size=(mixture_components, one_hot_features)
    )

    cliques = loss_fn.cliques

    letters = "bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def get_products(params):
        """Converts raw parameters into per-attribute product distribution components via softmax."""
        products = {}
        idx = 0
        for col in domain:
            n = domain[col]
            products[col] = jax.nn.softmax(params[:, idx : idx + n], axis=1)
            idx += n
        return products

    def marginals_from_params(params):
        """Computes the marginals (as a CliqueVector) implied by the mixture parameters."""
        products = get_products(params)
        arrays = {}
        for cl in cliques:
            let = letters[: len(cl)]
            formula = ",".join(["a%s" % l for l in let]) + "->" + "".join(let)
            components = [products[col] for col in cl]
            ans = jnp.einsum(formula, *components) * known_total / mixture_components
            arrays[cl] = Factor(domain.project(cl), ans)
        return CliqueVector(domain, cliques, arrays)

    def params_loss(params: jax.Array) -> float:
        """Calculates the loss based on the marginals derived from current parameters."""
        mu = marginals_from_params(params)
        return loss_fn(mu)

    params_loss_and_grad = jax.value_and_grad(params_loss)

    params = adam(params_loss_and_grad, params, iters=iters)
    products = get_products(params)
    return MixtureOfProducts(products, domain, known_total)

