from mbi import Dataset, Factor, CliqueVector
from scipy.optimize import minimize
from collections import defaultdict
import numpy as np
from scipy.special import softmax
from functools import reduce
from scipy.sparse.linalg import lsmr

""" This file is experimental.

It is a close approximation to the method described in RAP (https://arxiv.org/abs/2103.06641)
and an even closer approximation to RAP^{softmax} (https://arxiv.org/abs/2106.07153)

Notable differences:
- Code now shares the same interface as Private-PGM (see FactoredInference)
- Named model "MixtureOfProducts", as that is one interpretation for the relaxed tabular format
(at least when softmax is used).
- Added support for unbounded-DP, with automatic estimate of total.
"""


def estimate_total(measurements):
    # find the minimum variance estimate of the total given the measurements
    variances = np.array([])
    estimates = np.array([])
    for Q, y, noise, proj in measurements:
        o = np.ones(Q.shape[1])
        v = lsmr(Q.T, o, atol=0, btol=0)[0]
        if np.allclose(Q.T.dot(v), o):
            variances = np.append(variances, noise**2 * np.dot(v, v))
            estimates = np.append(estimates, np.dot(v, y))
    if estimates.size == 0:
        return 1
    else:
        variance = 1.0 / np.sum(1.0 / variances)
        estimate = variance * np.sum(estimates / variances)
        return max(1, estimate)

def adam(loss_and_grad, x0, iters=250):
    a = 1.0
    b1, b2 = 0.9, 0.999
    eps = 10e-8

    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, iters+1):
        l, g = loss_and_grad(x)
        #print(l)
        m = b1 * m + (1- b1) * g
        v = b2 * v + (1 - b2) * g**2
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        x = x - a * mhat / (np.sqrt(vhat) + eps)
#        print np.linalg.norm(A.dot(x) - y, ord=2)
    return x

class ProductDist:
    def __init__(self, factors, domain, total):
        """
        :param factors: a list of factors
                defined over disjoint subsets of attributes
        :param domain: the domain object
        :param total: known or estimated total
        """
        self.factors = factors
        self.domain = domain
        self.total = total

    def project(self, cols):
        domain = self.domain.project(cols)
        factors = { col : self.factors[col] for col in cols }
        return ProductDist(factors, domain, self.total)

    def datavector(self, flatten=False):
        ans = reduce(lambda x,y: x*y, self.factors.values(), 1.0)
        ans = ans.transpose(self.domain.attrs)
        return ans.datavector(flatten) * self.total

class MixtureOfProducts:
    def __init__(self, products):
        self.products = products
        self.domain = products[0].domain
        self.total = sum(P.total for P in products)

    def project(self, cols):
        return MixtureOfProducts([P.project(cols) for P in self.products])
    
    def datavector(self, flatten=False):
        return sum(P.datavector(flatten) for P in self.products)

class MixtureInference:
    def __init__(self, domain, components=10, metric='L2'):
        """
        :param domain: A Domain object
        :param components: The number of mixture components
        :metric: The metric to use for the loss function (can be callable)
        """
        self.domain = domain
        self.components = components
        self.metric = metric

    def estimate(self, measurements, total=None, alpha=0.1):
        if total == None:
            total = estimate_total(measurements)
        self.measurements = measurements
        cliques = [M[-1] for M in measurements]

        def get_model(params):
            idx = 0
            products = []
            for _ in range(self.components):
                factors = {}
                for col in self.domain:
                    n = self.domain[col]
                    vals = softmax(params[idx:idx+n])
                    idx += n
                    factors[col] = Factor(self.domain.project(col), vals)
                products.append(ProductDist(factors, self.domain, total/self.components))
            return MixtureOfProducts(products)


        def loss_and_grad(params):
            # first create the model
            model = get_model(params)

            # Now calculate the necessary marginals
            mu = CliqueVector.from_data(model, cliques)
            loss, dL = self._marginal_loss(mu)

            # Now back-propagate gradient to params
            dparams = np.zeros(params.size) # what to do with total?
            for cl in dL:
                idx = 0
                for i in range(self.components):
                    submodel = model.products[i]
                    mui = Factor(self.domain.project(cl), submodel.project(cl).datavector())
                    tmp = dL[cl] * mui
                    for col in self.domain:
                        n = self.domain[col]
                        if col in cl:
                            dpij = (tmp / submodel.factors[col]).project([col]).datavector()
                            pij = submodel.factors[col].datavector()
                            dparams[idx:idx+n] += dpij * pij - pij*(pij @ dpij)
                        idx += n
            return loss, dparams
                        
        params = np.random.normal(loc=0, scale=0.25, size=sum(self.domain.shape) * self.components)

        params = adam(loss_and_grad, params)
        return get_model(params)                     

    
    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal 
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = { cl : Factor.zeros(marginals[cl].domain) for cl in marginals }

        for Q, y, noise, cl in self.measurements:
            mu = marginals[cl]
            c = 1.0/noise
            x = mu.datavector()
            diff = c*(Q @ x - y)
            if metric == 'L1':
                loss += abs(diff).sum()
                sign = diff.sign() if hasattr(diff, 'sign') else np.sign(diff)
                grad = c*(Q.T @ sign)
            else:
                loss += 0.5*(diff @ diff)
                grad = c*(Q.T @ diff)
            gradient[cl] += Factor(mu.domain, grad)
        return float(loss), CliqueVector(gradient)
