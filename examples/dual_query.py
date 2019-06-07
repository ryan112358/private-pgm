from mbi import Dataset, Factor, FactoredInference, Domain
from mbi import graphical_model, callbacks
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad
import benchmarks
from IPython import embed
import pandas as pd
import argparse
from ektelo import matrix

""" 
An implementation of DualQuery, with and without graphical-model based inference.

Gaboardi, Marco, Emilio Jesús Gallego Arias, Justin Hsu, Aaron Roth, and Zhiwei Steven Wu. "Dual query: Practical private query release for high dimensional data." In International Conference on Machine Learning, pp. 1170-1178. 2014.

This file demonstrates an example where the graphical model is estimated from non-linear
measurements.  Instead of passing in a set of (Q, y, noise, proj) tuples, we instead pass
in a custom marginal loss function, which calculates the negative log likelihood of the 
dual query observations.

Note that we solve the hard dual query problem *exactly* using max-sum variable elimination,
rather than CPLEX as suggested in the DualQuery paper.
"""

class Negated(matrix.EkteloMatrix):
    def __init__(self, Q):
        self.Q = Q
        self.shape = Q.shape
        self.dtype = np.float64

    @property
    def matrix(self):
        return 1 - self.Q.dense_matrix()
    
    def _matvec(self, x):
        return x.sum() - self.Q.dot(x)
    
    def _transpose(self):
        return Negated(self.Q.T)

def max_sum_ve(factors, domain = None, elim = None):
    """ run max-product variable elimination on the factors
    return the most likely assignment as a dictionary where
        keys are attributes
        values are elements of the domain
    """
    # step 0: choose an elimination order
    if domain is None:
        domain = reduce(Domain.merge, [F.domain for F in factors])

    if elim is None:
        cliques = [F.domain.attrs for F in factors]
        elim = graphical_model.greedy_order(domain, cliques, domain.attrs)

    # step 1: variable elimination
    k = len(factors)
    phi = dict(zip(range(k), factors))
    psi = {}
    for z in elim:
        phi2 = [phi.pop(i) for i in list(phi.keys()) if z in phi[i].domain]
        psi[z] = sum(phi2, Factor.ones(domain.project(z)))
        phi[k] = psi[z].max([z])
        k += 1

    value = phi[k-1]

    # step 2: traceback-MAP
    x = { }
    for z in reversed(elim):
        x[z] = psi[z].condition(x).values.argmax()

    # step 3 convert to a Dataset object
    df = pd.DataFrame(x, index=[0])
    return Dataset(df, domain)

def answer_workload(workload, data):
    ans = [W.dot(data.project(cl).datavector()) for cl, W in workload]
    return np.concatenate(ans)

def DualQuery(data, workload, eps=1.0, delta=0.001, seed=0):
    prng = np.random.RandomState(seed)
    total = data.df.shape[0]
    domain = data.domain
    answers = answer_workload(workload, data) / total
    
    nu = 2.0
    s = 50
    #T = int(0.5 * ( np.sqrt(4 * eps * total + s * nu) / np.sqrt(s*nu) + 1 ))
    T = 2
    while 2*nu*(T-1)/total * (np.sqrt(2*s*(T-1)*np.log(1.0/delta) + s*(T-1)*np.exp(2*nu*(T-1)/total)-1)) < eps:
        T = T + 1
    T = T - 1   
 
    Qsize = sum(W.shape[0] for _, W in workload)
    Xsize = data.domain.size()
    
    Q = np.ones(Qsize) / Qsize
    cache = []
    #lookup = [Factor(domain.project(cl), q) for cl, W in workload for q in W]
    lookup = [(cl, W, i) for cl, W in workload for i in range(W.shape[0])]
    results = []
    
    for i in range(T):
        idx = prng.choice(Qsize, s, True, Q)
       
        #queries = [lookup[i] for i in idx]
        queries = []
        for i in idx:
            cl, W, e = lookup[i]
            dom = domain.project(cl)
            n = W.shape[0]
            z = np.zeros(n)
            z[e] = 1.0
            q = W.T.dot(z)
            queries.append(Factor(dom, -q))

        best = max_sum_ve(queries, data.domain)
        curr = answer_workload(workload, best)

        Q *= np.exp(-nu * (answers - curr))
        Q /= Q.sum()
        
        cache.append( (idx, curr) )
        results.append(best.df)

    synthetic = Dataset(pd.concat(results), data.domain)
       
    print('Iterations', T) 
    print('Privacy level', nu*T*(T-1)*s / total)
    
    delta = 1e-3
    eps = 2*nu*(T-1)/total * (np.sqrt(2*s*(T-1)*np.log(1.0/delta) + s*(T-1)*np.exp(2*nu*(T-1)/total)-1))
    print('Approx privacy level', eps, delta)
    
    return synthetic, cache

def log_likelihood(answers, cache):
    nu = 2.0 
    Qsize = sum(W.shape[0] for _, W in workload)
    logQ = np.zeros(Qsize) - np.log(Qsize) 
    ans = 0
    for idx, a in cache:
        probas = logQ[idx]
        logQ = logQ - nu * (answers - a)
        logQ = logQ - logsumexp(logQ)
        ans += np.sum(probas)
    return -ans

def marginal_loss(marginals, workload, cache):
    answers = []
    for proj, W in workload:
        for cl in marginals:
            if set(proj) <= set(cl):
                mu = marginals[cl].project(proj)
                x = mu.values.flatten()
                answers.append(W.dot(x))
                break
    total = x.sum()
    answers = np.concatenate(answers) / total

    gradient = grad(log_likelihood, argnum=0)
    loss = log_likelihood(answers, cache)
    danswers = gradient(answers, cache)

    i = 0
    gradients = { cl : Factor.zeros(marginals[cl].domain) for cl in marginals }
    for proj, W in workload:
        for cl in marginals:
            if set(proj) <= set(cl):
                m = W.shape[0]
                dmu = W.T.dot(danswers[i:i+m]) / total
                dom = gradients[cl].domain.project(proj)
                gradients[cl] += Factor(dom, dmu)
                i += m
                break

    print(loss)
    return loss, graphical_model.CliqueVector(gradients)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'adult'
    params['iters'] = 10000
    params['epsilon'] = 1.0
    params['seed'] = 0

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['adult'], help='dataset to use')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data, workloads = benchmarks.adult_benchmark()

    total = data.df.shape[0]
   
    workload = []
    for cl, W in workloads:
        workload.append( (cl, matrix.VStack([W, Negated(W)])) )

    synthetic, cache = DualQuery(data, workload, eps=args.epsilon, delta=1e-3, seed=args.seed)

    metric = lambda marginals: marginal_loss(marginals, workload, cache)

    engine = FactoredInference(data.domain, metric=metric, iters=args.iters)
    measurements = [(Q, None, 1.0, cl) for cl, Q in workload]

    ans = engine.mirror_descent(measurements, total)
    model = engine.model
    mb = []
    dq = []

    for proj, W in workloads:
        est = W.dot(model.project(proj).datavector())
        x = synthetic.project(proj).datavector()
        x *= total / x.sum()
        est2 = W.dot(x)
        true = W.dot(data.project(proj).datavector())
        err = np.abs(est - true).sum() / np.abs(true).sum()
        err2 = np.abs(est2 - true).sum() / np.abs(true).sum()
        mb.append(err)
        dq.append(err2)

    print('Error of DualQuery    : %.3f' % np.mean(dq))
    print('Error of DualQuery+PGM: %.3f' % np.mean(mb))
