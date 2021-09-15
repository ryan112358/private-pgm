import numpy as np
from ektelo import matrix
from scipy import sparse
from collections import OrderedDict
from functools import reduce
from mbi import Factor, FactoredInference
#from mbi.other import FactoredMultiplicativeWeights
import argparse
import benchmarks
import time

class ProductDist:
    """ factored representation of data from MWEM paper """
    def __init__(self, factors, domain, total):
        """
        :param factors: a list of contingency tables, 
                defined over disjoint subsets of attributes
        :param domain: the domain object
        :param total: known or estimated total
        """
        self.factors = factors
        self.domain = domain
        self.total = total

        for a in domain:
            if not any(a in f.domain for f in factors):
                sub = domain.project([a])
                x = np.ones(domain[a]) / domain[a]
                factors.append(Factor(sub, x))

    def project(self, cols):
        domain = self.domain.project(cols)
        factors = []
        for factor in self.factors:
            pcol = [c for c in cols if c in factor.domain]
            if pcol != []:
                factors.append(factor.project(pcol))
        return ProductDist(factors, domain, self.total)

    def datavector(self):
        ans = reduce(lambda x,y: x*y, self.factors, 1.0)
        ans = ans.transpose(self.domain.attrs)
        return ans.values.flatten() * self.total

class FactoredMultiplicativeWeights:
    def __init__(self, domain, iters = 100):
        self.domain = domain
        self.iters = iters

    def infer(self, measurements, total):
        self.multWeightsFast(measurements, total)
        return self.model

    def multWeightsFast(self, measurements, total):
        domain = self.domain
        groups, projections = _cluster(measurements)
        factors = []
        for group, proj  in zip(groups, projections):
            dom = self.domain.project(proj)
            fact = Factor.uniform(dom)
            for i in range(self.iters):
                update = Factor.zeros(dom)
                for Q, y, noise_scale, p in group:
                    dom2 = dom.project(p)
                    hatx = fact.project(p).values.flatten()*total
                    error = y - Q.dot(hatx)
                    update += Factor(dom2, Q.T.dot(error).reshape(dom2.shape))
                fact *= np.exp(update / (2*total))
                fact /= fact.sum()
            factors.append(fact)

        self.model = ProductDist(factors, self.domain, total) 
 

def _cluster(measurement_cache):
    """
    Cluster the measurements into disjoint subsets by finding the connected 
    components of the graph implied by the measurement projections
    """
    # create the adjacency matrix
    k = len(measurement_cache)
    G = sparse.dok_matrix((k,k))
    for i, (_, _, _, p) in enumerate(measurement_cache):
        for j, (_, _, _, q) in enumerate(measurement_cache):
            if len(set(p) & set(q)) >= 1:
                G[i,j] = 1
    # find the connected components and group measurements
    ncomps, labels = sparse.csgraph.connected_components(G)
    groups = [ [] for _ in range(ncomps) ]
    projections = [ set() for _ in range(ncomps) ]
    for i, group in enumerate(labels):
        groups[group].append(measurement_cache[i])
        projections[group] |= set(measurement_cache[i][3])
    projections = [tuple(p) for p in projections]
    return groups, projections

def average_error(workload, data, est):
    errors = []
    for ax in workload:
        x = data.project(ax).datavector()
        xest = est.project(ax).datavector()
        W = matrix.Identity(x.size)
        ans = W.dot(x - xest)
        err = np.linalg.norm(W.dot(x-xest), 1) / np.linalg.norm(W.dot(x), 1)
        errors.append(err)
    return np.mean(errors)

def worst_approximated(data, est, workload, eps, prng=None):
    if prng == None:
        prng = np.random
    K = len(workload)
    errors = np.zeros(K)
    log = np.zeros(K)
    
    #answers = est.calculate_many_marginals(workload)
    for i, ax in enumerate(workload):
        x = data.project(ax).datavector()
        xest = est.project(ax).datavector()
        #xest = answers[ax].datavector()
        W = matrix.Identity(x.size)
        ans = W.dot(x - xest)
        log[i] = np.linalg.norm(ans, 1) / np.linalg.norm(W.dot(x), 1)
        errors[i] = np.linalg.norm(ans, 1) - W.shape[0]
    merr = np.max(errors)
    #print('Max error', merr, np.mean(errors), np.median(errors))
    print(np.max(log), np.mean(log), np.median(log))
    prob = np.exp(eps * (errors - merr) / 2.0)
    
    key = prng.choice(K, p = prob/prob.sum())
    ax = workload[key]
    Q = matrix.Identity(data.domain.size(ax))
    return ax, Q

def mwem(workload, data, eps, engine, iters=10, prng=None, out=None):
    workload_copy = [proj for proj in workload]
    t0 = time.time()
    if iters is None: # run until it crashes
        eps1 = eps/2.0
        iters = len(workload)
    else:
        eps1 = eps / (2.0 * iters)
    total = data.df.shape[0]

    if out: f = open(out, 'w')
    #engine = FactoredMultiplicativeWeights(data.domain)
    #engine = FactoredInference(data.domain, log=False, iters=1000)
    measurements = []
    est = engine.infer(measurements, total)
    for i in range(iters):
        ta = time.time()
        ax, Q = worst_approximated(data, est, workload, eps1, prng)
        tb = time.time()
        proj = data.project(ax)
        ans = Q.dot(proj.datavector())
        noise = np.random.laplace(loc=0, scale=2.0 / eps1, size=Q.shape[0])
        y = ans + noise
        measurements.append( (Q, y, np.sqrt(2)/eps1, ax) )
        print(i, proj.domain)
        est = engine.infer(measurements, total)
        tc = time.time()
        workload.remove(ax)
        try: size = est.potentials.size()
        except: size = sum(f.domain.size() for f in est.factors)
        print('%.3f MB' % (8*size/2**20))
        if out:
            err = average_error(workload_copy, data, est)
            dt = time.time() - t0
            s = '%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f \n' % (i, ' x '.join(ax), 8*size/2**20, dt, tb-ta, tc-tb, err)
            f.write(s)
            f.flush()

    if out: f.close()

    return est

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'adult'
    params['workload'] = 500
    params['engine'] = 'MW'
    params['iters'] = 1000
    params['rounds'] = None
    params['epsilon'] = 0.1
    params['seed'] = 0
    params['save'] = None

    return params 
 

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['adult','titanic','msnbc','loans','nltcs','fire','stroke','salary'], help='dataset to use')
    parser.add_argument('--workload', type=int, help='number of marginals in workload')
    parser.add_argument('--engine', choices=['MW','MB'], help='inference engine')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--rounds', type=int, help='number of rounds to run mwem')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', action='store_true', help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data, workloads = benchmarks.random3way(args.dataset, args.workload)
    prng = np.random.RandomState(args.seed)

    if args.engine == 'MW':
        engine = FactoredMultiplicativeWeights(data.domain, iters=args.iters)
    else:
        engine = FactoredInference(data.domain, iters=args.iters) 

    if args.save:
        out = 'results/mwem/%s_%s_%s.out' % (args.dataset, args.engine, args.seed)
    else:
        out = None

    ans = mwem(workloads, data, args.epsilon, engine, iters=args.rounds, prng=prng, out=out)
