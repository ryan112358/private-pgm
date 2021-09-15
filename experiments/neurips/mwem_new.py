import numpy as np
from hdmm import matrix
from scipy import sparse
from collections import OrderedDict
from functools import reduce
from mbi import Factor, FactoredInference, RegionGraph, Dataset, callbacks
import itertools
#from mbi.other import FactoredMultiplicativeWeights
import argparse
import benchmarks
import time
import os
from scipy.special import logsumexp

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

    def estimate(self, measurements, total, callback=None):
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

    score = eps*(errors - merr) / 2.0
    prob = np.exp(score - logsumexp(score))
    
    key = prng.choice(K, p = prob)
    ax = workload[key]
    Q = matrix.Identity(data.domain.size(ax))
    return ax, Q

def mwem(workload, data, eps_per_iter, engine, prng=None, out=None):
    workload_copy = [proj for proj in workload]
    t0 = time.time()
    total = data.df.shape[0]
    eps1 = eps_per_iter/2

    if out: f = open(out, 'w')
    #engine = FactoredMultiplicativeWeights(data.domain)
    #engine = FactoredInference(data.domain, log=False, iters=1000)
    measurements = []
    est = engine.estimate(measurements, total)
    cb = None #callbacks.Logger(engine, frequency=1)
    for i in range(len(workload)):
        ta = time.time()
        print('checkpt')
        ax, Q = worst_approximated(data, est, workload, eps1, prng)
        tb = time.time()
        proj = data.project(ax)
        ans = Q.dot(proj.datavector())
        noise = np.random.laplace(loc=0, scale=2.0 / eps1, size=Q.shape[0])
        y = ans + noise
        #measurements.append( (Q, y, np.sqrt(2)/eps1, ax) )
        measurements.append( (Q, y, 1.0, ax) )
        print(i, proj.domain)
        est = engine.estimate(measurements, total, callback=cb)#, options={'stepsize':0.25})
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
    params['dataset'] = 'fire'
    params['engine'] = 'MW'
    params['iters'] = 100
    params['epsilon_per_iter'] = 0.01
    params['seed'] = 0
    params['save'] = None

    return params 
 

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--engine', choices=['MW','MB-exact', 'MB-convex'], help='inference engine')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--epsilon_per_iter', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', action='store_true', help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    #data, workloads = benchmarks.random3way(args.dataset, args.workload)
    prng = np.random.RandomState(args.seed)
   
    prefix = os.getenv('HD_DATA') #'/home/rmckenna/Repos/hd-datasets/clean/'
    attrs = ['ALS Unit', 'Battalion', 'Call Final Disposition', 'Call Type', 'Call Type Group', 'City', 'Final Priority', 'Fire Prevention District', 'Neighborhooods - Analysis Boundaries', 'Original Priority', 'Priority', 'Station Area', 'Supervisor District', 'Unit Type', 'Zipcode of Incident']

    name = args.dataset

    data = Dataset.load(prefix+name+'.csv', prefix+name+'-domain.json')

    if name == 'fire':
        data = data.project(attrs)

    workload = list(itertools.combinations(data.domain, 2))
 

    if args.engine == 'MW':
        engine = FactoredMultiplicativeWeights(data.domain, iters=args.iters)
    elif args.engine == 'MB-exact':
        engine = FactoredInference(data.domain, iters=args.iters, warm_start=False) 
    else:
        model = RegionGraph(data.domain, workload, minimal=True, convex=True, iters=1)
        model = 'convex'
        # Don't use warm start with approximate inference, things get finnicky
        engine = FactoredInference(data.domain,iters=args.iters,marginal_oracle=model,warm_start=False,log=False)
    if args.save:
        out = 'results/mwem_%s_%s_%.3f_%s.out' % (args.dataset, args.engine, args.epsilon_per_iter, args.seed)
    else:
        out = None

    ans = mwem(workload, data, args.epsilon_per_iter, engine, prng=prng, out=out)
