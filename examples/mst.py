import numpy as np
from mbi import FactoredInference, Dataset, Domain
from scipy import sparse
from disjoint_set import DisjointSet
import networkx as nx
import itertools

"""
This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""

def MST(data, epsilon, delta):
    # This mechanism is designed for relatively large high-dimensional datasets
    # for lower-dimensional datasets (like adult), simpler mechanisms may be better
    sigma = calibrate_gaussian_noise(epsilon*2.0/3.0, delta)
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    cliques = select(data, epsilon/3.0, log1)
    log2 = measure(data, cliques, sigma)
    engine = FactoredInference(data.domain, iters=5000)
    est = engine.estimate(log1+log2)
    synth = est.synthetic_data()
    return undo_compress_fn(synth)

def calibrate_gaussian_noise(epsilon, delta):
    # calibrate noise for 2-fold adaptive composition of sensitivity 1 gaussian mechanisms
    d = np.log(1/delta)
    return (np.sqrt(d) + np.sqrt(d + epsilon)) / epsilon

def measure(data, cliques, sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma/wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append( (Q, y, sigma/wgt, proj) )
    return measurements

def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3*sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append( (Q, y, sigma, proj) )
        else: # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append( (I2, y2, sigma, proj) )
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn

def permute_and_flip(qualities, epsilon, sensitivity=1.0):
    q = qualities - qualities.max()
    p = np.exp(0.5*epsilon/sensitivity*q)
    for i in np.random.permutation(p.size):
        if np.random.rand() <= p[i]:
            return i

def select(data, epsilon, measurement_log, cliques=[]):
    engine = FactoredInference(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a,b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet()

    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))

    for i in range(r-1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        idx = permute_and_flip(wgts, epsilon/(r-1), sensitivity=1.0)
        e = candidates[idx]
        T.add_edge(*e)
        ds.union(*e)

    return list(T.edges)

def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)

def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)

if __name__ == '__main__':
    data = Dataset.load('../data/adult.csv', '../data/adult-domain.json')
    synth = MST(data, 1.0, 1e-6)
    
    # measure error (total variation distance) on 3-way marginals
    errors = []
    for proj in itertools.combinations(data.domain.attrs, 2):
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1) 
        errors.append(e)
    print('Average Error: ', np.mean(errors))
    
