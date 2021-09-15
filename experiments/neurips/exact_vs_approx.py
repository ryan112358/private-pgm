from mbi import FactoredInference, GraphicalModel, Domain, CliqueVector, RegionGraph, FactorGraph
from mbi import callbacks
import numpy as np
from IPython import embed
import argparse
import itertools
from scipy import sparse
import pandas as pd
from multiprocessing import Pool
from functools import partial
import time
import os
from scipy.sparse.csgraph import minimum_spanning_tree

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['iters'] = 1000
    params['attributes'] = 10
    params['domain'] = 2
    params['measurements'] = 5
    params['records'] = 10000
    params['epsilon'] = 1.0
    params['seed'] = 0
    params['oracle'] = 'exact'
    params['temperature'] = 3.0
    params['selection'] = 'random'
    params['save'] = None

    return params

def get_random_cliques(attrs, size, number, prng=np.random):
    allcl = list(itertools.combinations(attrs, size))
    idx = prng.choice(len(allcl), number, replace=False)
    return [allcl[i] for i in idx]

def compute_error(data_marg, model_marg):
    error = 0
    for cl in data_marg:
        x = data_marg[cl]
        y = model_marg[cl]
        error += 0.5*np.abs(x-y).sum() / x.sum()
    return error / len(data_marg)  

def tree(dims, vals=10, prng=np.random, total=10000, temperature=3.0):
    W =  prng.rand(dims, dims)
    cliques = list(zip(*minimum_spanning_tree(W).toarray().nonzero()))
    cliques = [tuple(str(i) for i in cl) for cl in cliques]
    attrs = [str(i) for i in range(dims)]
    domain = Domain(attrs, [vals]*dims)

    model = GraphicalModel(domain, cliques, total=total)
    model.potentials = CliqueVector.normal(domain, model.cliques, prng)*temperature
    return model.synthetic_data()


if __name__ == '__main__':

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--oracle',choices=['exact','approx','approx2','pairwise','gum'],help='marginal oracle')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--attributes', type=int, help='number of attributes')
    parser.add_argument('--domain', type=int, help='domain size per attribute')
    parser.add_argument('--measurements', type=int, help='number of measurements')
    parser.add_argument('--records', type=int, help='number of records in synthetic data')
    parser.add_argument('--temperature', type=float, help='temperature of synthetic data')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--selection', choices=['random','greedy'], help='measurement selection')
    parser.add_argument('--skiperror', action='store_true', help='skip error calculations')
    parser.add_argument('--save', type=str, help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    prng = np.random.RandomState(args.seed)


    # Make the synthetic data
    data = tree(args.attributes, args.domain, prng, args.records, args.temperature)
    print('Data has been generated')

    num_measurements = args.measurements
    all_3way = list(itertools.combinations(data.domain, 3))
    if args.selection == 'random':
        in_cliques = get_random_cliques(data.domain.attrs, 3, num_measurements, prng)
    elif args.selection == 'greedy':
        in_cliques = sorted(all_3way, key=lambda t: int(t[-1])-int(t[0]))[:num_measurements]
    
    measurements = []
    noise = num_measurements / args.epsilon
    out_cliques = list(set(all_3way) - set(in_cliques))
    noisy_in = {}
    for cl in in_cliques:
        x = data.project(cl).datavector() 
        y = x + prng.laplace(loc=0, scale=noise, size=x.size)
        I = sparse.eye(x.size)
        measurements.append((I, y, 1.0, cl))
        noisy_in[cl] = y


    if args.oracle == 'exact':
        engine = FactoredInference(data.domain, iters=args.iters, marginal_oracle='exact', log=True)
        t0 = time.time()
        if args.learning_rate is not None:
            opts = {'stepsize':args.learning_rate}
            model = engine.estimate(measurements, engine='MD2', options=opts)
        else: 
            model = engine.estimate(measurements, engine='MD')
        t1 = time.time()

    elif args.oracle == 'approx':
        oracle = RegionGraph(data.domain, in_cliques, minimal=True, convex=True, iters=1)
        engine = FactoredInference(data.domain, iters=args.iters, marginal_oracle=oracle, log=True)
        t0 = time.time()
        cb = None #callbacks.Logger(engine, frequency=1)
        opts = {'stepsize':args.learning_rate}
        #model = engine.estimate(measurements, engine='MD2', options=opts, callback=cb)
        model = engine.estimate(measurements, engine='MD3', callback=cb)
        t1 = time.time()

    elif args.oracle == 'approx2':
        oracle = RegionGraph(data.domain, all_3way, minimal=True, convex=True, iters=1)
        engine = FactoredInference(data.domain, iters=args.iters, marginal_oracle=oracle, log=True)
        t0 = time.time()
        cb = None #callbacks.Logger(engine, frequency=1)
        opts = {'stepsize':args.learning_rate}
        model = engine.estimate(measurements, engine='MD2', options=opts, callback=cb)
        t1 = time.time()
    
    elif args.oracle == 'pairwise':
        oracle = FactorGraph(data.domain, in_cliques, convex=True, iters=1)
        engine = FactoredInference(data.domain, iters=args.iters, marginal_oracle=oracle, log=True)
        t0 = time.time()
        cb = None #callbacks.Logger(engine, frequency=1)
        opts = {'stepsize':args.learning_rate}
        model = engine.estimate(measurements, engine='MD2', options=opts, callback=cb)
        t1 = time.time()

    if args.oracle == 'gum':
        from gum.gum import transform
        from mbi import Factor
        noisy_marginals = {}
        for _, y, _, cl in measurements:
            noisy_marginals[cl] = Factor(data.domain.project(cl), y)
        t0 = time.time()
        mu = transform(noisy_marginals, data.domain, consistent_iter=1000)
        t1 = time.time()
        model = lambda: None
        model.project = lambda cl: mu[cl] if cl in in_cliques else Factor.zeros(data.domain.project(cl))

    results = vars(args)
    path = results.pop('save')
    results['time'] = t1 - t0
    if args.skiperror:
        results['error_in'] = None
        results['error_out'] = None
        results['baseline_in'] = None
        results['dist_noisy'] = None
    else:
        data_in = { cl : data.project(cl).datavector() for cl in in_cliques }
        model_in = { cl : model.project(cl).datavector() for cl in in_cliques }
        data_out = { cl : data.project(cl).datavector() for cl in out_cliques }
        model_out = { cl : model.project(cl).datavector() for cl in out_cliques }
        results['error_in'] = compute_error(data_in, model_in)
        results['error_out'] = compute_error(data_out, model_out)
        results['baseline_in'] = compute_error(data_in, noisy_in)
        results['dist_noisy'] = compute_error(model_in, noisy_in)
    results = pd.DataFrame(results, index=[0])
    #embed(); import sys; sys.exit()

    if path is None:
        print(results)
    else:
        with open(path, 'a') as f:
            results.to_csv(f, mode='a', index=False, header=f.tell()==0)
