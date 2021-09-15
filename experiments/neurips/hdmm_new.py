from mbi import mechanism, FactoredInference, RegionGraph, callbacks
import numpy as np
import argparse
from hdmm import workload, templates, error
from mbi import Dataset
import itertools
from autodp import privacy_calibrator
import os
import pandas as pd
import pickle

def convert_matrix(domain, cliques):
    weights = {}
    for proj in cliques:
        tpl = tuple([domain.attrs.index(i) for i in proj])
        weights[tpl] = 1.0
    return workload.Marginals.fromtuples(domain.shape, weights)

def convert_back(domain, Q):
    result = []
    for Qi in Q.matrices:
        wgt = Qi.weight
        key = tuple([domain.attrs[i] for i in Qi.base.tuple()])
        result.append((key, wgt))
    return result


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'fire'
    params['workload'] = 15
    params['iters'] = 10000
    params['epsilon'] = 1.0
    params['delta'] = 0.0
    params['restarts'] = 1
    params['oracle'] = 'convex'
    params['seed'] = 0
    params['save'] = None

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', type=str, help='dataset to use')
    parser.add_argument('--workload', type=int, help='number of marginals in workload')
    parser.add_argument('--oracle', choices=['convex','exact'], help='marginal oracle')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--restarts', type=int, help='number of HDMM restarts')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', type=str, help='path to save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    prng = np.random.RandomState(args.seed)
    prefix = os.getenv('HD_DATA') #'/home/rmckenna/Repos/hd-datasets/clean/'
    attrs = ['ALS Unit', 'Battalion', 'Call Final Disposition', 'Call Type', 'Call Type Group', 'City', 'Final Priority', 'Fire Prevention District', 'Neighborhooods - Analysis Boundaries', 'Original Priority', 'Priority', 'Station Area', 'Supervisor District', 'Unit Type', 'Zipcode of Incident']
    #attrs = attrs[:8]

    name = args.dataset
    
    data = Dataset.load(prefix+name+'.csv', prefix+name+'-domain.json')
    if name == 'fire':
        data = data.project(attrs)

    #A = W

    strategy_path = 'hdmm-%s-%d.pkl' % (name, args.workload)
    if os.path.exists(strategy_path):
        W,A = pickle.load(open(strategy_path, 'rb'))
    else:
        all3way = list(itertools.combinations(data.domain, 3))
        cliques = [all3way[i] for i in prng.choice(len(all3way), args.workload, replace=False)]
        W = convert_matrix(data.domain, cliques)
    
        best_obj = np.inf
        for _ in range(args.restarts):
            temp = templates.Marginals(data.domain.shape, approx=args.delta>0, seed=args.seed)
            obj = temp.optimize(W)
            if obj < best_obj:
                best_obj = obj
                A = temp.strategy()
        pickle.dump((W,A), open(strategy_path, 'wb'))

    workload_cliques = [x[0] for x in convert_back(data.domain, W)]
    strategy_cliques = [x[0] for x in convert_back(data.domain, A)]

    WtW = W.gram()
    AtA1 = A.gram().pinv()
    trace = (WtW @ AtA1).trace()
    prng = np.random
    if args.delta == 0:
        var = 2.0 / args.epsilon**2
        sensitivity = np.linalg.norm(A.weights, 1)
        add_noise = lambda x: x+sensitivity*prng.laplace(loc=0, scale=1.0/args.epsilon, size=x.size)
    else:
        sigma = privacy_calibrator.gaussian_mech(args.epsilon, args.delta)['sigma']
        var = sigma**2
        sensitivity = np.linalg.norm(A.weights, 2)
        add_noise = lambda x: x + sensitivity*prng.normal(loc=0, scale=sigma, size=x.size)

    results = vars(args)
    print(results)
    path = results.pop('save')
    results['expected error'] = var * sensitivity**2 * trace
    tse = var * sensitivity**2 * trace
    rmse = np.sqrt(tse / W.shape[0])

    measurements = []
    for proj, wgt in convert_back(data.domain, A):
        Q = wgt*workload.Identity(data.domain.size(proj))
        x = data.project(proj).datavector()
        y = add_noise(Q @ x)
        measurements.append((Q,y,1.0,proj))

    if args.oracle == 'exact':
        oracle = 'exact'
    else:
        oracle = RegionGraph(data.domain, strategy_cliques, minimal=True, convex=True, iters=1)
        #oracle.belief_propagation = oracle.wiegerinck

    engine = FactoredInference(data.domain,iters=args.iters,marginal_oracle=oracle,log=True) 
    #model = engine.estimate(measurements, engine='RDA', options={'lipschitz':3})
    alg = 'MD3' if args.oracle == 'convex' else 'MD'
    cb = callbacks.Logger(engine, frequency=50)
    model = engine.estimate(measurements, engine=alg, callback=cb)#, options=opts)
    #model = engine.estimate(measurements, engine='MD', callback=cb, options={'stepsize':None})
    #model = engine.estimate(measurements, engine='RDA', callback=cb)

    tse = 0
    for cl in workload_cliques:
        xest = model.project(cl).datavector()
        x = data.project(cl).datavector()
        diff = x - xest
        tse += diff @ diff

    print('Expected Error: RMSE=%.3f' % rmse)
    print('Observed Error: RMSE=%.3f' % np.sqrt(tse / W.shape[0]))

    results['observed error'] = tse
    results['queries'] = W.shape[0]
    results = pd.DataFrame(results, index=[0])
    
    if path is None:
        print(results)
    else:
        with open(path, 'a') as f:
            results.to_csv(f, mode='a', index=False, header=f.tell()==0)
    #rmse = np.sqrt(tse / W.shape[0])
    #print('Error with JunctionTree: RMSE=%.3f' % rmse)
