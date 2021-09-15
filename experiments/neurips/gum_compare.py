from mbi import FactoredInference, GraphicalModel, Domain, CliqueVector, RegionGraph, FactorGraph, Dataset, Factor
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
from gum.gum import transform
from autodp import privacy_calibrator

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['iters'] = 1000
    params['dataset'] = 'fire'
    params['measurements'] = 5
    params['epsilon'] = 1.0
    params['kway'] = 3
    params['seed'] = 0
    params['oracle'] = 'exact'
    params['save'] = None

    return params

def get_random_cliques(attrs, size, number, prng=np.random):
    allcl = list(itertools.combinations(attrs, size))
    idx = prng.choice(len(allcl), number, replace=False)
    return [allcl[i] for i in idx]

def compute_error(data_marg, model_marg):
    error = 0
    for cl in data_marg:
        x = data_marg[cl].datavector()
        y = model_marg[cl].datavector()
        error += 0.5*np.abs(x-y).sum() / x.sum()
    return error / len(data_marg)  

class GUM:
    def __init__(self, domain):
        self.domain = domain

    def estimate(self, measurements, callback=None):
        noisy_mu = {}
        for _, y, _, cl in measurements:
            noisy_mu[cl] = Factor(self.domain.project(cl), y)
        mu = transform(noisy_mu, self.domain, consistent_iter=100)
        cliques = list(mu.keys())
        model = lambda: None
        model.project = lambda cl: mu[cl] if cl in cliques else Factor.zeros(self.domain.project(cl))
        return model
        

if __name__ == '__main__':

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--oracle',choices=['exact','convex','gum'],help='marginal oracle')
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--measurements', type=int, help='number of measurements')
    parser.add_argument('--kway', type=int, help='size of marginals (2-way, 3-way, etc.)')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', type=str, help='save results')

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

    cliques = get_random_cliques(data.domain.attrs, args.kway, args.measurements, prng)
    
    measurements = []
    noise = privacy_calibrator.gaussian_mech(args.epsilon, 1e-6)['sigma']*np.sqrt(args.measurements)
    noisy_mu = {}
    for cl in cliques:
        x = data.project(cl).datavector() 
        y = x + np.random.normal(loc=0, scale=noise, size=x.size)
        I = sparse.eye(x.size)
        measurements.append((I, y, 1.0, cl))
        noisy_mu[cl] = Factor(data.domain.project(cl), y)

    if args.oracle == 'gum':
        engine = GUM(data.domain)
    else:
        engine = FactoredInference(data.domain,iters=args.iters,marginal_oracle=args.oracle,log=True)

    #alg = 'RDA' if args.oracle == 'exact' else 'MD3'
    t0 = time.time()
    if args.oracle == 'convex' and args.epsilon == 0.01:
        model = engine.estimate(measurements, engine='MD', options={'stepsize':0.00001})
    else:
        model = engine.estimate(measurements)
    t1 = time.time()
    data_marg = CliqueVector.from_data(data, cliques)
    model_marg = CliqueVector.from_data(model, cliques)
    

    results = vars(args)
    path = results.pop('save')
    results['noisy_error'] = compute_error(data_marg, noisy_mu)
    results['error'] = compute_error(data_marg, model_marg)
    results['time'] = t1 - t0
    results = pd.DataFrame(results, index=[0])

    if path is None:
        print(results)
    else:
        with open(path, 'a') as f:
            results.to_csv(f, mode='a', index=False, header=f.tell()==0)
