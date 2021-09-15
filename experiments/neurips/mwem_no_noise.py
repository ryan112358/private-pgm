import numpy as np
from hdmm import matrix
from scipy import sparse
from collections import OrderedDict
from functools import reduce
from mbi import Factor, FactoredInference, RegionGraph, Dataset, callbacks, FactorGraph, CliqueVector, GraphicalModel
import itertools
#from mbi.other import FactoredMultiplicativeWeights
import argparse
import time
import os
from scipy.special import logsumexp
from scipy import optimize

def average_error(workload, data, est):
    errors = []
    for ax in workload:
        x = data.project(ax).datavector()
        xest = est.project(ax).datavector()
        W = matrix.Identity(x.size)
        ans = W.dot(x - xest)
        err = np.linalg.norm(W.dot(x-xest), 1) / np.linalg.norm(W.dot(x), 1)
        #print(ax, err)
        errors.append(err)
    return np.mean(errors)

def worst_approximated(data, est, workload):
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
    #merr = np.max(errors)
    #print('Max error', merr, np.mean(errors), np.median(errors))
    print(np.max(log), np.mean(log), np.median(log))
    key = np.argmax(errors)
    ax = workload[key]
    return ax

def learn_model_exact(domain, total, marginals):
    data_marginals = CliqueVector(marginals)*(1.0/total)
    original_cliques = list(marginals.keys())
    model = GraphicalModel(domain, original_cliques, 1.0)
    if len(marginals) == 0:
        model.potentials = CliqueVector.zeros(domain, model.cliques)
        model.marginals = model.belief_propagation(model.potentials)
        model.total = total
        return model
    index = {}
    idx = 0
    for cl in original_cliques:
        end = idx + domain.size(cl)
        index[cl] = (idx, end)
        idx = end

    def to_vector(mu):
        return np.concatenate([mu[cl].datavector() for cl in original_cliques])

    def to_cliquevector(vector):
        marginals = {}
        for cl in original_cliques:
            start, end = index[cl]
            dom = domain.project(cl)
            marginals[cl] = Factor(dom, vector[start:end])
        return CliqueVector(marginals)
    
    def expand(pot):
        potentials = CliqueVector.zeros(domain, model.cliques)
        for cl in pot:
            for cl2 in model.cliques:
                if set(cl) <= set(cl2):
                    potentials[cl2] += pot[cl]
                    break
        return potentials
    
    def contract(mu):
        mu2 = {}
        for cl in original_cliques:
            for cl2 in mu:
                if set(cl) <= set(cl2):
                    mu2[cl] = mu[cl2].project(cl)
                    break
        return CliqueVector(mu2)

    def loss_and_grad(potentials):
        pot = to_cliquevector(potentials)
        pot2 = expand(pot)
        logZ = model.belief_propagation(pot2, logZ=True)
        model_marginals = contract(model.belief_propagation(pot2))
        
        #print(pot, model_marginals)

        loss = pot.dot(data_marginals) - logZ
        grad = data_marginals + -1*model_marginals
        return -loss, -to_vector(grad)

    x0 = to_vector(CliqueVector.zeros(domain, original_cliques))
    ans = optimize.minimize(loss_and_grad, x0=x0, method='L-BFGS-B', jac=True)
    
    model.total = total
    model.potentials = expand(to_cliquevector(ans['x']))
    model.marginals = model.belief_propagation(model.potentials)
    return model


def learn_model_approx(domain, total, marginals):
    cliques = list(marginals.keys())
    model = FactorGraph(domain, cliques, total)
    model.marginals = CliqueVector(marginals)
    for cl in cliques:
        for v in cl:
            model.beliefs[v] = marginals[cl].project([v]).log()
    return model

def mwem(workload, data, learn_model=learn_model_exact, out=None):
    workload_copy = [proj for proj in workload]
    t0 = time.time()
    total = data.records

    if out: f = open(out, 'w')
    #engine = FactoredMultiplicativeWeights(data.domain)
    #engine = FactoredInference(data.domain, log=False, iters=1000)
    print('checkpt1')
    marginals = {}
    est = learn_model(data.domain, total, marginals)
    if out:
        err = average_error(workload_copy, data, est)
        s = '%s,None,0,%.12f \n' % (0, err)
        f.write(s)
        
    for i in range(1,len(workload)+1):
        ax = worst_approximated(data, est, workload)
        print(ax)
        mu = data.project(ax)
        marginals[ax] = Factor(mu.domain, mu.datavector())
        est = learn_model(data.domain, total, marginals)
        workload.remove(ax)
        size = sum(data.domain.size(cl) for cl in est.cliques)
        print('%.3f MB' % (8*size/2**20))
        err = average_error(workload_copy, data, est)
        if out:
            err = average_error(workload_copy, data, est)
            s = '%s,%s,%.3f,%.12f \n' % (i, ' x '.join(ax), 8*size/2**20, err)
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
    params['engine'] = 'exact'
    params['save'] = None

    return params 
 

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--engine', choices=['exact', 'approx'], help='inference engine')
    parser.add_argument('--save', action='store_true', help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    #data, workloads = benchmarks.random3way(args.dataset, args.workload)
   
    prefix = os.getenv('HD_DATA') #'/home/rmckenna/Repos/hd-datasets/clean/'
    attrs = ['ALS Unit', 'Battalion', 'Call Final Disposition', 'Call Type', 'Call Type Group', 'City', 'Final Priority', 'Fire Prevention District', 'Neighborhooods - Analysis Boundaries', 'Original Priority', 'Priority', 'Station Area', 'Supervisor District', 'Unit Type', 'Zipcode of Incident']

    name = args.dataset

    data = Dataset.load(prefix+name+'.csv', prefix+name+'-domain.json')

    if name == 'fire':
        data = data.project(attrs)

    workload = list(itertools.combinations(data.domain, 2))
    if args.engine == 'exact':
        learn_model = learn_model_exact
    else:
        learn_model = learn_model_approx
 
    if args.save:
        out = 'results/mwem-no-noise-%s_%s.out' % (args.dataset, args.engine)
    else:
        out = None

    ans = mwem(workload, data, learn_model, out)
