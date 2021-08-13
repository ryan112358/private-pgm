import numpy as np
import itertools
from mbi import Dataset, GraphicalModel, FactoredInference
from scipy.special import softmax
from scipy import sparse

"""
This file contains an implementation of MWEM+PGM that is designed specifically for marginal query workloads.
Unlike mwem.py, which selects a single query in each round, this implementation selects an entire marginal 
in each step.  It leverages parallel composition to answer many more queries using the same privacy budget.

This enhancement of MWEM was described in the original paper in section 3.3 (https://arxiv.org/pdf/1012.4763.pdf).

There are two additional improvements not described in the original Private-PGM paper:
- In each round we only consider candidate cliques to select if they result in sufficiently small model sizes
- At the end of the mechanism, we generate synthetic data (rather than query answers)
"""

def worst_approximated(workload_answers, est, workload, eps, penalty=True):
    """ Select a (noisy) worst-approximated marginal for measurement.
    
    :param workload_answers: a dictionary of true answers to the workload
        keys are cliques
        values are numpy arrays, corresponding to the counts in the marginal
    :param est: a GraphicalModel object that approximates the data distribution
    :param: workload: The list of candidates to consider in the exponential mechanism
    :param eps: the privacy budget to use for this step.
    """
    errors = np.array([])
    for cl in workload:
        bias = est.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = est.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xest).sum()-bias)
    prob = softmax(0.5*eps*(errors - errors.max()))
    key = np.random.choice(len(errors), p=prob)
    return workload[key]

def mwem_pgm(data, epsilon, delta=0.0, workload=None, rounds=None, maxsize_mb = 25, pgm_iters=100):
    """
    Implementation of MWEM + PGM

    :param data: an mbi.Dataset object
    :param epsilon: privacy budget
    :param delta: privacy parameter (ignored)
    :param workload: A list of cliques (attribute tuples) to include in the workload (default: all pairs of attributes)
    :param rounds: The number of rounds of MWEM to run (default: number of attributes)
    :param maxsize_mb: [New] a limit on the size of the model (in megabytes), used to filter out candidate cliques from selection.
        Used to avoid MWEM+PGM failure models (intractable model sizes).   

    Implementation Notes:
    - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does
        not increase size of the graphical model too much
    """ 
    if workload is None:
        workload = list(itertools.combinations(data.domain, 2))
    if rounds is None:
        rounds = len(data.domain)

    domain = data.domain
    total = data.records

    def size(cliques):
        return GraphicalModel(domain, cliques).size * 8 / 2**20

    eps1 = epsilon / (2.0 * rounds)

    workload_answers = { cl : data.project(cl).datavector() for cl in workload }

    engine = FactoredInference(data.domain, log=False, iters=pgm_iters, warm_start=True)
    measurements = []
    est = engine.estimate(measurements, total)
    cliques = []
    for i in range(1, rounds+1):
        # [New] Only consider candidates that keep the model sufficiently small
        candidates = [cl for cl in workload if size(cliques+[cl]) <= maxsize_mb*i/rounds]
        if len(candidates) == 0:
            break # Terminate early and forfeit remaining privacy budget
        ax = worst_approximated(workload_answers, est, candidates, eps1)
        print('Round', i, 'Selected', ax, 'Model Size (MB)', est.size*8/2**20)
        n = domain.size(ax)
        x = data.project(ax).datavector()
        y = x + np.random.laplace(loc=0, scale=2.0 / eps1, size=n)
        Q = sparse.eye(n)
        measurements.append((Q, y, 1.0, ax))
        est = engine.estimate(measurements, total)
        workload.remove(ax)
        cliques.append(ax)

    print('Generating Data...')
    return est.synthetic_data()

if __name__ == '__main__':
    data = Dataset.load('../data/adult.csv', '../data/adult-domain.json')
    synth = mwem_pgm(data, 1.0)

    # measure error (total variation distance) on 3-way marginals
    errors = []
    for proj in itertools.combinations(data.domain.attrs, 2):
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))
