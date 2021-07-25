from mbi import mechanism, FactoredInference
import benchmarks
import numpy as np
from scipy.sparse.linalg import lsmr
import argparse
from ektelo.matrix import Identity, EkteloMatrix, Kronecker

"""
Run HDMM with and without graphical-model inference on the adult dataset.

McKenna, Ryan, Gerome Miklau, Michael Hay, and Ashwin Machanavajjhala. "Optimizing error of high-dimensional statistical queries under differential privacy." Proceedings of the VLDB Endowment 11, no. 10 (2018): 1206-1219.

Note that least squares is the bottleneck of HDMM for this dataset, as the vector representation is far too large to fit in memory, preventing the original algorithm to run in this setting.  We thus compare against a simple substitute instead, HDMM+LLS (Local Least Squares).

Also note that this is not a fully general implementation of HDMM.  This file is designed to highlight the benefits of Private-PGM for a particular choice of measurements chosen by HDMM.  In this case, the measurements were obtained by running HDMM with OPT+ for a specific workload of 15 random Prefix-Marginals.  These measurements were hard-coded in this script.  For different workloads (e.g., plain marginals) this script should not be used.  Please refer to https://github.com/dpcomp-org/hdmm, which
 contains a fully general implementation of HDMM. 
"""


def get_measurements(domain, workload):
    # get measurements using OPT+ parameterization
    lookup = {}
    # optimal strategy for Identity is Identity
    for attr in domain:
        n = domain.size(attr)
        lookup[attr] = Identity(n)
    # optimal strategy for Prefix is precomputed and loaded
    lookup['age'] = EkteloMatrix(np.load('prefix-85.npy'))
    lookup['fnlwgt'] = EkteloMatrix(np.load('prefix-100.npy'))
    lookup['capital-gain'] = EkteloMatrix(np.load('prefix-100.npy'))
    lookup['capital-loss'] = EkteloMatrix(np.load('prefix-100.npy'))
    lookup['hours-per-week'] = EkteloMatrix(np.load('prefix-99.npy'))

    measurements = []
    for proj, _ in workload:
        Q = Kronecker([lookup[a] for a in proj])
        measurements.append( (proj, Q.sparse_matrix()) )

    return measurements

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

    data, workload = benchmarks.adult_benchmark()

    measurements = get_measurements(data.domain, workload) 

    model, log, answers = mechanism.run(data, measurements, eps=args.epsilon, frequency=50, seed=args.seed, iters=args.iters)

    local_ls = {}
    for Q, y, _, proj in answers:
        local_ls[proj] = lsmr(Q, y)[0]

    ls = []
    mb = []

    for proj, W in workload:
        est = W.dot(model.project(proj).datavector())
        est2 = W.dot(local_ls[proj])
        true = W.dot(data.project(proj).datavector())
        err = np.abs(est - true).sum() / np.abs(true).sum()
        err2 = np.abs(est2 - true).sum() / np.abs(true).sum()
        mb.append(err)
        ls.append(err2)

    print('Error of HDMM    : %.3f' % np.mean(ls))
    print('Error of HDMM+PGM: %.3f' % np.mean(mb))
