from mbi import Dataset, FactoredInference, callbacks, Factor, CliqueVector, Domain, GraphicalModel
from mbi.optimization import Optimizer
import numpy as np
from IPython import embed
import argparse
import itertools
from scipy import sparse
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from gum.gum import transform

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['attributes'] = 3
    params['temperature'] = 3
    params['noise'] = list(np.logspace(-2,1,7))
    params['records'] = 1000
    params['trials'] = 1
    params['seed'] = 0
    params['save'] = None

    return params

if __name__ == '__main__':

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--attributes', type=int, help='number of attributes')
    parser.add_argument('--temperature', type=float, help='strength of potentials')
    parser.add_argument('--noise', type=float, nargs='+', help='noise magnitude')
    parser.add_argument('--records', type=int, help='number of data points')
    parser.add_argument('--trials', type=int, help='number of trials')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', action='store_true', help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    attrs = list(map(str, range(args.attributes)))
    sizes = [2+i for i in range(args.attributes)]  
    domain = Domain(attrs, sizes)
    cliques = list(zip(attrs, attrs[1:]))
    potentials = CliqueVector.random(domain, cliques)*args.temperature 

    
    model = GraphicalModel(domain, cliques)
    model.potentials = potentials
    model.marginals = model.belief_propagation(potentials)

    N = args.records
    data = model.synthetic_data(N)
    def error(mu):
        ans = 0
        for cl in cliques:
            x = data.project(cl).datavector()/N
            y = mu[cl].datavector()
            ans += np.linalg.norm(x-y,1)    
        return ans

    logs = defaultdict(lambda: 0)

    noisy_marginals = {}

    for noise in args.noise:
        for _ in range(args.trials):
            measurements = []
            for cl in cliques:
                n = domain.size(cl)
                I = sparse.eye(n)
                x = data.project(cl).datavector()/N
                y = x + np.random.normal(loc=0, scale=noise, size=x.size)
                #y = x + np.random.laplace(loc=0, scale=noise, size=x.size)
                measurements.append( (I, y, 1.0, cl) )
                noisy_marginals[cl] = Factor(data.domain.project(cl), y)

            for consistency in [True,False]:
                for simplex in [True,False]:    
                    opt = Optimizer(domain, cliques, total=1.0, consistency=consistency, simplex=simplex)
                    mu = opt.estimate(measurements, total=1.0, backend='scipy')
                    #print(consistency, simplex, error(mu))
                    logs[consistency,simplex,noise] += error(mu)

            mu = transform(noisy_marginals, data.domain, consistent_iter=1000)
            logs['gum',noise] += error(mu)

    results = []
    xs = args.noise
    for consistency in [True,False]:
        for simplex in [True,False]:    
            ys = [logs[consistency, simplex, noise]/args.trials for noise in args.noise]
            results.append(ys)
            plt.plot(xs, ys, '.', label='%s, %s' % (consistency, simplex))
    ys = [logs['gum',noise]/args.trials for noise in args.noise]
    results.append(ys)
    plt.plot(xs, ys, '.', label='GUM')

    plt.xlabel('Noise Magnitude', fontsize='x-large')
    plt.ylabel('Distance to True Marginals')
    #plt.xscale('log')
    plt.loglog()
    plt.legend()
    plt.savefig('fig1.png')

    columns = ['11', '10', '01', '00', 'GUM']
    results = np.array(results).reshape(5,-1).T
    results = pd.DataFrame(index=args.noise, data=results, columns=columns)
    print(results)



