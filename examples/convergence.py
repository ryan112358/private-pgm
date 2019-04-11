from mbi import Dataset, FactoredInference
from mbi import mechanism
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy import sparse

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'adult'
    params['engines'] = ['MD','RDA']
    params['iters'] = 10000
    params['epsilon'] = 1.0
    params['delta'] = 0.0
    params['bounded'] = True
    params['frequency'] = 1
    params['seed'] = 0
    params['save'] = None
    params['load'] = None
    params['plot'] = None

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['adult'], help='dataset to use')
    parser.add_argument('--engines', nargs='+', choices=['MD','MD2','RDA','LBFGS','EM','IG'], help='inference engines')
    parser.add_argument('--iters', type=int, help='number of iterations')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--bounded', type=bool, help='bounded or unbounded privacy definition')
    parser.add_argument('--frequency', type=int, help='logging frequency')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', type=str, help='path to save results')
    parser.add_argument('--load', type=str, help='path to load results from (skips experiment)')
    parser.add_argument('--plot', type=str, help='path to save plot')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    if args.load:
        results = pickle.load(open(args.load,'rb'))
    else:
        data = Dataset.load('../data/adult.csv', '../data/adult-domain.json')
        projections = [['race', 'capital-loss', 'income>50K'],
               ['marital-status', 'capital-gain', 'income>50K'],
               ['race', 'native-country','income>50K'],
               ['workclass', 'sex','hours-per-week'],
               ['fnlwgt','marital-status', 'relationship'],
               ['workclass','education-num','occupation'],
               ['age','relationship','sex'],
               ['occupation','sex','hours-per-week'],
               ['occupation','relationship','income>50K']]

        measurements = []
        for p in projections:
            Q = sparse.eye(data.domain.size(p))
            measurements.append( (p, Q) )
       
        aux = { 'iters' : args.iters, 'eps' : args.epsilon, 'delta' : args.delta,
                'bounded' : args.bounded, 'frequency' : args.frequency, 'seed' : args.seed }
        results = {}
        for engine in args.engines: 
            results[engine] = mechanism.run(data, measurements, engine=engine, **aux)[1].results

    if args.save:
        pickle.dump(results, open(args.save, 'wb'))

    if args.plot:
        best = min(results[engine].l2_loss.min() for engine in args.engines)
        for engine in args.engines:
            df = results[engine]
            plt.plot(df.time, df.l2_loss, label=engine)
        plt.loglog()
        plt.xlabel('Time (seconds)')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.plot) 
