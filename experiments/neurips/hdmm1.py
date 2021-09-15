from mbi import mechanism, FactoredInference
import benchmarks
from IPython import embed
import numpy as np
from scipy.sparse.linalg import lsmr
import argparse


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'adult'
    params['workload'] = 15
    params['iters'] = 10000
    params['epsilon'] = 1.0
    params['seed'] = 0
    params['save'] = None

    return params

if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['adult','titanic','msnbc','loans','nltcs','fire','stroke','salary'], help='dataset to use')
    parser.add_argument('--workload', type=int, help='number of marginals in workload')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--save', action='store_true', help='save results')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data, measurements, workloads = benchmarks.random_hdmm(args.dataset, args.workload)

    model, log, answers = mechanism.run(data, measurements, eps=args.epsilon, frequency=50, seed=args.seed, iters=args.iters, oracle='approx', engine='MD2')

    local_ls = {}
    for Q, y, _, proj in answers:
        local_ls[proj] = lsmr(Q, y)[0]

    fake_measurements = []
    for proj, W in workloads:
        y = W.dot(local_ls[proj])
        fake_measurements.append((W, y, 1.0, proj))

    engine = FactoredInference(data.domain, metric='L2', iters=args.iters)
    from mbi import callbacks
    callback = callbacks.Logger(engine)
    model2 = engine.infer(fake_measurements, data.df.shape[0])

    ls = []
    mb = []
    mb2 = []

    for proj, W in measurements:
        est = W.dot(model.project(proj).datavector())
        est2 = W.dot(local_ls[proj])
        est3 = W.dot(model2.project(proj).datavector())
        true = W.dot(data.project(proj).datavector())
        err = np.abs(est - true).sum() / np.abs(true).sum()
        err2 = np.abs(est2 - true).sum() / np.abs(true).sum()
        err3 = np.abs(est3 - true).sum() / np.abs(true).sum()
        mb.append(err)
        ls.append(err2)
        mb2.append(err3)

    err_hdmm1 = np.mean(ls)
    err_pgm1 = np.mean(mb)
    err_pgm1a = np.mean(mb2)
        
    ls = []
    mb = []
    mb2 = []

    for proj, W in workloads:
        est = W.dot(model.project(proj).datavector())
        est2 = W.dot(local_ls[proj])
        est3 = W.dot(model2.project(proj).datavector())
        true = W.dot(data.project(proj).datavector())
        err = np.abs(est - true).sum() / np.abs(true).sum()
        err2 = np.abs(est2 - true).sum() / np.abs(true).sum()
        err3 = np.abs(est3 - true).sum() / np.abs(true).sum()
        print(proj, err, err2, err3)
        mb.append(err)
        ls.append(err2)
        mb2.append(err3)

    err_hdmm2 = np.mean(ls)
    err_pgm2 = np.mean(mb)
    err_pgm2a = np.mean(mb2)
 
    path = 'results/hdmm.csv'

    with open(path, 'a') as f:
        f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s \n' % (args.dataset,args.seed,args.epsilon,err_hdmm1, err_pgm1, err_hdmm2, err_pgm2, err_pgm1a, err_pgm2a))

