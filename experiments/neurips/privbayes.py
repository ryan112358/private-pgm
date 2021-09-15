from ektelo.algorithm.privBayes import privBayesSelect
import numpy as np
from mbi import Dataset, Factor, FactoredInference, mechanism
from ektelo.matrix import Identity
import pandas as pd
import itertools
import argparse
import benchmarks

def privbayes_measurements(data, eps=1.0, seed=0):
    domain = data.domain
    config = ''
    for a in domain:
        values = [str(i) for i in range(domain[a])]
        config += 'D ' + ' '.join(values) + ' \n'
    config = config.encode('utf-8')
    
    values = np.ascontiguousarray(data.df.values.astype(np.int32))
    ans = privBayesSelect.py_get_model(values, config, eps/2, 1.0, seed)
    ans = ans.decode('utf-8')[:-1]
    
    projections = []
    for m in ans.split('\n'):
        p = [domain.attrs[int(a)] for a in m.split(',')[::2]]
        projections.append(tuple(p))
  
    prng = np.random.RandomState(seed) 
    measurements = []
    delta = len(projections)
    for proj in projections:
        x = data.project(proj).datavector()
        I = Identity(x.size)
        y = I.dot(x) + prng.laplace(loc=0, scale=4*delta/eps, size=x.size)
        measurements.append( (I, y, 1.0, proj) )
     
    return measurements

def privbayes_inference(domain, measurements, total):
    synthetic = pd.DataFrame()

    _, y, _, proj = measurements[0]
    y = np.maximum(y, 0)
    y /= y.sum()
    col = proj[0]
    synthetic[col] = np.random.choice(domain[col], total, True, y)
        
    for _, y, _, proj in measurements[1:]:
        # find the CPT
        col, dep = proj[0], proj[1:]
        print(col)
        y = np.maximum(y, 0)
        dom = domain.project(proj)
        cpt = Factor(dom, y.reshape(dom.shape))
        marg = cpt.project(dep)
        cpt /= marg
        cpt2 = np.moveaxis(cpt.project(proj).values, 0, -1)
        
        # sample current column
        synthetic[col] = 0
        rng = itertools.product(*[range(domain[a]) for a in dep])
        for v in rng:
            idx = (synthetic.loc[:,dep].values == np.array(v)).all(axis=1)
            p = cpt2[v].flatten()
            if p.sum() == 0:
                p = np.ones(p.size) / p.size
            n = domain[col]
            N = idx.sum()
            if N > 0:
                synthetic.loc[idx,col] = np.random.choice(n, N, True, p)

    return Dataset(synthetic, domain)

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

    data, _, workload = benchmarks.random_hdmm(args.dataset, args.workload)
    total = data.df.shape[0]

    measurements = privbayes_measurements(data, 1.0, args.seed) 

    est = privbayes_inference(data.domain, measurements, total=total)

    elim_order = [m[3][0] for m in measurements][::-1]
    #elim_order = None

    projections = [m[3] for m in measurements]
    est2, _, _ = mechanism.run(data, projections, eps=args.epsilon, frequency=50, seed=args.seed, iters=args.iters)
    model = RegionGraph(data.domain, projections + [w[0] for w in workload], iters=1, minimal=True)
    est3, _, _ = mechanism.run(data, projections, eps=args.epsilon, frequency=50, seed=args.seed, iters=args.iters, oracle=model, engine='MD2')

    #engine=FactoredInference(data.domain, iters=args.iters, warm_start=True, elim_order=elim_order)
    #engine.infer(measurements, total)
    #est2 = engine.model

    def err(true, est):
        #return np.linalg.norm(true-est,1)
        return np.sum(np.abs(true - est)) / true.sum()

    err_pb = []
    err_pgm = []
    err_approx = []
    for Q, y, _, p in measurements:
        true = data.project(p).datavector()
        pb = est.project(p).datavector()
        pgm = est2.project(p).datavector()
        approx = est3.project(p).datavector()
        d = true.mean()
        #print('%s, %.4f, %.4f' % (' x '.join(p), err(true, pb), err(true, pgm)))
        err_pb.append(err(true, pb))
        err_pgm.append(err(true, pgm))
        err_approx.append(err(true, approx))

    err_pb1 = np.mean(err_pb)
    err_pgm1 = np.mean(err_pgm)
    err_approx1 = np.mean(err_approx)
    print('Error on measurements', err_pb1, err_pgm1, err_approx1)

    err_pb = []
    err_pgm = []
    err_approx = []
    for p, W in workload:
        true = W.dot(data.project(p).datavector())
        pb = W.dot(est.project(p).datavector())
        pgm = W.dot(est2.project(p).datavector())
        approx = W.dot(est3.project(p).datavector())
        err_pb.append(err(true, pb))
        err_pgm.append(err(true, pgm))
        err_approx.append(err(true, approx))

    err_pb2 = np.mean(err_pb)
    err_pgm2 = np.mean(err_pgm)
    err_approx2 = np.mean(err_approx)
    print('Error on workload', err_pb2, err_pgm2, err_approx2)

    path = 'results/privbayes.csv'

    with open(path, 'a') as f:
        f.write('%s,%s,%s,%s,%s,%s,%s,%s \n' % (args.dataset,args.seed,err_pb1, err_pgm1, err_approx1, err_pb2, err_pgm2, err_approx2))
        

