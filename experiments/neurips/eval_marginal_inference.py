from mbi import Dataset, FactoredInference, Factor, GraphicalModel, Domain, CliqueVector, RegionGraph, FactorGraph
import numpy as np
import itertools
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt

def benchmark1(hard=True):
    domain = Domain(['A', 'B', 'C'], [2,3,4])
    cliques = [('A','B'), ('B','C'), ('A','C')]
    model = RegionGraph(domain, cliques, iters=10000, convex=True, minimal=True)
    if hard:
        model.counting_numbers = {('B',): 1.0, 
                               ('C',): 0.01, 
                               ('A',): 0.01, 
                              ('A', 'B'): 1.0, 
                              ('A', 'C'): 0.01, 
                              ('B', 'C'): 1.0}
    model.potentials = CliqueVector.random(domain, model.cliques)*3
    return model

def benchmark2(hard=False):
    domain = Domain(['A', 'B', 'C','D', 'E','F','G','H','I','J'], [2,3,2,3,2,3,2,3,2,3])
    cliques = list(itertools.combinations(domain.attrs, 2))
    model = RegionGraph(domain, cliques, iters=10000, convex=True, minimal=True)
    if hard:
        model.counting_numbers = { r : 10**(-2*np.random.rand()) for r in model.regions } 
    model.potentials = CliqueVector.random(domain, model.cliques)*3
    return model

if __name__ == '__main__':
    #model = benchmark1()
    hard = True
    model = benchmark2(hard)
       
    potentials = model.potentials 
    mu1 = model.optimize_kikuchi(potentials)
    log2, log3, log4 = [], [], []
    
    def cb(mu, log):
        diff = mu1 - CliqueVector(mu)
        log.append(np.sqrt(diff.dot(diff)))

    marginals2 = model.hazan_peng_shashua(potentials, callback=lambda mu: cb(mu, log2))
    marginals3 = model.wiegerinck(potentials, callback=lambda mu: cb(mu, log3))
    marginals4 = model.loh_wibisono(potentials, callback=lambda mu: cb(mu, log4))

    iters = np.arange(model.iters)

    plt.plot(iters, log2, label='Hazan, Peng, Shashua')
    plt.plot(iters, log3, label='Wiegerinck')
    plt.plot(iters, log4, label='Loh, Wibisono')
    #plt.yscale('log')
    plt.loglog()
    plt.xlabel('Iteration')
    plt.ylabel('Distance to True Solution')
    plt.legend()
    if hard: 
        plt.title('Hard Counting Numbers')
    else:
        plt.title('Easy Counting Numbers')


    plt.show()


