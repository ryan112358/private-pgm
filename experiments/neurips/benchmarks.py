import numpy as np
from mbi import Dataset
from hdmm import matrix, workload
import itertools
from scipy import sparse
from functools import reduce

prefix = '/home/rmckenna/Repos/hd-datasets/clean/'

def load(name):
    return Dataset.load(prefix + name + '.csv', prefix + name + '-domain.json')

def adult_benchmark():
    cols = ['age',
             'workclass',
             'fnlwgt',
             #'education',
             'education-num',
             'marital-status',
             'occupation',
             'relationship',
             'race',
             'sex',
             'capital-gain',
             'capital-loss',
             'hours-per-week',
             'native-country',
             'income>50K']

    data = load('adult').project(cols)
    domain = data.domain

    workloads = {}
    for a,b,c in itertools.combinations(domain.attrs, 3):
        n = domain[a]*domain[b]*domain[c]
        workloads[(a,b,c)] = matrix.Identity(n)

#    workloads = {}
#    for a,b in itertools.combinations(domain.attrs, 2):
#        n = domain[a]*domain[b]
#        workloads[(a,b)] = matrix.Identity(n)
 
       
    return data, workloads 

def adult_toy_tree():
    cols = ['race', 'native-country', 'marital-status']
    projections = [('race', 'native-country'),
                    ('marital-status', 'native-country')]

    data = load('adult').project(cols)
    return data, projections

def adult_toy_cycle():
    cols = ['race', 'native-country', 'marital-status']
    projections = [('race', 'native-country'),
                    ('marital-status', 'native-country'),
                    ('race', 'marital-status')]

    data = load('adult').project(cols)
    return data, projections

   


def adult2():
    data = load('adult').drop('education')
    projections = [['race', 'capital-loss', 'income>50K'],
               ['marital-status', 'capital-gain', 'income>50K'],
               ['race', 'native-country','income>50K'],
               ['workclass', 'sex','hours-per-week'],
               ['fnlwgt','marital-status', 'relationship'],
               ['workclass','education-num','occupation'],
               ['age','relationship','sex'],
               ['occupation','sex','hours-per-week'],
               ['occupation','relationship','income>50K']]
    projections = [tuple(x) for x in projections]
    return data, projections 

def adult_hdmm():
    data = load('adult').drop(['education-num'])
    projections  = [('workclass', 'education'),
                    ('education', 'income>50K'),
                    ('marital-status', 'relationship'),
                    ('age', 'marital-status'),
                    ('age', 'relationship'),
                    ('marital-status', 'sex', 'hours-per-week'),
                    ('race','native-country'),
                    ('capital-gain','workclass'),
                    ('capital-loss', 'workclass'),
                    ('native-country', 'marital-status'),
                    ('fnlwgt', 'education')]

    A85 = sparse.csr_matrix(np.load('prefix-85.npy'))
    A100 = sparse.csr_matrix(np.load('prefix-100.npy'))

    lookup = {}
    lookup_W = {}
    for attr in data.domain:
        lookup[attr] = sparse.eye(data.domain.size(attr), format='csr')
        lookup_W[attr] = matrix.Identity(data.domain.size(attr))

    lookup['age'] = A85
    lookup['capital-gain'] = A100
    lookup['capital-loss'] = A100
    lookup['fnlwgt'] = A100

    lookup_W['age'] = workload.Prefix(85)
    lookup_W['capital-gain'] = workload.Prefix(100)
    lookup_W['capital-loss'] = workload.Prefix(100)
    lookup_W['fnlwgt'] = workload.Prefix(100)

    measurements = []
    workloads = []
    for attr in data.domain:
        proj = (attr,)
        measurements.append( (proj, lookup[attr]) )
        workloads.append( (proj, lookup_W[attr]) )

    for proj in projections:
        Q = reduce(sparse.kron, [lookup[a] for a in proj]).tocsr()
        measurements.append( (proj, Q) )
        W = matrix.Kronecker([lookup_W[a] for a in proj])
        workloads.append( (proj, W) )
                   
    return data, measurements, workloads

def random_hdmm(name, number, seed=0):
    data, projections = random3way(name, number, seed)
    lookup = {}
    lookup_W = {}
    
    A100 = sparse.csr_matrix(np.load('prefix-100.npy'))
    A101 = sparse.csr_matrix(np.load('prefix-100-missing.npy'))
    P100 = workload.Prefix(100)
    P101 = workload.Prefix(101)

    for attr in data.domain:
        n = data.domain.size(attr)
        if n == 100:
            lookup[attr] = A100
            lookup_W[attr] = P100
        elif n == 101:
            lookup[attr] = A101
            lookup_W[attr] = P101
        else:
            lookup[attr] = sparse.eye(n, format='csr')
            lookup_W[attr] = matrix.Identity(n)

    measurements = []
    workloads = []

    for proj in projections:
        Q = reduce(sparse.kron, [lookup[a] for a in proj]).tocsr()
        measurements.append( (proj, Q) )
        W = matrix.Kronecker([lookup_W[a] for a in proj])
        workloads.append( (proj, W) )

    return data, measurements, workloads

def all3way(name):
    data = load(name)
    proj = list(itertools.combinations(data.domain.attrs, 3))
    return data, proj

def small3way(name, number):
    data = load(name)
    dom = data.domain
    proj = sorted(itertools.combinations(data.domain.attrs, 3), key=dom.size)[:number]
    return data, proj

def random3way(name, number, seed=0):
    prng = np.random.RandomState(seed)
    data = load(name)
    total = data.df.shape[0]
    dom = data.domain
    proj = [p for p in itertools.combinations(data.domain.attrs, 3) if dom.size(p) <= total]
    if len(proj) > number:
        proj = [proj[i] for i in prng.choice(len(proj), number, replace=False)]
    return data, proj

def msnbc1():
    data = load('msnbc')
    curr = ('site_0', 'site_1', 'site_2')
    measurements = [curr]

    for i in range(3, len(data.domain)):
        curr = (curr[1], curr[2], 'site_%d'%i)
        measurements.append(curr)

    return data, measurements

def titanic1():
    data = load('titanic')
    projections = [('Survived', other) for other in data.domain if other != 'Survived']
    return data, projections

def random(name, number, seed=0):
    state = np.random.RandomState(seed)
    data = load(name)
    choices = []
    for i in [2,3,4]:
        choices += list(itertools.combinations(data.domain.attrs, i))
    probas = np.array([1.0/data.domain.project(proj).size() for proj in choices])
    probas /= probas.sum()

    number = min(number, len(choices))
    idx = state.choice(len(choices), number, False, probas)
    projections = [(col,) for col in data.domain.attrs] + [choices[i] for i in idx]
    #projections = [choices[i] for i in idx]
    return data, projections

