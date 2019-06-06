from mbi import Dataset, FactoredInference, Domain
import numpy as np
from scipy import sparse

"""
This file is essentially the same as adult_example with two key differences:

    (1) it is run with the torch backend
    (2) scipy.sparse.eye(n) is replaced with Identity(n), which improves speed by a factor of 2

Note that for this example, running on a GPU does not speed up inference, likely because
the total clique size is small (6516).  With more complicated measurements/models torch may help.
"""

class Identity(sparse.linalg.LinearOperator):
    def __init__(self, n):
        self.shape = (n,n)
        self.dtype = np.float64
    def _matmat(self, X):
        return X
    def __matmul__(self, X):
        return X
    def _transpose(self):
        return self
    def _adjoint(self):
        return self

# load adult dataset

data = Dataset.load('../data/adult.csv', '../data/adult-domain.json')
domain = data.domain
total = data.df.shape[0]

print(domain)

# spend half of privacy budget to measure all 1 way marginals
np.random.seed(0)

epsilon = 1.0
sigma = 1.0 / len(data.domain) / 2.0

measurements = []
for col in data.domain:
    x = data.project(col).datavector()
    y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
    I = Identity(x.size)
    measurements.append( (I, y, sigma, (col,)) )

# spend half of privacy budget to measure some more 2 and 3 way marginals

cliques = [('age', 'education-num'), 
            ('marital-status', 'race'), 
            ('sex', 'hours-per-week'),
            ('hours-per-week', 'income>50K'),
            ('native-country', 'marital-status', 'occupation')]

sigma = 1.0 / len(cliques) / 2.0

for cl in cliques:
    x = data.project(cl).datavector()
    y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
    I = Identity(x.size)
    measurements.append( (I, y, sigma, cl) )

# now perform inference to estimate the data distribution

engine = FactoredInference(domain, backend='torch', log=True, iters=10000)
model = engine.estimate(measurements, total=total, engine='RDA')

# now answer new queries

y1 = model.project(('sex', 'income>50K')).datavector()
y2 = model.project(('race', 'occupation')).datavector()
