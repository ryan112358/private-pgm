from mbi import Dataset, FactoredInference, Domain
import numpy as np
from scipy import sparse

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
    I = sparse.eye(x.size)
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
    I = sparse.eye(x.size)
    measurements.append( (I, y, sigma, cl) )

# now perform inference to estimate the data distribution

engine = FactoredInference(domain, log=True, iters=10000)
model = engine.estimate(measurements, total=total, engine='RDA')

# now answer new queries

y1 = model.project(('sex', 'income>50K')).datavector()
y2 = model.project(('race', 'occupation')).datavector()
