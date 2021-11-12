from mbi import Dataset, FactoredInference, Domain, LocalInference, MixtureInference, PublicInference
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
epsilon_split = epsilon / (len(data.domain) + len(cliques))
sigma = 2.0 / epsilon_split

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

for cl in cliques:
    x = data.project(cl).datavector()
    y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
    I = sparse.eye(x.size)
    measurements.append( (I, y, sigma, cl) )

# now perform inference to estimate the data distribution
# We can either use Private-PGM (FactoredInference) or 
# Approx-Private-PGM (LocalInference), both share the same interface.

engine = FactoredInference(domain, log=True, iters=2500)
#engine = LocalInference(domain, log=True, iters=2500, marginal_oracle='convex')

model = engine.estimate(measurements, total=total)

# now answer new queries

y1 = model.project(('sex', 'income>50K')).datavector()
y2 = model.project(('race', 'occupation')).datavector()
