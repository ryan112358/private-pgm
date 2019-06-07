from mbi import Dataset, FactoredInference, Domain
import numpy as np

# discrete domain with attributes A, B, C and corresponding size 4 x 5 x 6 
domain = Domain(['A','B','C'], [2, 3, 4])

# synthetic dataset with 1000 rows
data = Dataset.synthetic(domain, 1000) 

# project data onto subset of cols, and vectorize
ab = data.project(['A','B']).datavector()
bc = data.project(['B','C']).datavector()

# add noise to preserve differential privacy
epsilon = np.sqrt(2)
sigma = np.sqrt(2.0) / epsilon

np.random.seed(0)
yab = ab + np.random.laplace(loc=0, scale=sigma, size=ab.size)
ybc = bc + np.random.laplace(loc=0, scale=sigma, size=bc.size)

# record the measurements in a form needed by inference
Iab = np.eye(ab.size)
Ibc = np.eye(bc.size)

measurements = [(Iab, yab, sigma, ['A', 'B']),
                (Ibc, ybc, sigma, ['B', 'C'])]

# estimate the data distribution
engine = FactoredInference(domain)
model = engine.estimate(measurements, engine='MD')

# recover consistent estimates of measurements
ab2 = model.project(['A','B']).datavector()
bc2 = model.project(['B','C']).datavector()

print(ab2)

print(bc2)

# estimate answer to unmeasured queries
ac2 = model.project(['A','C']).datavector()
print(ac2)

# generate synthetic data
synth = model.synthetic_data(rows=10)
print(synth.df)
