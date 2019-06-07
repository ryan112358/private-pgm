from mbi import Dataset
from ektelo import workload

def adult_benchmark():
    data = Dataset.load('../data/adult.csv', '../data/adult-domain.json')

    projections = [('occupation', 'race', 'capital-loss'),
                     ('occupation', 'sex', 'native-country'),
                     ('marital-status', 'relationship', 'income>50K'),
                     ('age', 'education-num', 'sex'),
                     ('workclass', 'education-num', 'occupation'),
                     ('marital-status', 'occupation', 'income>50K'),
                     ('race', 'native-country', 'income>50K'),
                     ('occupation', 'capital-gain', 'income>50K'),
                     ('marital-status', 'hours-per-week', 'income>50K'),
                     ('workclass', 'race', 'capital-gain'),
                     ('marital-status', 'relationship', 'capital-gain'),
                     ('workclass', 'education-num', 'capital-gain'),
                     ('education-num', 'relationship', 'race'),
                     ('fnlwgt', 'hours-per-week', 'income>50K'),
                     ('workclass', 'sex', 'native-country')]

    lookup = {}
    for attr in data.domain:
        n = data.domain.size(attr)
        lookup[attr] = workload.Identity(n)

    lookup['age'] = workload.Prefix(85)
    lookup['fnlwgt'] = workload.Prefix(100)
    lookup['capital-gain'] = workload.Prefix(100)
    lookup['capital-loss'] = workload.Prefix(100)
    lookup['hours-per-week'] = workload.Prefix(99)
    
    workloads = []

    for proj in projections:
        W = workload.Kronecker([lookup[a] for a in proj])
        workloads.append( (proj, W) )

    return data, workloads
