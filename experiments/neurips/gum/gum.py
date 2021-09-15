from gum.consistent import Consistenter
from gum.non_negativity import NonNegativity
from gum.view import View
from mbi import *
from collections import defaultdict
import numpy as np
import copy

def calcOneHot(key, domain):
    indicator = [0 for _ in range(len(domain))]
    
    for k in key:
        for d in range(len(domain)):
            if domain[d] == k:
                indicator[d] = 1
                
    return np.array(indicator, dtype=np.uint8)

def transform(factor_dict, domain, consistent_iter=100):
    """
    input params:
    factor_dict: Dictionary consisting of key, value pairs of the form: (k=factor domain as a tuple, v=factor object associated)
    full_domain: Full domain associated with all factors in factor_dict. Can be tuple like output of Domain.attrs 
    domain_size: Domain sizes of full_domain. Can be tuple like output of Domain.shape
    consistent_iter: Number of iterations to run consist_views for. 
                    This probably determines how well consistency between common attributes is.
    """
    #setting up variables
    full_domain, domain_size = domain.attrs, domain.shape
    consist_views = {}
    consist_parameters = {"consist_iterations": consistent_iter}
    domain_size = np.array(copy.deepcopy(domain_size), dtype=np.int64)
    full_domain = list(copy.deepcopy(full_domain))
    #creating dictionary of Views for consistenter
    for k in factor_dict:
        if len(k) == 1:
            temp_key = k[0]
        else:
            temp_key = k
            
        indicator = calcOneHot(k, full_domain)
    
        temp_view = View(indicator, domain_size)
        temp_view.count = factor_dict[k].datavector()
        
        consist_views[temp_key] = temp_view
    #running consist_views()
    consistenter = Consistenter(consist_views, domain_size, consist_parameters)
    consistenter.consist_views()
    
    #setting up output
    output_dict = {}
    output_domain = Domain(full_domain,domain_size)
    
    #reconverting to dictionary of Factor
    for v in consistenter.views:
        output_dict[tuple(v)] = Factor(output_domain.project(tuple(v)),consistenter.views[v].count)
        
    return CliqueVector(output_dict)
    
if __name__ == "__main__":
    var = ('A','B','C','D')
    size = (2,3,4,5)
    
    domain = Domain(var,size)
    
    factors = {}
    factors[('A','B','C')] = Factor.zeros(domain.project(('A','B','C')))
    factors[('C',)] = Factor.zeros(domain.project(('C',)))
    factors[('B','C', 'D')] = Factor.uniform(domain.project(('B','C', 'D')))

    
    
    
    for f in factors:
        print(factors[f].datavector(flatten=False))
    
    factors = transform(factors, domain.attrs, domain.shape)
    
    for f in factors:
        print(factors[f].datavector(flatten=False))

