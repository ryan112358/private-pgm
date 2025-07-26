"""
Implementation of an adaptive marginal-based neural network method (MargNet) for DP Synthetic Data
"""

import os
import numpy as np
import pandas as pd
import torch
import tomli
import json
import itertools
import argparse
from mbi import Dataset, Domain, MargNN
from cdp2adp import cdp_rho


def exponential_mechanism(score, rho, sensitivity):
    max_score = np.max(score)
    scaled_score = [s - max_score for s in score]
    exp_score = [np.exp(np.sqrt(2*rho)/sensitivity * s) for s in scaled_score]
    sample_prob = [score/sum(exp_score) for score in exp_score]
    id = np.random.choice(np.arange(len(exp_score)), p = sample_prob)
    return id


def scale_vector(vec):
    temp_vec = np.clip(vec, 0, np.inf)
    return temp_vec/np.sum(temp_vec)


class MargDLGen():
    def __init__(
            self, 
            epsilon: float,
            delta: float,
            dataset: Dataset,  
            domain: Domain
        ):
        self.rho = 0 if delta == 0 else cdp_rho(epsilon, delta)
        self.dataset = dataset
        self.domain = domain
        self.save_loss = False

        self.model = MargNN(
            domain = self.domain.config
        )


    def exponential_marginal_selection(self, marginal_candidates, rho_select, rho_measure, candidates_weight):
        score = []
        weight = [candidates_weight[cl] for cl in marginal_candidates]

        syn_marginals = self.model.obtain_sample_marginals(marginal_candidates)
        real_marginals = [self.dataset.project(cl).datavector() for cl in marginal_candidates]

        for i in range(len(marginal_candidates)):
            score.append(weight[i] * (np.linalg.norm(self.est_n * syn_marginals[i] - real_marginals[i], 1)\
                         - np.sqrt(1/(np.pi * rho_measure)) * real_marginals[i].size))

        idx = exponential_mechanism(score, rho_select, max(weight))

        return idx
    

    def update_est_n(self, num, rho):
        if rho is not None:
            if self.est_n == 0:
                self.est_n = num
                self.acc_weight = np.sqrt(rho)
            else:
                self.est_n = (self.acc_weight * self.est_n + np.sqrt(rho) * num)/(self.acc_weight + np.sqrt(rho))
                self.acc_weight += np.sqrt(rho)
        else:
            pass 



    def fit_adaptive(self):
        select_rho = 0.1*self.rho/(16*len(self.dataset.domain))
        measure_rho = 0.9*self.rho/(16*len(self.dataset.domain))
        self.est_n = 0  # an estimation of number of data records
        rho_used = 0.0
        weight = 1.0
        enhance_weight = len(self.dataset.domain) # a training trick

        one_way_marginals = list(itertools.combinations(self.domain.attrs, 1))
        selected_marginals = []
        for cl in one_way_marginals:
            marginal = self.dataset.project(cl).datavector()
            marginal += np.random.normal(loc=0, scale=1/np.sqrt(2*measure_rho), size=marginal.shape)

            selected_marginals.append(
                (cl, scale_vector(marginal), weight)
            )
            rho_used += measure_rho * len(one_way_marginals)
            self.update_est_n(np.sum(marginal), measure_rho)

        print('-'*100)
        print('Initialization')
        self.model.store_marginals(selected_marginals)
        self.model.train_model()
        print('-'*100)

        two_way_marginals = list(itertools.combinations(self.domain.attrs, 2))
        candidates_mask = {cl: 1 for cl in one_way_marginals}
        terminate = False

        round = 1
        while not terminate:
            marg_candidates = two_way_marginals
            candidates_select_weight = {cl: 2.0 for cl in marg_candidates}

            id = self.exponential_marginal_selection(marg_candidates, select_rho, measure_rho, candidates_select_weight)
            cl = marg_candidates[id]

            if cl not in candidates_mask.keys():
                candidates_mask[cl] = 1
            else:
                candidates_mask[cl] += 1
            print('selected marginal:', cl)

            marginal = self.dataset.project(cl).datavector()
            marginal += np.random.normal(loc=0, scale=1/np.sqrt(2*measure_rho), size=marginal.shape)

            one_selected_marginals = [(cl, scale_vector(marginal), enhance_weight*weight)]
            selected_marginals += one_selected_marginals
            w_t = self.model.obtain_sample_marginals([cl])[0]

            self.model.store_marginals(selected_marginals)   
            self.model.train_model()
            selected_marginals[-1] = (one_selected_marginals[0][0], one_selected_marginals[0][1], weight)

            rho_used += measure_rho + select_rho
            if rho_used + measure_rho + select_rho > self.rho:
                weight = weight * np.sqrt(0.9 * (self.rho - rho_used)/measure_rho)
                measure_rho = 0.9*(self.rho - rho_used) 
                select_rho = 0.1*(self.rho - rho_used) 
                terminate = True
            else:
                w_t_plus_1 = self.model.obtain_sample_marginals([cl])[0]
                if self.est_n * np.linalg.norm(w_t_plus_1 - w_t, 1) < np.sqrt(1/(measure_rho * np.pi)) * w_t_plus_1.size:
                    if candidates_mask[cl] == 1:
                        print('-'*100)
                        print('!!!!!!!!!!!!!!!!! sigma updated')
                        print('-'*100)
                        weight *= np.sqrt(2)
                        measure_rho *= 2
                        select_rho *= 2
            
            print('-'*100)
            round += 1

        print('finish marginal selection')
        print('selected marginals:', list(candidates_mask.keys()))
        self.model.store_marginals(selected_marginals)
        self.model.train_model()

        return selected_marginals
    

    def sample(self, num_samples):
        syn_data = self.model.sample(num_samples)
        syn_df = pd.DataFrame(np.array(syn_data), columns=self.dataset.domain.attrs)
        return Dataset(syn_df, self.dataset.domain)
   

# --- Example Usage ---
def default_params():
    """Returns a dictionary of default parameters for the JAM mechanism."""
    params = {}
    params["dataset"] = "../data/adult.csv"
    params["domain"] = "../data/adult-domain.json"
    params["epsilon"] = 10.0
    params["delta"] = 1e-9
    
    return params


if __name__ == '__main__':
    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    generator = MargDLGen(
        epsilon = args.epsilon,
        delta = args.delta,
        dataset = data,
        domain = data.domain
    )
    generator.fit_adaptive()
    synth = generator.sample(data.df.shape[0])

    workload = list(itertools.combinations(data.domain, 2))
    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print("Average Error: ", np.mean(errors))

    

