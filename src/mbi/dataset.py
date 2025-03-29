import numpy as np
import pandas as pd
import os
import json
from mbi import Domain


class Dataset:
    def __init__(self, df, domain, weights=None):
        """create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(
            df.columns
        ), "data must contain domain attributes"
        assert weights is None or df.shape[0] == weights.size
        self.domain = domain
        self.df = df.loc[:, domain.attrs]
        self.weights = weights

    @staticmethod
    def synthetic(domain, N):
        """Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path, domain):
        """Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)

    def project(self, cols):
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        return self.df.shape[0]

    def datavector(self, flatten=True):
        """return the database in vector-of-counts form"""
        if flatten is not True:
            bins = [range(n + 1) for n in self.domain.shape]
            return np.histogramdd(self.df.to_numpy(), bins, weights=self.weights)[0]

        # Faster implementation using a contiguous array (assuming the domain matches the data)
        arr = self.df.to_numpy()
        shape_row = arr.shape[1]
        rowtype = np.dtype((np.void, arr.dtype.itemsize * shape_row))
        arr_flat = np.ascontiguousarray(arr).view(rowtype)
        arr_flat = arr_flat.reshape(-1, arr_flat.shape[1])
        return np.unique(arr_flat, axis=0, return_counts=True)[1]
