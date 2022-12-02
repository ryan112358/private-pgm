# Graphical-model based estimation and inference for differential privacy



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5548533.svg)](https://doi.org/10.5281/zenodo.5548533)

Published articles: 

- McKenna, Ryan, Daniel Sheldon, and Gerome Miklau. 2021. "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data."  *Journal of Privacy and Confidentiality* 11 (3).  [![DOI:10.29012/jpc.778](https://zenodo.org/badge/DOI/10.29012/jpc.778.svg)](https://doi.org/10.29012/jpc.778) 


- McKenna, Ryan, Daniel Sheldon, and Gerome Miklau. "Graphical-model based estimation and inference for differential privacy." In Proceedings of the 36th International Conference on Machine Learning. 2019. https://arxiv.org/abs/1901.09136

# Important Update

For anyone interested in using this code-base, the "examples" folder has been reorganized.  Specifically, the files in the "examples" folder are meant to demonstrate how to use Private-PGM for different problems.  These files are typically not designed to run on arbitrary datasets and/or workloads.  They are more useful if you are trying to learn how to use this code base and would like to build your own mechanisms on top of it.

If you would simply like to compare against mechanisms that are built on top of Private-PGM, please refer to the "mechanisms" folder.  These will contain implementations of several mechanisms that are designed to work on a broad class of datasets and workloads.  The mechanisms currently available here are:

* [AIM](https://arxiv.org/abs/2201.12677) - An adaptive and iterative mechanism
* [MST](https://arxiv.org/abs/2108.04978) - Winning solution to the 2018 NIST synthetic data challenge.
* [Adaptive Grid](https://github.com/ryan112358/nist-synthetic-data-2021) - Second place solution to the 2020 NIST synthetic data challenge.
* [MWEM+PGM](https://arxiv.org/pdf/1901.09136.pdf) - A scalable instantiation of the MWEM algorithm for marginal query workloads.
* [HDMM+APPGM](https://arxiv.org/abs/2109.06153) - A scalable instantiation of the HDMM algorithm for marginal query workloads.

For the methods above, MST and Adaptive Grid are workload agnostic.  Nevertheless, we expect them to do well on general workloads like all 3-way marginals.  MWEM+PGM, HDMM+APPGM, and AIM are all workload aware, and are designed to offer low error on the marginals specified in the workload.  

NOTE: The first three mechanisms produce synthetic data, but HDMM+APPGM only produces query answers, not synthetic data.

# Toy example

Suppose we have a unknown data distribution P(A,B,C) defined over three variables and we invoke a noise-addition mechanism to get a noisy answer to the two-way marginals P(A,B) and P(B,C).  We want to use this information to recover a representation of the data distribution that approximates the true data with respect to the measured marginals.  We can do this as follows:

First, load synthetic data with 1000 rows over a domain of size 2 x 3 x 4

```
>>> from mbi import Dataset, Domain, FactoredInference
>>> import numpy as np
>>> 
>>> domain = Domain(['A','B','C'], [2,3,4])
>>> data = Dataset.synthetic(domain, 1000)
```

Then measure the AB marginal and BC marginal using the Laplace mechanism
```
>>> epsilon = 1.0
>>> sigma = 1.0 / epsilon
>>> ab = data.project(['A','B']).datavector()
>>> bc = data.project(['B','C']).datavector()
>>> yab = ab + np.random.laplace(loc=0, scale=sigma, size=ab.size)
>>> ybc = bc + np.random.laplace(loc=0, scale=sigma, size=bc.size)
```

Now feed these noisy measurements into the inference engine using the Mirror Descent (MD) algorithm

```
>>> Iab = np.eye(ab.size)
>>> Ibc = np.eye(bc.size)
>>> measurements = [(Iab, yab, sigma, ('A','B')), (Ibc, ybc, sigma, ('B','C'))]
>>> engine = FactoredInference(domain, log=True)
>>> model = engine.estimate(measurements, engine='MD')
```

Now model can be used as if it were the true data to answer any new queries

```
>>> ab2 = model.project(['A','B']).datavector()
>>> bc2 = model.project(['B','C']).datavector()
>>> ac2 = model.project(['A','C']).datavector()
```

# Automatic Setup

```
$ pip install git+https://github.com/ryan112358/private-pgm.git
```

# Manual Setup

We officially support python3, and have the following dependencies: numpy, scipy, pandas, matplotlib, and networkx.  These can be installed with pip as follows:

```
$ pip install -r requirements.txt
```

Additionally, you have to add the src folder to the PYTHONPATH.  If you are using Ubuntu, add the following line to your .bashrc file:

```
PYTHONPATH=$PYTHONPATH:/path/to/private-pgm/src
```

This allows you to import modules from this package like ``` from mbi import FactoredInference ``` no matter what directory you are working in.  Once this is done, check to make sure the tests are passing

```
$ cd /path/to/private-pgm/test
$ nosetests
........................................
----------------------------------------------------------------------
Ran 40 tests in 5.009s

OK
```

# PyTorch

In addition to the above setup, if you have access to a GPU machine you can use PyTorch to accelerate the computations for the Inference engine.  This requires changing only one line of code:

```
>>> engine = FactoredInference(domain, backend='torch', log=True)
```

See https://pytorch.org/ for instructions to install PyTorch in your environment.

# Documentation

This package contains the following public-facing classes: **Domain, Dataset, Factor, GraphicalModel, and FactoredInference**.

* **Domain**: contains information about the attributes in the dataset and the number of possible values for each attribute.  **NOTE**: It is implicitly assumed that the set of possible values for an attribute is { 0, ..., n-1 }.

* **Dataset**: a class for storing tabular data.  Can convert to the vector representation of the data by calling **datavector()** and can project the data onto a subset of attributes by calling **project()**.  **NOTE**: This class requires the underlying data to conform to the domain (i.e., the set of possible values for an attribute should be { 0, ..., n-1 }).

* **Factor**: A representation of a multi-dimensional array that also stores domain information.  Is used by GraphicalModel.

* **GraphicalModel**: A factored representation of a probability distribution that allows for efficient calculation of marginals and other things.  The interface for **GraphicalModel** is similar to **Dataset**, so they can be used in the same way (i.e., you can **project()** onto a subset of attributes and obtain the **datavector()**).

* **FactoredInference**: A class for performing efficient estimation/inference.  This class contains methods for estimating the data distribution (with a **GraphicalModel**) that approximates some true underlying distribution with respect to noisy measurements over the marginals.  Measurements must be represented as a list of 4-tuples:  (Q, y, noise, proj), where
    * proj (tuple): is a subset of attributes corresponding the marginal the measurements were taken over.
    * Q (matrix): is the measurement matrix (can be a numpy array, scipy sparse matrix, or any subclass of scipy.sparse.LinearOperator).
    * y (vector) is the noisy answers to the measurement queries (should be a numpy array).
    * noise (scalar): is the standard deviation of the noise added to y.

The **estimate()** method allows you to estimate the data distribution from noisy measurements in this form.  Optional arguments are **total** (if using bounded differential privacy) and **engine** (to specify which estimation algorithm should be used).  This class also has a number of other optional arguments, including **structural_zeros** (if some combinations of attribute settings are impossible), **metric** to specify the marginal loss function (L1, L2, or custom), **log** (to display progress of estimation), **iters** (number of iterations), and some others.

In addition to these public-facing classes, there are utilities such as **mechanism** (for running end-to-end experiments) and **callbacks** (for customized monitoring of the estimation procedure). 

# Examples

Additional examples can be found in the examples folder.  In addition to the toy example above, there are a number of more realistic examples using the adult dataset.  The file **adult_example** is similar to **toy_example**, but is bigger and slightly more complicated.  The file **torch_example** shows how the torch backend can be used.  The file **convergence** compares the rate of convergence for the different estimation algorithms.  

The files **hdmm, privbayes, mwem, and dualquery** show how to use our technique to improve these existing algorithms.  These files have two additional dependencies: [Ektelo](https://github.com/ektelo/ektelo) and [autograd](https://github.com/HIPS/autograd).
