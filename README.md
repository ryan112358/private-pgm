# Graphical-model based estimation and inference for differential privacy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5548533.svg)](https://doi.org/10.5281/zenodo.5548533)
[![Continuous integration](https://github.com/ryan112358/private-pgm/actions/workflows/main.yml/badge.svg)](https://github.com/ryan112358/private-pgm/actions/workflows/main.yml)

Published articles: 

- McKenna, Ryan, Daniel Sheldon, and Gerome Miklau. 2021. "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data."  *Journal of Privacy and Confidentiality* 11 (3).  [![DOI:10.29012/jpc.778](https://zenodo.org/badge/DOI/10.29012/jpc.778.svg)](https://doi.org/10.29012/jpc.778) 

- McKenna, Ryan, Daniel Sheldon, and Gerome Miklau. "Graphical-model based estimation and inference for differential privacy." In Proceedings of the 36th International Conference on Machine Learning. 2019. https://arxiv.org/abs/1901.09136


- McKenna, Ryan, Siddhant Pradhan, Daniel Sheldon, and Gerome Miklau. "Relaxed Marginal Consistency for Differentially Private Query Answering".  In Proceedings of the 35th Conference on Nueral Information Processing Systems, 2021. https://arxiv.org/pdf/2109.06153

- McKenna, Ryan, Brett Mullins, Daniel Sheldon, and Gerome Miklau. "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data".  In Proceedings of the VLDB Endowment, Vol 15, 2023. https://arxiv.org/abs/2201.12677

# Update 11/2024

This library recently underwent major refactorings.  Below are a list of notable changes.

* Library has been modernized to use more recent python features.  Therefore, we require Python>=3.9.
* JAX is now used as the numerical backend rather than numpy.  This means the code now natively supports running on GPUs, although the scalability advantages have not yet been tested.
* To more naturally support JAX with JIT compilation, the code has been reorganized into a more functional design.  This design follows closely from how we describe the approach mathematically in our papers.
* Classes now have more narrowly defined scope, and have been removed where they did not provide significant utility.  Dataclasses are used liberally throughout the code.
* A new belief propagation algorithm is used which is more space efficient than the prior one in settings where the maximal cliques in the Junction Tree are larger than the measured cliques.
* Expanded test coverage for marginal inference, we correctly handle a number of tricky edge cases.
* Added type information to most functions for better documentation.  Also added example usages to some functions in the form of doctests.  More will be added in the future.
* Setup continuous integration tests on GitHub.


Currently, not all functionality that was previously supported been integrated into the new design.  However, the core features that are used by the majority of the use cases have been. These left out functionalities will be added back in on a best effort basis, including:

* **Marginal Inference Utilities**
    * Calculate many marginals (theoretically more efficient than calling "project" multiple times)
    * Answering Kronecker product queries
* **Estimation Algorithms**: 
    * Regularized Dual Averaging and Interior Gradient
    * Calculation of Lipschitz constant for L2 losses
    * Fitting graphical model parameters without noise.
* **Other Marginal-Based Inference Algorithms**:
    * We did not refactor other marginal-based inference algorithms in terms of the current design.  [PublicInference](https://ppai21.github.io/files/26-paper.pdf), [MixtureInference](https://arxiv.org/abs/2103.06641), and [Approx-PGM](https://arxiv.org/abs/2109.06153) are still implemented in terms of the old API, and are in the experimental subpackage.  Approx-PGM may be moved over to the main directory int eh future, while the others will remain in experimental and may be updated to match the API in the future.

# Codebase summary

The core library is implemented in "src/mbi" where mbi stands for "marginal-based inference".  The files in the "examples" folder are meant to demonstrate how to use Private-PGM for different problems.  These files are typically not designed to run on arbitrary datasets and/or workloads.  They are more useful if you are trying to learn how to use this code base and would like to build your own mechanisms on top of it.

If you would simply like to compare against mechanisms that are built on top of Private-PGM, please refer to the "mechanisms" folder.  These will contain implementations of several mechanisms that are designed to work on a broad class of datasets and workloads.  The mechanisms currently available here are:

* [AIM](https://arxiv.org/abs/2201.12677) - An adaptive and iterative mechanism
* [MST](https://arxiv.org/abs/2108.04978) - Winning solution to the 2018 NIST synthetic data challenge.
* [Adaptive Grid](https://github.com/ryan112358/nist-synthetic-data-2021) - Second place solution to the 2020 NIST synthetic data challenge.
* [MWEM+PGM](https://arxiv.org/pdf/1901.09136.pdf) - A scalable instantiation of the MWEM algorithm for marginal query workloads.
* [HDMM+APPGM](https://arxiv.org/abs/2109.06153) - A scalable instantiation of the HDMM algorithm for marginal query workloads.

For the methods above, MST and Adaptive Grid are workload agnostic.  Nevertheless, we expect them to do well on general workloads like all 3-way marginals.  MWEM+PGM, HDMM+APPGM, and AIM are all workload aware, and are designed to offer low error on the marginals specified in the workload.  

NOTE: The first three mechanisms produce synthetic data, but HDMM+APPGM only produces query answers, not synthetic data.

NOTE: As this is research (non-production) code, all of the mechanisms in this repository use np.random.normal or np.random.laplace for noise generation.  These floating-point approximations of the Gaussian/Laplace mechanism are susceptible to floating point attacks like those described by [Mironov](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/10/lsbs.pdf).  If you would like to use these mechanisms in a production system that is robust to such attacks, we recommend you use an implementation of these mechanisms that was specifically designed with floating point artifacts in mind, such as the [Discrete Gaussian Mechanism](https://github.com/IBM/discrete-gaussian-differential-privacy), [OpenDP](https://docs.opendp.org/en/stable/user/measurements/additive-noise-mechanisms.html?highlight=laplace), or [Tumult Analytics](https://docs.tmlt.dev/analytics/latest/reference/tmlt/analytics/query_expr/index.html?highlight=laplace#tmlt.analytics.query_expr.CountMechanism.LAPLACE).  For the purposes of evaluating the utility of our mechanisms in research code, our implementations are fine to use, i.e., the floating point vulnerability does not meaningfully change the utility of the mechansim in any way. 

# Toy example

Suppose we have a unknown data distribution P(A,B,C) defined over three variables and we invoke a noise-addition mechanism to get a noisy answer to the two-way marginals P(A,B) and P(B,C).  We want to use this information to recover a representation of the data distribution that approximates the true data with respect to the measured marginals.  We can do this as follows:

First, load synthetic data with 1000 rows over a domain of size 2 x 3 x 4

```
>>> from mbi import Dataset, Domain, LinearMeasurement, estimation
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
>>> measurements = [LinearMeasurement(yab, ('A', 'B'), sigma), LinearMeasurement(ybc, ('B', 'C'), sigma)]
>>> model = estimation.mirror_descent(domain, measurements)
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
$ pytest
........................................
----------------------------------------------------------------------
Ran 40 tests in 5.009s

OK
```

# Documentation

This package contains the following public-facing classes: **Domain, Dataset, Factor**

* **Domain**: contains information about the attributes in the dataset and the number of possible values for each attribute.  **NOTE**: It is implicitly assumed that the set of possible values for an attribute is { 0, ..., n-1 }.

* **Dataset**: a class for storing tabular data.  Can convert to the vector representation of the data by calling **datavector()** and can project the data onto a subset of attributes by calling **project()**.  **NOTE**: This class requires the underlying data to conform to the domain (i.e., the set of possible values for an attribute should be { 0, ..., n-1 }).

* **Factor**: A representation of a multi-dimensional array that also stores domain information.  Is used by GraphicalModel.
