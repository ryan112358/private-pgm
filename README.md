# Marginal-based estimation and inference for differential privacy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5548533.svg)](https://doi.org/10.5281/zenodo.5548533)
[![Continuous integration](https://github.com/ryan112358/private-pgm/actions/workflows/main.yml/badge.svg)](https://github.com/ryan112358/private-pgm/actions/workflows/main.yml)

# Summary

This is a general-purpose library for estimating discrete distributions from noisy
observations of their marginals.  To that end, we provide scalable algorithms for solving convex optimization problems over the marginal polytope.  From that, we obtain an undirected graphical model (i.e., a Markov random field) which can be used to (1) obtain more accurate estimates of the observed noisy marginals, (2) estimate answers to new queries (e.g., marginals) based on a maximum entropy assumption, and (3) generate synthetic data that approximately preserves those marginals.  

The library is designed with the following core principles in mind:

* **Consistency:** This library produces consistent estimates of measured and unmeasured queries, (i.e., answers that could arise from some global data distribution).
* **Utility**: The estimation algorithms make the best use of the noisy measurements
by combining all sources of information in a principled manner (e.g., by maximizing the likelihood of the noisy observations).  The algorithms work well in practice.
* **Scalability**: The library scales effectively to high-dimensional datasets.
* **Flexibility**: The library supports generic loss functions defined over the low-dimensional marginals of a data distribution.  We provide APIs and utilities for constructing a loss function based on the broad class of noisy linear measurements of the marginals.  Flexibility beyond this is possible and easy to configure.  
* **Extensibility:** The library is designed to be built upon by future work.  We focus on the important but narrow problem of estimating a data distribution from noisy measurements, and provide a good solution for that.  This is only one component of a strong mechanism, and must be combined with principled methods for selecting *which* queries should be privately measured based on the data, privacy budget, downstream task, and other considerations.
* **Simplicity:** The library is simple to use, toy examples only require a few lines of code, and fairly powerful mechanisms can be developed on top of this library with surprisingly little code.  

Published articles: 

- McKenna, Ryan, Terrance Liu, "A simple recipe for private synthetic data generation."
[https://differentialprivacy.org/synth-data-1/](https://differentialprivacy.org/synth-data-1)

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
* A new belief propagation algorithm is used which is more space efficient than the prior one in settings where the maximal cliques in the Junction Tree are larger than the measured cliques.  This new algorithm is also significantly faster when running on GPUs in some cases.
* Expanded test coverage for marginal inference, we correctly handle a number of tricky edge cases.
* Added type information to most functions for better documentation.  Also added example usages to some functions in the form of doctests.  More will be added in the future.
* Setup continuous integration tests on GitHub.

Currently, not all functionality that was previously supported been integrated into the new design.  However, the core features that are used by the majority of the use cases have been.  These left out functionalities are discussed in the end of this document. 

# Codebase organization

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
>>> measurements = [
    LinearMeasurement(yab, ('A', 'B'), sigma), 
    LinearMeasurement(ybc, ('B', 'C'), sigma)
]
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

We require Python>=3.9.  Dependencies can be installed via the standard pip install command:

```
$ pip install -r requirements.txt
```

Additionally, you have to add the src folder to the PYTHONPATH.  If you are using Ubuntu, add the following line to your .bashrc file:

```
PYTHONPATH=$PYTHONPATH:/path/to/private-pgm/src
```

This allows you to import modules from this package like ``` from mbi import Domain ``` no matter what directory you are working in.  Once this is done, check to make sure the tests are passing

```
$ cd /path/to/private-pgm/test
$ pytest
test_approximate_oracles.py ............                                            [  4%]
test_dataset.py ..                                                                  [  4%]
test_domain.py ..........                                                           [  8%]
test_estimation.py ................................................                 [ 24%]
test_factor.py ......                                                               [ 26%]
test_marginal_oracles.py .......................................................... [ 46%]
................................................................................... [ 74%]
...........................................................................         [100%]

================================== 294 passed in 30.65s ===================================
```

# Contributing to this repository

Contributions to this repository are welcome and encouraged.

* Adding new functionality (new estimators, synthetic data generators, etc.).
* Filing bugs if you run into any errors or unexpected performance characteristics.
* Improving documentation.
* Checking in mechanisms you built on top of this library.
* Code that got cut during the migration might be nice to add back in under the new design.
>* Calculate many marginals (theoretically more efficient than calling "project" multiple times)
>* Answering Kronecker product queries
>* Some approximate marginal inference oracles that were previously implemented did not get moved over, only the one we presented in our paper.
>* Some examples were deleted rather than ported, since they took dependencies on other github repositories.  
>* Calculation of Lipschitz constant for L2 losses
>* We did not refactor other marginal-based estimation algorithms in terms of the current design.  [PublicInference](https://ppai21.github.io/files/26-paper.pdf) and [MixtureInference](https://arxiv.org/abs/2103.06641), are still implemented in terms of the old API, and are in the experimental subpackage.  


