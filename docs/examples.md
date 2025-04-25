# Examples

## Toy example

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

## Additional Examples

The files in the "examples" folder are meant to demonstrate how to use Private-PGM for different problems.  These files are typically not designed to run on arbitrary datasets and/or workloads.  They are more useful if you are trying to learn how to use this code base and would like to build your own mechanisms on top of it.

If you would simply like to compare against mechanisms that are built on top of Private-PGM, please refer to the "mechanisms" folder.  These will contain implementations of several mechanisms that are designed to work on a broad class of datasets and workloads.  The mechanisms currently available here are:

* [AIM](https://arxiv.org/abs/2201.12677) - An adaptive and iterative mechanism
* [MST](https://arxiv.org/abs/2108.04978) - Winning solution to the 2018 NIST synthetic data challenge.
* [Adaptive Grid](https://github.com/ryan112358/nist-synthetic-data-2021) - Second place solution to the 2020 NIST synthetic data challenge.
* [MWEM+PGM](https://arxiv.org/pdf/1901.09136.pdf) - A scalable instantiation of the MWEM algorithm for marginal query workloads.
* [HDMM+APPGM](https://arxiv.org/abs/2109.06153) - A scalable instantiation of the HDMM algorithm for marginal query workloads.

For the methods above, MST and Adaptive Grid are workload agnostic.  Nevertheless, we expect them to do well on general workloads like all 3-way marginals.  MWEM+PGM, HDMM+APPGM, and AIM are all workload aware, and are designed to offer low error on the marginals specified in the workload.  

NOTE: The first three mechanisms produce synthetic data, but HDMM+APPGM only produces query answers, not synthetic data.

NOTE: As this is research (non-production) code, all of the mechanisms in this repository use np.random.normal or np.random.laplace for noise generation.  These floating-point approximations of the Gaussian/Laplace mechanism are susceptible to floating point attacks like those described by [Mironov](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/10/lsbs.pdf).  If you would like to use these mechanisms in a production system that is robust to such attacks, we recommend you use an implementation of these mechanisms that was specifically designed with floating point artifacts in mind, such as the [Discrete Gaussian Mechanism](https://github.com/IBM/discrete-gaussian-differential-privacy), [OpenDP](https://docs.opendp.org/en/stable/user/measurements/additive-noise-mechanisms.html?highlight=laplace), or [Tumult Analytics](https://docs.tmlt.dev/analytics/latest/tutorials/first-steps.html).  For the purposes of evaluating the utility of our mechanisms in research code, our implementations are fine to use, i.e., the floating point vulnerability does not meaningfully change the utility of the mechansim in any way. 