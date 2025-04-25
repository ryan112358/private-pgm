************
Introduction
************

Welcome to the `private-pgm` library! This project provides tools and algorithms for generating synthetic data using **Probabilistic Graphical Models (PGMs)** while ensuring **Differential Privacy (DP)**.

The core goal of `private-pgm` is to enable data analysts and researchers to share and work with sensitive datasets in a privacy-preserving manner. By leveraging the strengths of PGMs to capture complex dependencies within data and integrating rigorous DP mechanisms, this library aims to produce synthetic data that is both useful for analysis and safe to share.

Key Concepts
============

* **Differential Privacy:** A formal mathematical definition of privacy that provides strong guarantees against re-identification and information leakage. Algorithms in this library are designed to satisfy specific DP constraints (e.g., (epsilon, delta)-DP).
* **Probabilistic Graphical Models:** A framework for representing and reasoning about uncertainty and dependencies among variables using graph structures. PGMs are used here to learn the underlying distribution of the original data.
* **Synthetic Data:** Artificially generated data that mimics the statistical properties of real-world data but does not contain actual records from the original dataset.

What can you do with `private-pgm`?
===================================

* Learn privacy-preserving graphical models from sensitive datasets.
* Generate high-fidelity synthetic datasets that maintain statistical utility.
* Experiment with various state-of-the-art DP mechanisms tailored for graphical models and synthetic data generation.
* Evaluate the trade-off between privacy guarantees and the utility of the generated synthetic data.

Getting Around
==============

* For installation instructions, see :doc:`installation`.
* To learn how to use the library, head over to the :doc:`user_guide`.
* Practical examples can be found in :doc:`examples/index`.
* For detailed information on modules and functions, consult the :doc:`api/index`.

We hope `private-pgm` helps you in your privacy-preserving data analysis endeavors!
