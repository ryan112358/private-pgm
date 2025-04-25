**********
User Guide
**********

This guide provides a comprehensive overview of how to use the `private-pgm` library to generate differentially private synthetic data.

Core Workflow
=============

The typical workflow for using `private-pgm` involves the following steps:

1.  **Load and Prepare Data:** Load your dataset and define its domain (variable types, ranges, etc.).
2.  **Select and Configure a Mechanism:** Choose a differentially private mechanism appropriate for your data and privacy requirements.
3.  **Run the Mechanism:** Execute the chosen mechanism to learn a privacy-preserving model from your data.
4.  **Generate Synthetic Data:** Use the learned model to synthesize new data points.
5.  **Evaluate Utility (Optional):** Assess the quality and utility of the generated synthetic data.

Step 1: Loading and Preparing Data
==================================

The first step is to load your data into a suitable format and define its domain.

Loading Data
------------
You'll typically start with a CSV file or a Pandas DataFrame. The `mbi.Dataset` class can be used to represent your data.

.. code-block:: python

    from mbi import Dataset
    import pandas as pd

    # Example: Loading the adult dataset
    data_df = pd.read_csv('../data/adult.csv') # Adjust path as needed
    # Define domain information (attributes and their possible values/ranges)
    # This might be loaded from a JSON file or defined in code
    # e.g., domain_info_path = '../data/adult-domain.json'

    # Assuming you have a way to create a Domain object
    # from mbi import Domain
    # domain = Domain.load(domain_info_path) # Hypothetical load method
    # dataset = Dataset(df=data_df, domain=domain)

    # For now, let's assume a simpler instantiation if your Dataset class handles it:
    # dataset = Dataset.load(data_path='../data/adult.csv', domain_path='../data/adult-domain.json')

Refer to :class:`mbi.Dataset` and :class:`mbi.Domain` in the :doc:`API documentation <api/index>` for details.

Defining the Domain
-------------------
The domain specifies the attributes in your dataset, their types (e.g., categorical, numerical), and their possible values or ranges. This is crucial for many DP algorithms. The `mbi.Domain` class is used for this.

.. code-block:: python

    from mbi import Domain

    # Example: Defining a domain programmatically
    # attributes = {
    #     'age': {'type': 'numerical', 'min': 17, 'max': 90},
    #     'workclass': {'type': 'categorical', 'values': ['Private', 'Self-emp-not-inc', ...]},
    #     # ... other attributes
    # }
    # domain = Domain(attributes)

    # Or loading from a file (e.g., JSON)
    # domain = Domain.from_json('../data/adult-domain.json')

Step 2: Selecting and Configuring a Mechanism
=============================================
`private-pgm` implements various differentially private mechanisms for learning graphical models and generating synthetic data. These can be found in the ``mechanisms`` directory of the repository and often involve specific modules within ``mbi``.

Some potential mechanisms you might work with (based on your repository structure):

* **MST (Maximum Spanning Tree):** Implemented in `mechanisms/mst.py`. Often used for selecting a set of low-order marginals.
* **AIM (Adaptive Iterative Mechanism):** Implemented in `mechanisms/aim.py`. An iterative mechanism for answering a large number of counting queries.
* **MWEM+PGM:** Implemented in `mechanisms/mwem+pgm.py`. Combines the Multiplicative Weights Exponential Mechanism with Probabilistic Graphical Models.
* **HDMM+APPGM:** Implemented in `mechanisms/hdmm+appgm.py`.
* **Gaussian+APPGM:** Implemented in `mechanisms/gaussian+appgm.py`.

You'll need to instantiate the chosen mechanism and configure its parameters, such as the privacy budget (epsilon, delta).

.. code-block:: python

    # Example: Placeholder for using a mechanism
    # from mechanisms import MSTMechanism # Fictional import for illustration
    # from mbi import GraphicalModel

    # epsilon = 1.0
    # delta = 1e-9
    # mechanism = MSTMechanism(epsilon=epsilon, delta=delta, domain=dataset.domain)

Step 3: Running the Mechanism
=============================
Once the mechanism is configured, you run it on your dataset. This process typically involves privately selecting queries (e.g., marginals), measuring them with noise, and then fitting a model.

.. code-block:: python

    # private_model_representation = mechanism.run(dataset)
    # graphical_model = GraphicalModel.from_private_representation(private_model_representation, dataset.domain)

The output will be a privacy-preserving representation of the data's structure and distributions, often encapsulated in an `mbi.GraphicalModel` object or similar.

Step 4: Generating Synthetic Data
=================================
With the learned private model, you can now generate synthetic data.

.. code-block:: python

    from mbi import SyntheticData

    # Assuming graphical_model is the output from Step 3
    # num_synthetic_samples = len(dataset.df) # Generate same number of samples as original
    # synthetic_dataset_generator = SyntheticData(graphical_model)
    # synthetic_df = synthetic_dataset_generator.sample(num_synthetic_samples)

    # print(synthetic_df.head())

Step 5: Evaluating Utility (Optional)
=====================================
It's good practice to evaluate how well the synthetic data preserves the statistical properties of the original data. This can involve:

* Comparing marginal distributions.
* Running machine learning tasks on both original and synthetic data and comparing performance.
* Calculating specific utility metrics.

The `mbi.marginal_loss` module might contain relevant functions for this.

.. code-block:: python

    # Example: Basic utility check (conceptual)
    # original_marginal = dataset.project(['age', 'workclass']).datavector()
    # synthetic_dataset_for_eval = Dataset(synthetic_df, dataset.domain)
    # synthetic_marginal = synthetic_dataset_for_eval.project(['age', 'workclass']).datavector()

    # from mbi.marginal_loss import total_variation_distance # if it exists
    # tvd = total_variation_distance(original_marginal, synthetic_marginal)
    # print(f"Total Variation Distance for (age, workclass): {tvd}")

Further Topics
==============

* **Advanced Configuration:** How to fine-tune parameters for different mechanisms.
* **Privacy Budget Management:** Strategies for allocating and tracking privacy budgets across multiple operations.
* **Supported Data Types:** Details on handling categorical, numerical, and other data types.

Please refer to the specific example scripts in the :doc:`examples/index` section and the detailed :doc:`api/index` for more in-depth information on classes and functions.
