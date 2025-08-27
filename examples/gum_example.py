import itertools
import numpy as np
from mbi import Domain, Factor, Dataset
from mbi.gum import synthesize_gum

def get_total_error(dataset, target_marginals, domain):
    """Helper to calculate total L1 error for a set of marginals."""
    total_error = 0
    attr_to_col_idx = {attr: i for i, attr in enumerate(domain.attrs)}
    if dataset.shape[0] == 0:
        # The L1 error for an empty dataset is the sum of all target counts.
        # Here we calculate Total Variation distance, so it's 1.0 for each marginal.
        return len(target_marginals)

    for target in target_marginals:
        cols = [attr_to_col_idx[attr] for attr in target.domain.attrs]

        # Calculate current marginal from dataset
        marginal_data = dataset[:, cols]
        flat_indices = np.ravel_multi_index(
            marginal_data.T, target.domain.shape
        )
        current_counts = np.bincount(flat_indices, minlength=target.values.size)

        # Normalize both to compare distributions
        current_dist = current_counts / current_counts.sum()
        target_dist = target.values / target.values.sum()

        total_error += 0.5 * np.sum(np.abs(target_dist.flatten() - current_dist.flatten()))

    return total_error


import logging

def main():
    """
    An example of using the GUM synthesizer with the Adult dataset.

    This script loads the Adult dataset, calculates its true marginals,
    and then uses the GUM algorithm to generate a synthetic dataset that approximates
    these marginals. Finally, it measures and prints the L1 error.
    """
    # Configure logging to see the progress of the synthesizer
    logging.basicConfig(level=logging.INFO)

    # 1. Load the Adult dataset
    print("Loading Adult dataset...")
    data = Dataset.load(path='data/adult.csv', domain='data/adult-domain.json')
    true_data = data.df.values
    domain = data.domain
    n_records = true_data.shape[0]


    # 2. Define the target marginals (all 2-way marginals)
    print("Calculating target marginals from the true dataset...")
    marginals_to_target = list(itertools.combinations(domain.attrs, 2))

    target_marginals = []
    for attrs in marginals_to_target:
        factor = data.project(attrs)
        target_marginals.append(factor)

    # 3. Create an initial dataset for GUM (e.g., an empty dataset)
    # Starting with an empty dataset can be more stable when the target marginals are known.
    print("Creating an empty initial dataset for GUM...")
    initial_dataset = np.zeros((0, len(domain.attrs)), dtype=int)


    # 4. Run the GUM synthesizer
    print("Running GUM synthesizer...")
    final_dataset = synthesize_gum(
        initial_dataset,
        target_marginals,
        record_count=n_records,
        num_iterations=10, # Note: more iterations may yield better results
        alpha=0.1,
        beta=0.01, # Using a smaller beta is crucial when the initial dataset is very noisy
        verbose=True
    )

    # 5. Calculate and print the L1 error
    initial_error = get_total_error(initial_dataset, target_marginals, domain)
    final_error = get_total_error(final_dataset, target_marginals, domain)

    print("\n--- Results ---")
    print(f"Total L1 Error of Initial (Random) Dataset: {initial_error:.4f}")
    print(f"Total L1 Error of Final (GUM) Dataset:   {final_error:.4f}")
    print(f"Error Reduction: {(initial_error - final_error) / initial_error:.2%}")


if __name__ == '__main__':
    main()
