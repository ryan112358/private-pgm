import logging
from functools import reduce
from typing import Literal, Optional

import numpy as np

from mbi.domain import Domain
from mbi.factor import Factor

logger = logging.getLogger(__name__)


def _calculate_new_learning_rate(
    initial_val: float,
    iteration: int,
    schedule: Optional[Literal["step", "exponential", "linear", "sqrt"]],
    decay_rate: float,
    decay_step: int,
) -> float:
    """Calculates the new learning rate based on a decay schedule."""
    if schedule is None:
        return initial_val
    if schedule == "step":
        if iteration > 0 and iteration % decay_step == 0:
            return initial_val * decay_rate
        return initial_val
    if schedule == "exponential":
        return initial_val * (decay_rate**iteration)
    if schedule == "linear":
        # Linear decay to zero
        return initial_val * max(0, 1 - decay_rate * iteration)
    if schedule == "sqrt":
        return initial_val / np.sqrt(max(1, iteration * decay_rate))

    # This should not be reached if validation is done correctly in the main function
    raise ValueError(f"Unknown decay schedule: {schedule}")


def synthesize_gum(
    initial_dataset: np.ndarray,
    target_marginals: list[Factor],
    record_count: Optional[int] = None,
    num_iterations: int = 100,
    alpha: float = 0.1,
    beta: float = 0.1,
    decay_schedule: Optional[Literal["step", "exponential", "linear",
                                     "sqrt"]] = None,
    decay_rate: float = 0.99,
    decay_step: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generates a synthetic dataset using the Gradually Update Method (GUM) algorithm.

    This function iteratively updates an initial dataset to better match a set of target
    marginal distributions.

    Based on the paper: "PrivSyn: Differentially Private Data Synthesis" by Zhang et al.

    Args:
        initial_dataset (np.ndarray): An initial dataset of integer indices (n_records x n_attributes).
                                      This dataset will be modified to match the target marginals.
        target_marginals (List[Factor]): A list of target marginal distributions (as Factor objects).
                                          The values of these factors are assumed to be target *counts*.
        record_count (Optional[int]): The desired number of records in the final synthetic dataset.
                                      If None, it defaults to the number of records in `initial_dataset`.
        num_iterations (int): The number of iterations to run the algorithm.
        alpha (float): The initial multiplicative update factor for adding records to under-counted cells.
        beta (float): The initial multiplicative update factor for removing records from over-counted cells.
        decay_schedule (Optional[Literal...]): The schedule for decaying alpha and beta over iterations.
        decay_rate (float): The rate of decay for 'exponential', 'linear', and 'sqrt' schedules.
        decay_step (int): The step size for the 'step' decay schedule.
        verbose (bool): If True, enables logging of progress.

    Returns:
        np.ndarray: The final synthetic dataset with dimensions (record_count x n_attributes).
    """
    # --- Parameter Validation ---
    if decay_schedule not in [None, "step", "exponential", "linear", "sqrt"]:
        raise ValueError(f"Unknown decay_schedule: {decay_schedule}")
    if not target_marginals:
        raise ValueError("target_marginals list cannot be empty.")

    # --- Initial Setup ---
    if initial_dataset.size == 0:
        logger.warning(
            "Initial dataset is empty. This may lead to poor results if not intended."
        )
        # If empty, we can't infer attributes, so we must rely on marginals
        if not all(m.domain.attrs for m in target_marginals):
            raise ValueError(
                "Cannot infer attributes from empty dataset and incomplete marginal domains."
            )

    current_synthetic_data = initial_dataset.copy()
    n_records, n_attributes = current_synthetic_data.shape
    target_n_records = record_count if record_count is not None else n_records

    # Combine domains from all marginals to get the full domain
    from functools import reduce
    full_domain = reduce(lambda d1, d2: d1.merge(d2),
                         [m.domain for m in target_marginals])

    if n_attributes != len(full_domain.attrs):
        raise ValueError(
            f"Number of attributes in initial_dataset ({n_attributes}) does not match "
            f"the number of attributes in the combined marginals' domain ({len(full_domain.attrs)})."
        )

    # Create a map from attribute name to column index for quick lookups
    attr_to_col_idx = {attr: i for i, attr in enumerate(full_domain.attrs)}

    # --- Main Iteration Loop ---
    for i in range(num_iterations):
        current_alpha = _calculate_new_learning_rate(alpha, i, decay_schedule,
                                                     decay_rate, decay_step)
        current_beta = _calculate_new_learning_rate(beta, i, decay_schedule,
                                                    decay_rate, decay_step)
        logger.info(
            f"Iteration {i+1}/{num_iterations}, alpha={current_alpha:.4f}, beta={current_beta:.4f}"
        )

        np.random.shuffle(current_synthetic_data)

        for target_marginal in target_marginals:
            if current_synthetic_data.shape[0] == 0:
                logger.warning("Dataset is empty, skipping marginal update.")
                continue

            # --- Calculate Current Marginal (Vectorized) ---
            marginal_attrs = target_marginal.domain.attrs
            col_indices = [attr_to_col_idx[attr] for attr in marginal_attrs]

            # Note: Using a simplified way to get the current marginal that avoids
            # creating a full JaxDataset object each time, for performance.
            # This is equivalent to what dataset.project(marginal_attrs) would do.
            marginal_data = current_synthetic_data[:, col_indices]
            flat_indices = np.ravel_multi_index(marginal_data.T,
                                                target_marginal.domain.shape)
            current_counts = np.bincount(flat_indices,
                                         minlength=target_marginal.values.size)
            current_marginal_values = current_counts.reshape(
                target_marginal.domain.shape)

            # --- Calculate Difference and Identify Cells for Update ---
            difference = target_marginal.values - current_marginal_values

            # --- Additions (Vectorized) ---
            under_counted_mask = difference > 0
            if np.any(under_counted_mask):
                add_budget = current_alpha * target_n_records
                counts_from_diff = difference[under_counted_mask]
                # Use astype(int) to truncate (floor), matching the prototype's int()
                to_add_counts = np.minimum(counts_from_diff,
                                           add_budget).astype(int)

                under_counted_indices = np.transpose(
                    np.nonzero(under_counted_mask))

                records_to_add_list = []
                for multi_idx, num_to_add in zip(under_counted_indices,
                                                 to_add_counts):
                    if num_to_add > 0:
                        matching_mask = (marginal_data == multi_idx).all(axis=1)
                        matching_indices = np.where(matching_mask)[0]
                        if len(matching_indices) > 0:
                            # Sample from existing records that match the under-counted cell
                            # This preserves the conditional distribution of other attributes.
                            records_to_duplicate_indices = np.random.choice(
                                matching_indices, size=num_to_add, replace=True)
                            records_to_add_list.append(current_synthetic_data[
                                records_to_duplicate_indices])
                        else:
                            # Synthesize a new record for this cell since none exist
                            new_record = np.zeros(n_attributes, dtype=int)

                            # Get the attributes and their values for the current marginal
                            marginal_attrs_for_add = target_marginal.domain.attrs
                            marginal_attr_indices = [attr_to_col_idx[attr] for attr in marginal_attrs_for_add]

                            for attr_idx, val in zip(marginal_attr_indices, multi_idx):
                                new_record[attr_idx] = val

                            # Fill in remaining attributes randomly
                            all_attr_indices = set(range(n_attributes))
                            remaining_attr_indices = list(all_attr_indices - set(marginal_attr_indices))
                            for attr_idx in remaining_attr_indices:
                                cardinality = full_domain.shape[attr_idx]
                                new_record[attr_idx] = np.random.randint(0, cardinality)

                            # Add the new record num_to_add times
                            records_to_add_list.extend([new_record] * num_to_add)

                if records_to_add_list:
                    all_records_to_add = np.vstack(records_to_add_list)
                    current_synthetic_data = np.vstack(
                        [current_synthetic_data, all_records_to_add])

            # --- Removals (Vectorized) ---
            over_counted_mask = difference < 0
            if np.any(over_counted_mask):
                remove_budget = current_beta * target_n_records
                counts_from_diff = -difference[over_counted_mask]
                # Use astype(int) to truncate (floor), matching the prototype's int()
                to_remove_counts = np.minimum(counts_from_diff,
                                              remove_budget).astype(int)

                over_counted_indices = np.transpose(
                    np.nonzero(over_counted_mask))

                indices_to_remove = []
                for multi_idx, num_to_remove in zip(over_counted_indices,
                                                    to_remove_counts):
                    if num_to_remove > 0:
                        matching_mask = (marginal_data == multi_idx).all(axis=1)
                        matching_indices = np.where(matching_mask)[0]
                        if len(matching_indices) > 0:
                            # Do not remove more records than available
                            num_to_remove = min(num_to_remove,
                                                len(matching_indices))
                            chosen_indices = np.random.choice(
                                matching_indices,
                                size=num_to_remove,
                                replace=False)
                            indices_to_remove.extend(chosen_indices)

                if indices_to_remove:
                    # Remove duplicates and update dataset
                    unique_indices_to_remove = np.unique(indices_to_remove)
                    current_synthetic_data = np.delete(current_synthetic_data,
                                                       unique_indices_to_remove,
                                                       axis=0)

    # --- Final Rescaling ---
    n_final = len(current_synthetic_data)
    if n_final != target_n_records:
        if n_final == 0:
            logger.error(
                "All records were removed during synthesis. "
                "Returning a resampled version of the original empty dataset. "
                "Consider using a smaller beta or more iterations.")
            # Cannot sample from an empty array, so we must check initial
            if initial_dataset.shape[0] == 0:
                return np.zeros((target_n_records, n_attributes), dtype=int)
            else:
                # Fallback to re-sampling from original
                indices = np.random.choice(initial_dataset.shape[0],
                                           target_n_records,
                                           replace=True)
                return initial_dataset[indices].astype(int)

        logger.info(
            f"Rescaling dataset from {n_final} to {target_n_records} records.")
        resample_indices = np.random.choice(n_final,
                                            target_n_records,
                                            replace=True)
        current_synthetic_data = current_synthetic_data[resample_indices]

    return current_synthetic_data.astype(int)
