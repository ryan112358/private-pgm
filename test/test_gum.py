import numpy as np
import pytest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.gum import synthesize_gum


@pytest.fixture
def gum_setup():
    """Provides a standard setup for GUM tests."""
    domain = Domain(["A", "B", "C"], [2, 2, 2])
    initial_dataset = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])

    # Target marginals that are different from the initial uniform distribution
    # Target: More (0,0) in AB, More (1,1) in BC
    target_ab_values = np.array([[4, 1], [1, 2]])  # Sums to 8
    target_bc_values = np.array([[1, 2], [1, 4]])  # Sums to 8

    target_marginals = [
        Factor(domain.project(["A", "B"]), target_ab_values),
        Factor(domain.project(["B", "C"]), target_bc_values),
    ]
    return initial_dataset, target_marginals, domain


def get_total_error(dataset, target_marginals, domain):
    """Helper to calculate total L1 error for a set of marginals."""
    total_error = 0
    attr_to_col_idx = {attr: i for i, attr in enumerate(domain.attrs)}
    for target in target_marginals:
        cols = [attr_to_col_idx[attr] for attr in target.domain.attrs]
        current_marginal_data = dataset[:, cols]
        flat_indices = np.ravel_multi_index(current_marginal_data.T,
                                            target.domain.shape)
        current_counts = np.bincount(flat_indices, minlength=target.values.size)
        current_marginal_values = current_counts.reshape(target.domain.shape)
        total_error += np.sum(np.abs(target.values - current_marginal_values))
    return total_error


def test_synthesize_gum_error_reduction(gum_setup):
    """Tests that GUM reduces the L1 error on the target marginals."""
    initial_dataset, target_marginals, domain = gum_setup

    initial_error = get_total_error(initial_dataset, target_marginals, domain)

    final_dataset = synthesize_gum(
        initial_dataset,
        target_marginals,
        num_iterations=50,
        alpha=0.1,
        beta=0.1,
        verbose=False,
    )

    final_error = get_total_error(final_dataset, target_marginals, domain)

    # With a stochastic algorithm, we can't guarantee strict decrease in every run.
    # But the error should not significantly increase.
    assert final_error <= initial_error


def test_synthesize_gum_record_count(gum_setup):
    """Tests that the output dataset has the specified number of records."""
    initial_dataset, target_marginals, _ = gum_setup
    target_count = 100
    final_dataset = synthesize_gum(
        initial_dataset,
        target_marginals,
        record_count=target_count,
        num_iterations=5,
        verbose=False,
    )
    assert final_dataset.shape[0] == target_count


def test_synthesize_gum_invalid_schedule(gum_setup):
    """Tests that an invalid decay schedule raises a ValueError."""
    initial_dataset, target_marginals, _ = gum_setup
    with pytest.raises(ValueError, match="Unknown decay_schedule"):
        synthesize_gum(
            initial_dataset,
            target_marginals,
            decay_schedule="invalid_schedule",
            verbose=False,
        )


def test_synthesize_gum_empty_initial_dataset(gum_setup):
    """Tests that the function can run with an empty initial dataset."""
    _, target_marginals, domain = gum_setup
    empty_dataset = np.zeros((0, len(domain.attrs)), dtype=int)
    target_count = 50

    final_dataset = synthesize_gum(
        empty_dataset,
        target_marginals,
        record_count=target_count,
        num_iterations=10,
        alpha=0.8,  # Higher alpha to ensure records are added
        verbose=False,
    )

    # The behavior for an empty dataset is to add records based on under-counted cells
    # but since no records exist to duplicate, it might not be able to add any.
    # The final rescaling should still produce the correct number of records,
    # potentially by resampling an empty set if no records are ever added.
    # The implementation falls back to resampling the (empty) initial dataset.
    # Let's check if the final output has the correct shape.
    assert final_dataset.shape == (target_count, len(domain.attrs))
