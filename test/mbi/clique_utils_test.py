import unittest
import collections  # Import collections for defaultdict
from mbi.clique_utils import (
    reverse_clique_mapping,
    maximal_subset,
    clique_mapping,
    Clique,  # Import Clique type
)


class TestCliqueUtils(unittest.TestCase):

    def test_reverse_clique_mapping(self):
        # Doctest example
        maximal_cliques_1 = [("A", "B", "C"), ("C", "D")]
        all_cliques_1 = [("A", "B"), ("C",), ("D",), ("A", "C")]
        # Use defaultdict for easier comparison after sorting internal lists
        expected_mapping_1 = collections.defaultdict(
            list,
            {
                ("A", "B", "C"): [("A", "B"), ("C",), ("A", "C")],
                ("C", "D"): [("D",)],
            },
        )
        result_1 = reverse_clique_mapping(maximal_cliques_1, all_cliques_1)
        # Sort internal lists for comparison consistency
        result_1_sorted = collections.defaultdict(list)
        for k, v in result_1.items():
            result_1_sorted[k] = sorted(v)
        expected_mapping_1_sorted = collections.defaultdict(list)
        for k, v in expected_mapping_1.items():
            expected_mapping_1_sorted[k] = sorted(v)
        self.assertEqual(
            result_1_sorted,
            expected_mapping_1_sorted,
            "Doctest example failed",
        )

        # Non-overlapping maximal cliques
        maximal_cliques_2 = [("A", "B"), ("C", "D")]
        all_cliques_2 = [("A",), ("B",), ("C",), ("D",)]
        expected_mapping_2 = collections.defaultdict(
            list, {("A", "B"): [("A",), ("B",)], ("C", "D"): [("C",), ("D",)]}
        )
        result_2 = reverse_clique_mapping(maximal_cliques_2, all_cliques_2)
        result_2_sorted = collections.defaultdict(list)
        for k, v in result_2.items():
            result_2_sorted[k] = sorted(v)
        expected_mapping_2_sorted = collections.defaultdict(list)
        for k, v in expected_mapping_2.items():
            expected_mapping_2_sorted[k] = sorted(v)
        self.assertEqual(
            result_2_sorted,
            expected_mapping_2_sorted,
            "Non-overlapping cliques failed",
        )

        # All cliques contained in one maximal clique
        maximal_cliques_3 = [("A", "B", "C", "D")]
        all_cliques_3 = [("A", "B"), ("C",), ("B", "D")]
        expected_mapping_3 = collections.defaultdict(
            list, {("A", "B", "C", "D"): [("A", "B"), ("C",), ("B", "D")]}
        )
        result_3 = reverse_clique_mapping(maximal_cliques_3, all_cliques_3)
        result_3_sorted = collections.defaultdict(list)
        for k, v in result_3.items():
            result_3_sorted[k] = sorted(v)
        expected_mapping_3_sorted = collections.defaultdict(list)
        for k, v in expected_mapping_3.items():
            expected_mapping_3_sorted[k] = sorted(v)
        self.assertEqual(
            result_3_sorted,
            expected_mapping_3_sorted,
            "Single maximal clique failed",
        )

        # Empty all_cliques list
        maximal_cliques_4 = [("A", "B"), ("C", "D")]
        all_cliques_4 = []
        expected_mapping_4 = {("A", "B"): [], ("C", "D"): []}
        # Convert result to regular dict for comparison if empty
        self.assertEqual(
            dict(reverse_clique_mapping(maximal_cliques_4, all_cliques_4)),
            expected_mapping_4,
            "Empty all_cliques list failed",
        )

        # Empty maximal_cliques list
        maximal_cliques_5 = []
        all_cliques_5 = [("A",), ("B",)]
        expected_mapping_5 = {}
        self.assertEqual(
            reverse_clique_mapping(maximal_cliques_5, all_cliques_5),
            expected_mapping_5,
            "Empty maximal_cliques list failed",
        )

        # Empty both lists
        self.assertEqual(reverse_clique_mapping([], []), {}, "Empty lists failed")

    def test_maximal_subset(self):
        # Doctest example
        cliques_1 = [("A", "B"), ("B",), ("C",), ("B", "A")]
        expected_subsets_1 = {("A", "B"), ("C",)}
        result_1 = maximal_subset(cliques_1)
        # Convert result to set for comparison as order doesn't matter
        self.assertEqual(set(result_1), expected_subsets_1, "Doctest example failed")

        # Nested cliques
        cliques_2 = [("A", "B", "C"), ("A", "B"), ("A",)]
        expected_subsets_2 = {("A", "B", "C")}
        result_2 = maximal_subset(cliques_2)
        self.assertEqual(set(result_2), expected_subsets_2, "Nested cliques failed")

        # No nested cliques
        cliques_3 = [("A", "B"), ("C", "D"), ("E", "F")]
        expected_subsets_3 = {("A", "B"), ("C", "D"), ("E", "F")}
        result_3 = maximal_subset(cliques_3)
        self.assertEqual(set(result_3), expected_subsets_3, "No nested cliques failed")

        # Mixed nested and non-nested
        cliques_4 = [("A", "B", "C"), ("A", "B"), ("D", "E"), ("D",)]
        expected_subsets_4 = {("A", "B", "C"), ("D", "E")}
        result_4 = maximal_subset(cliques_4)
        self.assertEqual(
            set(result_4), expected_subsets_4, "Mixed nested/non-nested failed"
        )

        # Empty list
        cliques_5 = []
        expected_subsets_5 = set()
        result_5 = maximal_subset(cliques_5)
        self.assertEqual(set(result_5), expected_subsets_5, "Empty list failed")

        # Single clique
        cliques_6 = [("A",)]
        expected_subsets_6 = {("A",)}
        result_6 = maximal_subset(cliques_6)
        self.assertEqual(set(result_6), expected_subsets_6, "Single clique failed")

        # Duplicates
        cliques_7 = [("A", "B"), ("C",), ("A", "B"), ("C",)]
        expected_subsets_7 = {("A", "B"), ("C",)}
        result_7 = maximal_subset(cliques_7)
        self.assertEqual(set(result_7), expected_subsets_7, "Duplicate cliques failed")

    def test_clique_mapping(self):
        # Doctest example - test validity, not exact mapping due to order dependency
        maximal_cliques_1 = [("A", "B"), ("B", "C")]
        all_cliques_1 = [("B", "A"), ("B",), ("C",), ("B", "C")]
        result_1 = clique_mapping(maximal_cliques_1, all_cliques_1)
        self.assertEqual(
            len(result_1),
            len(all_cliques_1),
            "Doctest: Not all cliques were mapped",
        )  # Ensure all cliques are mapped
        for cl, max_cl in result_1.items():
            self.assertIn(
                max_cl,
                maximal_cliques_1,
                f"Doctest: Mapped maximal clique {max_cl} not in input maximal_cliques for {cl}",
            )
            self.assertTrue(
                set(cl) <= set(max_cl),
                f"Doctest: Clique {cl} is not a subset of its mapped maximal clique {max_cl}",
            )

        # More complex example - test validity
        maximal_cliques_2 = [("A", "B", "C"), ("C", "D", "E"), ("F",)]
        all_cliques_2 = [("A",), ("B", "C"), ("D", "E"), ("F",), ("C", "D")]
        result_2 = clique_mapping(maximal_cliques_2, all_cliques_2)
        self.assertEqual(
            len(result_2),
            len(all_cliques_2),
            "Complex: Not all cliques were mapped",
        )
        for cl, max_cl in result_2.items():
            self.assertIn(
                max_cl,
                maximal_cliques_2,
                f"Complex: Mapped maximal clique {max_cl} not in input maximal_cliques for {cl}",
                )
            self.assertTrue(
                set(cl) <= set(max_cl),
                f"Complex: Clique {cl} is not a subset of its mapped maximal clique {max_cl}",
                )
        # Check specific expected mappings based on subset rule (can check containment)
        self.assertTrue(set(("A",)) <= set(result_2[("A",)]))
        self.assertTrue(set(("B", "C")) <= set(result_2[("B", "C")]))
        self.assertTrue(set(("D", "E")) <= set(result_2[("D", "E")]))
        self.assertTrue(set(("F",)) <= set(result_2[("F",)]))
        self.assertTrue(set(("C", "D")) <= set(result_2[("C", "D")]))

        # Empty all_cliques list
        maximal_cliques_3 = [("A", "B"), ("C", "D")]
        all_cliques_3 = []
        expected_mapping_3 = {}
        self.assertEqual(
            clique_mapping(maximal_cliques_3, all_cliques_3),
            expected_mapping_3,
            "Empty all_cliques list failed",
        )

        # Empty maximal_cliques list
        maximal_cliques_4 = []
        all_cliques_4 = [("A",), ("B",)]
        expected_mapping_4 = {}  # Expect empty as no maximal clique can contain anything
        self.assertEqual(
            clique_mapping(maximal_cliques_4, all_cliques_4),
            expected_mapping_4,
            "Empty maximal_cliques list failed",
        )

        # Empty both lists
        self.assertEqual(clique_mapping([], []), {}, "Empty lists failed")


if __name__ == "__main__":
    unittest.main()
