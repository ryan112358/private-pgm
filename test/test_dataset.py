import unittest
from mbi.domain import Domain
from mbi.dataset import Dataset


class TestDomain(unittest.TestCase):
    def setUp(self):
        attrs = ["a", "b", "c", "d"]
        shape = [3, 4, 5, 6]
        domain = Domain(attrs, shape)
        self.data = Dataset.synthetic(domain, 100)

    def test_project(self):
        proj = self.data.project(["a", "b"])
        ans = Domain(["a", "b"], [3, 4])
        self.assertEqual(proj.domain, ans)
        proj = self.data.project(("a", "b"))
        self.assertEqual(proj.domain, ans)
        proj = self.data.project("c")
        self.assertEqual(proj.domain, Domain(["c"], [5]))

    def test_datavector(self):
        vec = self.data.datavector()
        self.assertTrue(vec.size, 3 * 4 * 5 * 6)


if __name__ == "__main__":
    unittest.main()
