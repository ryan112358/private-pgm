import unittest
from mbi.domain import Domain
from mbi.inference import FactoredInference
import numpy as np

class TestInference(unittest.TestCase):

    def setUp(self):
        attrs = ['a','b','c','d', 'e']
        shape = [2,3,4,5,6]
        domain = Domain(attrs, shape)

        #x = np.random.rand(*shape)

        self.measurements = []
        for i in range(4):
            I = np.eye(shape[i])
            y = np.random.rand(shape[i])
            y /= y.sum()
            self.measurements.append( (I, y, 1.0, attrs[i]) )

        self.engine = FactoredInference(domain, log=True)

    def test_mirror_descent(self):
        loss = self.engine.mirror_descent(self.measurements, 1.0)
        self.assertEqual(self.engine.model.total, 1.0)
        self.assertTrue(loss <= 1e-4)

    def test_dual_averaging(self):
        loss = self.engine.dual_averaging(self.measurements, 1.0)
        self.assertEqual(self.engine.model.total, 1.0)
        #self.assertTrue(loss <= 1e-5)

    def test_lbfgs(self):
        loss = self.engine.lbfgs(self.measurements, 1.0)
        self.assertEqual(self.engine.model.total, 1.0)
        self.assertTrue(loss <= 1e-6)
        #self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
