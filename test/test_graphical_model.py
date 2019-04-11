import unittest
from mbi.domain import Domain
from mbi.factor import Factor
from mbi.graphical_model import GraphicalModel, CliqueVector
import numpy as np

class TestGraphicalModel(unittest.TestCase):

    def setUp(self):
        attrs = ['a','b','c','d']
        shape = [2,3,4,5]
        domain = Domain(attrs, shape)
        cliques = [('a','b'), ('b','c'),('c','d')]
        self.model = GraphicalModel(domain, cliques)

    def test_datavector(self):
        x = self.model.datavector().flatten()
        ans = np.ones(2*3*4*5) / (2*3*4*5)
        self.assertTrue(np.allclose(x, ans))

    def test_project(self):
        model = self.model.project(['d','a'])
        x = model.datavector()
        ans = np.ones(2*5) / 10.0
        self.assertEqual(x.size, 10)
        self.assertTrue(np.allclose(x.flatten(), ans))

        model = self.model
        pot = { cl : Factor.random(model.domain.project(cl)) for cl in model.cliques }
        model.potentials = CliqueVector(pot)
        
        x = model.datavector().reshape(2,3,4,5)
        y0 = x.sum(axis=(2,3)).flatten()
        y1 = model.project(['a','b']).datavector() 
        self.assertEqual(y0.size, y1.size)
        self.assertTrue(np.allclose(y0, y1))

    def test_calculate_many_marginals(self):
        proj = [[],['a'],['b'],['c'],['d'],['a','b'],['a','c'],['a','d'],['b','c'],
                ['b','d'],['c','d'],['a','b','c'],['a','b','d'],['a','c','d'],['b','c','d'],
                ['a','b','c','d']]
        proj = [tuple(p) for p in proj]
        model = self.model
        model.total = 10.0
        pot = { cl : Factor.random(model.domain.project(cl)) for cl in model.cliques }
        model.potentials = CliqueVector(pot)
        
        results = model.calculate_many_marginals(proj)
        for pr in proj:
            ans = model.project(pr).values
            close = np.allclose(results[pr].values, ans)
            print(pr, close, results[pr].values, ans)
            self.assertTrue(close)

    def test_krondot(self):
        model = self.model
        pot = { cl : Factor.random(model.domain.project(cl)) for cl in model.cliques }
        model.potentials = CliqueVector(pot)
 
        A = np.ones((1,2))
        B = np.eye(3)
        C = np.ones((1,4))
        D = np.eye(5)
        res = model.krondot([A,B,C,D])
        x = model.datavector().reshape(2,3,4,5)
        ans = x.sum(axis=(0,2), keepdims=True)
        self.assertEqual(res.shape, ans.shape)
        self.assertTrue(np.allclose(res, ans))

    def test_dict_to_vector(self):
        rand = np.random.rand(2*3 + 3*4 + 4*5)

        tmp1 = self.model.vector_to_dict(rand)
        tmp2 = self.model.dict_to_vector(tmp1)
        
        self.assertTrue(np.allclose(tmp2, rand))

    def test_belief_prop(self):
        pot = self.model.potentials
        self.model.total = 10
        mu, logZ, cache = self.model.belief_propagation(pot)

        for key in mu:
            ans = self.model.total/np.prod(mu[key].domain.shape)
            self.assertTrue(np.allclose(mu[key].values, ans))

        pot = { cl : Factor.random(pot[cl].domain) for cl in pot }
        mu, logZ, cache = self.model.belief_propagation(pot)

        logp = sum(pot.values())
        logp -= logp.logsumexp()
        dist = logp.exp() * self.model.total

        for key in mu:
            ans = dist.project(key).values  
            res = mu[key].values
            self.assertTrue(np.allclose(ans, res))

    def test_back_belief_prop2(self):
        self.model.total = 10.0
        
        def loss(pot):
            mu, logZ, cache = self.model.belief_propagation(pot)
            dpot = self.model.back_belief_propagation(mu, cache)
            ans = 0.5 * sum((mu[cl]*mu[cl]).sum() for cl in mu)
            return ans, dpot

        pot = self.model.potentials
        pot = { cl : Factor.random(pot[cl].domain) for cl in pot }
        _, exact = loss(pot)
        approx = { cl : Factor.zeros(pot[cl].domain) for cl in pot }
        eps = 1e-5
        for cl in pot:
            vals = pot[cl].values
            for i in range(vals.size):
                curr = vals.item(i)
                vals.itemset(i, curr + eps)
                f1, _ = loss(pot)
                vals.itemset(i, curr - eps)
                f0, _ = loss(pot)
                vals.itemset(i, curr)
                approx[cl].values.itemset(i, (f1 - f0) / (2*eps))
            self.assertTrue(np.allclose(exact[cl].values, approx[cl].values))

    def test_back_belief_prop(self):
        from scipy.optimize import check_grad
        self.model.total = 10.0
        pot = self.model.potentials
        mu, logZ, cache = self.model.belief_propagation(pot)
        dpot = self.model.back_belief_propagation(mu, cache)

        def loss(params):
            pot = self.model.vector_to_dict(params)
            mu, logZ, cache = self.model.belief_propagation(pot)
            dpot = self.model.back_belief_propagation(mu, cache)
           
            ans = 0.5*np.sum(self.model.dict_to_vector(mu)**2)
            grad = self.model.dict_to_vector(dpot)
            return ans, grad

        params = np.random.rand(2*3 + 3*4 + 4*5)
        _, exact = loss(params)
        approx = np.zeros_like(exact)
        eps = 1e-5
        for i in range(params.size):
            params[i] -= eps
            l0 = loss(params)[0]
            params[i] += 2*eps
            l1 = loss(params)[0]
            params[i] -= eps
            approx[i] = (l1 - l0) / (2*eps)
        print(exact)
        print(approx)
        #err = check_grad(lambda x: loss(x)[0], lambda x: loss(x)[1], params, epsilon=1e-5)
        print(np.linalg.norm(exact - approx))
        self.assertTrue(np.allclose(exact, approx))

if __name__ == '__main__':
    unittest.main()
