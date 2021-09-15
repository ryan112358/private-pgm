import unittest
from mbi import Domain, Factor, FactorGraph, CliqueVector
import numpy as np

class TestFactorGraph(unittest.TestCase):

    def test_tree(self):
        dom = Domain(['A','B','C','D'], [2,3,4,5])
        cliques = [('A','B'), ('B','C'), ('C','D')]

        potentials = CliqueVector.random(dom, cliques)*3
        model = FactorGraph(dom, cliques, convex=False, iters=100)
        marginals = model.belief_propagation(potentials)

        logP = sum(potentials.values())
        P = (logP - logP.logsumexp()).exp()

        for cl in cliques:
            exact = P.project(cl).datavector()
            est = marginals[cl].datavector()
            np.testing.assert_almost_equal(exact, est)

    def test_cycle(self):
        dom = Domain(['A','B','C'], [2,3,4])
        cliques = [('A','B'), ('B','C'), ('A','C')]

        potentials = CliqueVector.random(dom, cliques)*3
        model = FactorGraph(dom, cliques, convex=False, iters=100)
        model.potentials = potentials

        marginals = model.belief_propagation(potentials)

        # check internal consistency
        for a in dom:
            ans = model.project(a).datavector()
            for cl in cliques:
                if a in cl:
                    est = marginals[cl].project(a).datavector()
                    np.testing.assert_almost_equal(ans, est)

        logP = sum(potentials.values())
        P = (logP - logP.logsumexp()).exp()
        true_marginals = {}
        for cl in cliques:
            true_marginals[cl] = P.project(cl)

        true_energy = model.energy_functional(potentials, true_marginals)[0]
        est_energy = model.energy_functional(potentials, marginals)[0]
       
        print(true_energy, est_energy)
        # Assumes LBP converged to minmium of Bethe energy
        # which is a convex objective when there is one cycle
        self.assertTrue(est_energy < true_energy) 

    def test_project(self):
        dom = Domain(['A','B','C'], [2,3,4])
        cliques = [('A','B'), ('B','C')]

        potentials = CliqueVector.random(dom, cliques)*3
        model = FactorGraph(dom, cliques, convex=False, iters=100)
        model.potentials = potentials
        model.marginals = model.belief_propagation(potentials)
            
        AC1 = model.project(('A','C')).datavector()

        cliques.append(('A','C'))
        potentials[('A','C')] = Factor.zeros(dom.project(('A','C')))
        model = FactorGraph(dom, cliques, convex=False, iters=100)
        marginals = model.belief_propagation(potentials)
        AC2 = marginals[('A','C')].datavector()

        np.testing.assert_almost_equal(AC1, AC2)
 

    def test_convex_tree(self):
        dom = Domain(['A','B','C','D','E','F'], [2,3,2,3,2,3])
        cliques = [('A','B'), ('B','C'), ('C','D'), ('C','E'), ('C','F')]
        potentials = CliqueVector.random(dom, cliques)*3
        model = FactorGraph(dom, cliques, convex=True, iters=100)
        marginals = model.belief_propagation(potentials)

        logP = sum(potentials.values())
        P = (logP - logP.logsumexp()).exp()

        for cl in cliques:
            exact = P.project(cl).datavector()
            est = marginals[cl].datavector()
            np.testing.assert_almost_equal(exact, est, decimal=5)

    def test_convex_cycles(self):
        dom = Domain(['A','B','C','D','E'], [2,3,2,3,2])
        cliques = [('A', 'E'), ('A', 'D'), ('B', 'D'), ('B', 'E'), ('C', 'D'), \
                    ('D', 'E'), ('C', 'E'), ('B', 'C')]
        
        potentials = CliqueVector.random(dom, cliques)*3
        model = FactorGraph(dom, cliques, convex=True, iters=100)
        marginals = model.belief_propagation(potentials)

        # check internal consistency
        for a in dom:
            ans = model.project(a).datavector()
            for cl in cliques:
                if a in cl:
                    est = marginals[cl].project(a).datavector()
                    np.testing.assert_almost_equal(ans, est, decimal=5)

        logP = sum(potentials.values())
        P = (logP - logP.logsumexp()).exp()
        true_marginals = {}
        for cl in cliques:
            true_marginals[cl] = P.project(cl)

        true_energy = model.energy_functional(potentials, true_marginals)[0]
        est_energy = model.energy_functional(potentials, marginals)[0]
        self.assertTrue(est_energy <= true_energy) 

        cvx_marginals = model.optimize_bethe(potentials, backend='cvxopt')
        cvx_energy = model.energy_functional(potentials, cvx_marginals)[0]
        print('Energies', true_energy, est_energy, cvx_energy)
        for cl in cliques:
            x = cvx_marginals[cl].datavector()
            y = marginals[cl].datavector()
            np.testing.assert_almost_equal(x,y,decimal=4)

    def test_bethe(self):
        # Here we try to show that theta = gradient(Bethe-Entropy(mu)), where mu = Belief-Prop(theta)
        dom = Domain(['A','B','C','D','E'], [2,3,2,3,2])
        cliques = [('A', 'E'), ('A', 'D'), ('B', 'D'), ('B', 'E'), ('C', 'D'), \
                    ('D', 'E'), ('C', 'E'), ('B', 'C')]

        potentials = CliqueVector.random(dom, cliques)*3
        model = FactorGraph(dom, cliques, convex=True, iters=100)
        mu = model.belief_propagation(potentials)

        f, df = model.bethe_entropy(mu)
        mu2 = model.belief_propagation(-1*df)
        for cl in cliques:
            x = mu[cl].datavector()
            y = mu2[cl].datavector()
            np.testing.assert_almost_equal(x,y,decimal=4)

if __name__ == '__main__':
    unittest.main()
