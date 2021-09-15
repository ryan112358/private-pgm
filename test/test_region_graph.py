import unittest
from mbi import Domain, Factor, FactorGraph, RegionGraph, CliqueVector
import numpy as np

class TestRegionGraph(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(0)

    def test_pairwise(self):
        dom = Domain(['A','B','C','D','E'], [2,3,2,3,2])
        cliques = [('A', 'E'), ('A', 'D'), ('B', 'D'), ('B', 'E'), ('C', 'D'), \
                    ('D', 'E'), ('C', 'E'), ('B', 'C')]

        potentials = CliqueVector.random(dom, cliques, self.prng)*3
        model = FactorGraph(dom, cliques, convex=False, iters=100)
        mu1 = model.belief_propagation(potentials)


        model = RegionGraph(dom, cliques, convex=False, iters=100)
        mu2 = model.belief_propagation(potentials)

        for cl in cliques:
            x = mu1[cl].datavector()
            y = mu2[cl].datavector()
            np.testing.assert_almost_equal(x,y)

    def test_junctiontree(self):
        dom = Domain(['A','B','C','D','E'], [2,3,2,3,2])
        cliques = [('A','B','C'), ('B','C','D'), ('C','D','E')]
                
        potentials = CliqueVector.random(dom, cliques, self.prng)*3
        model = RegionGraph(dom, cliques, iters=100, minimal=False, convex=False)
        mu = model.belief_propagation(potentials)

        logP = sum(potentials.values())
        P = (logP - logP.logsumexp()).exp()
        for cl in cliques:
            x = mu[cl].datavector()
            y = P.project(cl).datavector()
            np.testing.assert_almost_equal(x,y)

    def test_structure(self):
        dom = Domain(['A','B','C','D'], [2,3,4,5])
        cliques = [('A','B','C'), ('B','C','D'), ('A','C','D')]
        model = RegionGraph(dom, cliques, minimal=False, convex=False, iters=100)

        # Figure 11.9 from Koller & Friedman
        self.assertEqual(model.regions, set(cliques + [('B','C'), ('A','C'), ('C','D'), ('C',)]))

        # Example 11.5 from Koller & Friedman
        self.assertEqual(set(model.B[('A','B','C')]), { (('B','C','D'), ('B','C')), 
                                                        (('A','C','D'), ('A','C')),
                                                        (('C','D'), ('C',)) })
        self.assertEqual(set(model.B[('B','C','D')]), { (('A','B','C'),('B','C')),
                                                        (('A','C','D'),('C','D')),
                                                        (('A','C'), ('C',)) })
        self.assertEqual(set(model.B[('A','C','D')]), { (('A','B','C'),('A','C')),
                                                        (('B','C','D'),('C','D')),
                                                        (('B','C'), ('C',)) })
        self.assertEqual(set(model.B[('B','C')]), { (('A','B','C'), ('B','C')),
                                                    (('B','C','D'), ('B','C')),
                                                    (('A','C'), ('C',)),
                                                    (('C','D'), ('C',)) })
        self.assertEqual(set(model.B[('A','C')]), { (('A','B','C'),('A','C')),
                                                    (('A','C','D'),('A','C')),
                                                    (('B','C'),('C',)),
                                                    (('C','D'),('C',)) })
        self.assertEqual(set(model.B[('C','D')]), { (('B','C','D'), ('C','D')),
                                                    (('A','C','D'), ('C','D')),
                                                    (('B','C'), ('C',)),
                                                    (('A','C'), ('C',)) })
        self.assertEqual(set(model.B[('C',)]), {    (('B','C'), ('C',)),
                                                    (('A','C'), ('C',)),
                                                    (('C','D'), ('C',)) })
                                                    
        self.assertEquals(set(model.N[(('B','C'),('C',))]), { (('A','B','C'), ('B','C')),
                                                            (('B','C','D'), ('B','C')) })
        self.assertEquals(set(model.D[(('B','C'),('C',))]), set())

        self.assertEquals(set(model.N[(('A','B','C'), ('B','C'))]), { (('A','C','D'), ('A','C')) })
        self.assertEquals(set(model.D[(('A','B','C'), ('B','C'))]), { (('A','C'), ('C',)) })
                                                            
        # Check internal consistency
        potentials = CliqueVector.random(dom, cliques, self.prng)*3
        mu = model.belief_propagation(potentials)

        for r in cliques:
            np.testing.assert_almost_equal(mu[r].sum(),1.0)
            for s in cliques:
                if r == s: break
                t = tuple(set(r) & set(s))
                if len(t) > 0: 
                    x = mu[r].project(t).datavector()
                    y = mu[s].project(t).datavector()
                    np.testing.assert_almost_equal(x,y)
        
    def test_general(self, convex=None, minimal=None):
        if convex is None:
            self.test_general(False)
            self.test_general(True)
            return
        if minimal is None:
            self.test_general(convex, False)
            self.test_general(convex, True)
            return 
        dom = Domain(['A','B','C','D','E','F'], [2,3,2,3,2,3]) 
        cliques = [('A', 'E', 'F'), ('A', 'C', 'E'), ('B', 'C', 'D'), ('C', 'D'), ('B', 'C', 'E'), ('D', 'E', 'F'), ('E', 'F'), ('A', 'B'), ('B', 'C'), ('B', 'E', 'F')]

        model = RegionGraph(dom, cliques, convex=convex, minimal=minimal, iters=250)
        potentials = CliqueVector.random(dom, model.cliques, self.prng)*3
        mu = model.belief_propagation(potentials)

        # check internal consistency
        for r in model.cliques:
            np.testing.assert_almost_equal(mu[r].sum(),1.0)
            for s in model.cliques:
                if r == s: break
                t = tuple(set(r) & set(s))
                if len(t) > 0: 
                    x = mu[r].project(t).datavector()
                    y = mu[s].project(t).datavector()
                    print(r,s,t,np.linalg.norm(x-y,1))
                    np.testing.assert_almost_equal(x,y, decimal=2)


    def test_minimality(self, convex=None, test=None):
        if convex is None and test is None:
            self.test_minimality(False, 'easy')
            self.test_minimality(True, 'easy')
            self.test_minimality(True, 'hard')
            # Note, non-convex generalized BP may not give same results with minimal graph.
            # Both message upates have the same fixed points, but the algorithms have different
            # dynamics.  Moreover, neither are guaranteed to converge to said fixed point.
            return

        if test == 'easy':
            dom = Domain(['A','B','C','D'], [2,3,2,3]) 
            cliques = [('A','B','C'), ('B', 'C', 'D'), ('A', 'C', 'D')]
        if test == 'hard':
            dom = Domain(['A','B','C','D','E','F'], [2,3,2,3,2,3]) 
            cliques = [('D', 'F'), ('A', 'C', 'D', 'F'), ('C', 'D', 'E', 'F'), ('A', 'E', 'F'), ('A', 'E'), ('B', 'C', 'D', 'F'), ('C', 'E'), ('B', 'F'), ('B', 'D', 'E', 'F'), ('A', 'C', 'E', 'F')]
   

        model1 = RegionGraph(dom, cliques, convex=convex, minimal=False, iters=250)
        model2 = RegionGraph(dom, cliques, convex=convex, minimal=True, iters=250)
        
        self.assertEqual(model1.cliques,  model2.cliques)
        pot = CliqueVector.random(dom, model1.cliques, self.prng)*3

        mu1 = model1.belief_propagation(pot)
        mu2 = model2.belief_propagation(pot)
        #from IPython import embed; embed()

        for cl in model1.cliques:
            x = mu1[cl].datavector()
            y = mu2[cl].datavector()
            print(cl, np.linalg.norm(x-y, 1))
            self.assertTrue(np.linalg.norm(x-y,1) <= 1e-3) #np.testing.assert_almost_equal(x,y)
        #self.assertTrue(False)


    def test_convex(self, minimal=True, test='hard'):
        if minimal is None:
            self.test_convex(False)
            self.test_convex(True)
            return

        if test == 'easy':
            dom = Domain(['A','B','C'], [2,3,2]) 
            cliques = [('A','B'), ('B', 'C'), ('A', 'C')]
        else: 
            dom = Domain(['A','B','C','D','E'], [2,3,2,3,2]) 
            cliques = [('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'B', 'E'), ('A', 'C', 'D'), ('A', 'C', 'E'), ('A', 'D', 'E'), ('B', 'C', 'D'), ('B', 'C', 'E'), ('B', 'D', 'E'), ('C', 'D', 'E')]

        model = RegionGraph(dom, cliques, convex=True, minimal=minimal, iters=100)
        pot = CliqueVector.random(dom, model.cliques, self.prng)*3


        mu = model.optimize_kikuchi(pot, backend='cvxopt')

        mu1 = model.wiegerinck(pot)
        mu2 = model.loh_wibisono(pot)
        mu3 = model.hazan_peng_shashua(pot)

        logP = sum(pot.values())
        P = (logP - logP.logsumexp()).exp()
        mu0 = {}
        for cl in model.cliques:
            mu0[cl] = P.project(cl)
        
        for cl in model.cliques:
            w = mu[cl].datavector()
            x = mu1[cl].datavector()
            y = mu2[cl].datavector()
            z = mu3[cl].datavector()
            print(cl); print(w); print(x); print(y); print(z); print()
            self.assertTrue(np.linalg.norm(w-x,1) <= 1e-3)
            self.assertTrue(np.linalg.norm(w-y,1) <= 1e-3)
            self.assertTrue(np.linalg.norm(w-z,1) <= 1e-3)

        def expand(mu):
            for r in model.regions:
                if not r in model.cliques:
                    for cl in model.cliques:
                        if set(r) < set(cl):
                            mu[r] = mu[cl].project(r)
            return mu

        f = model.energy_functional(pot, expand(mu))[0]
        f0 = model.energy_functional(pot, expand(mu0))[0]
        f1 = model.energy_functional(pot, expand(mu1))[0]
        f2 = model.energy_functional(pot, expand(mu2))[0]
        f3 = model.energy_functional(pot, expand(mu3))[0]
        print(f,f0,f1,f2,f3)
        self.assertTrue(f < f0)
        self.assertTrue(f1 < f0)
        self.assertTrue(f2 < f0)
        self.assertTrue(f3 < f0)
