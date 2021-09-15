import numpy as np
from collections import defaultdict
from mbi import Domain, Factor, CliqueVector
from scipy.linalg import block_diag
from scipy import optimize
from functools import reduce
import itertools
import networkx as nx
from cvxopt import matrix, solvers
from disjoint_set import DisjointSet

class RegionGraph():
    def __init__(self,domain,cliques,total = 1.0,minimal=True,convex=True,iters=25,convergence=1e-3,damping=0.5):
        self.domain = domain
        self.cliques = cliques
        if not convex:
            self.cliques = []
            for r in cliques:
                if not any(set(r) < set(s) for s in cliques):
                    self.cliques.append(r)
        self.total = total
        self.minimal = minimal
        self.convex = convex
        self.iters = iters
        self.convergence = convergence
        self.damping = damping
        if convex:
            self.belief_propagation = self.wiegerinck
            self.belief_propagation = self.hazan_peng_shashua
        else:
            self.belief_propagation = self.generalized_belief_propagation 
        self.build_graph()
        self.cliques = sorted(self.regions, key=len)
        self.potentials = CliqueVector.zeros(domain, self.cliques)
        self.marginals = CliqueVector.uniform(domain, self.cliques)*total

    def show(self):
        import matplotlib.pyplot as plt
        labels = { r : ''.join(r) for r in self.regions }
        
        pos = {}
        xloc = defaultdict(lambda: 0)
        for r in sorted(self.regions): 
            y = len(r)
            pos[r] = (xloc[y]+0.5*(y%2), y)
            xloc[y] += 1

        colormap = { r : 'red' if r in self.cliques else 'blue' for r in self.regions }

        nx.draw(self.G, pos=pos, node_color='orange', node_size=1000)
        nx.draw(self.G, pos=pos, nodelist=self.cliques, node_color='green', node_size=1000)
        nx.draw_networkx_labels(self.G, pos=pos, labels=labels)
        plt.show()

    def project(self,attrs,maxiter=100,alpha=None):
        if type(attrs) is list:
            attrs = tuple(attrs)

        for cl in self.cliques:
            if set(attrs) <= set(cl):
                return self.marginals[cl].project(attrs)

        # Use multiplicative weights/entropic mirror descent to solve projection problem
        intersections = [set(cl)&set(attrs) for cl in self.cliques]
        target_cliques = [tuple(t) for t in intersections if not any(t < s for s in intersections)]
        target_cliques = list(set(target_cliques))
        target_mu = CliqueVector.from_data(self, target_cliques)

        if len(target_cliques) == 0:
            return Factor.uniform(self.domain.project(attrs))*self.total
        #P = Factor.uniform(self.domain.project(attrs))*self.total
        # Use a smart initialization 
        P = estimate_kikuchi_marginal(self.domain.project(attrs), self.total, target_mu)
        if alpha is None:
            # start with a safe step size
            alpha = 1.0 / (self.total*len(target_cliques))

        curr_mu = CliqueVector.from_data(P, target_cliques)
        diff = curr_mu - target_mu
        curr_loss, dL = diff.dot(diff), sum(diff.values()).expand(P.domain)
        begun = False

        for _ in range(maxiter):
            if curr_loss <= 1e-8:
                return P # stop early if marginals are almost exactly realized
            Q = P * (-alpha*dL).exp()
            Q *= self.total / Q.sum()
            curr_mu = CliqueVector.from_data(Q, target_cliques)
            diff = curr_mu - target_mu
            loss = diff.dot(diff)
            #print(alpha, diff.dot(diff))
   
            if curr_loss - loss >= 0.5*alpha*dL.dot(P-Q):
                P = Q
                curr_loss = loss
                dL = sum(diff.values()).expand(P.domain)
                # increase step size if we haven't already decreased it at least once
                if not begun: alpha *= 2 
            else:
                alpha *= 0.5
                begun = True
            
        return P

    def primal_feasibility(self, mu):
        ans = 0
        count = 0
        for r in self.cliques:
            for s in self.children[r]:
                x = mu[r].project(s).datavector()
                y = mu[s].datavector()
                err = np.linalg.norm(x-y, 1)
                ans += err
                count += 1
        return 0 if count==0 else ans/count

    def is_converged(self, mu):
        return self.primal_feasibility(mu) <= self.convergence

    def build_graph(self):
        # Alg 11.3 of Koller & Friedman
        regions = set(self.cliques)
        size = 0
        while len(regions) > size:
            size = len(regions)
            for r1, r2 in itertools.combinations(regions, 2):
                z = tuple(sorted(set(r1) & set(r2)))
                if len(z) > 0 and not z in regions:
                    regions.update({z})

        G = nx.DiGraph()
        G.add_nodes_from(regions)
        for r1 in regions:
            for r2 in regions:
                if set(r2) < set(r1) and not \
                    any(set(r2) < set(r3) and set(r3) < set(r1) for r3 in regions):
                    G.add_edge(r1, r2)

        H = G.reverse()
        G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

        self.children = { r : list(G.neighbors(r)) for r in regions }
        self.parents = { r : list(H.neighbors(r)) for r in regions }
        self.descendants = { r : list(G1.neighbors(r)) for r in regions }
        self.ancestors = { r : list(H1.neighbors(r)) for r in regions }
        self.forebears = { r : set([r] + self.ancestors[r]) for r in regions }
        self.downp = { r : set([r] + self.descendants[r]) for r in regions }

        if self.minimal:
            min_edges = []
            for r in regions:
                ds = DisjointSet()
                for u in self.parents[r]: ds.find(u)
                for u, v in itertools.combinations(self.parents[r], 2):
                    uv = set(self.ancestors[u]) & set(self.ancestors[v])
                    if len(uv) > 0: ds.union(u,v)
                canonical = set()
                for u in self.parents[r]:
                    canonical.update({ds.find(u)})
                #if len(canonical) > 1:# or r in self.cliques:
                min_edges.extend([(u,r) for u in canonical])
            #G = nx.DiGraph(min_edges)
            #regions = list(G.nodes)
            G = nx.DiGraph()
            G.add_nodes_from(regions)
            G.add_edges_from(min_edges)

            H = G.reverse()
            G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

            self.children = { r : list(G.neighbors(r)) for r in regions }
            self.parents = { r : list(H.neighbors(r)) for r in regions }
            #self.descendants = { r : list(G1.neighbors(r)) for r in regions }
            #self.ancestors = { r : list(H1.neighbors(r)) for r in regions }
            #self.forebears = { r : set([r] + self.ancestors[r]) for r in regions }
            #self.downp = { r : set([r] + self.descendants[r]) for r in regions }
 
        self.G = G
        self.regions = regions

        if self.convex: 
            self.counting_numbers = { r : 1.0 for r in regions }
        else:
            moebius = {}
            def get_counting_number(r):
                if not r in moebius: 
                    moebius[r] = 1 - sum(get_counting_number(s) for s in self.ancestors[r])
                return moebius[r]
            for r in regions: get_counting_number(r)
            self.counting_numbers = moebius

            if self.minimal:
                # https://people.eecs.berkeley.edu/~ananth/2002+/Payam/submittedkikuchi.pdf
                # Eq. 30 and 31
                N, D, B = {}, {}, {}
                for r in regions:
                    B[r] = set()
                    for p in self.parents[r]:
                        B[r].add((p,r))
                    for d in self.descendants[r]:
                        for p in set(self.parents[d]) - {r} - set(self.descendants[r]):
                            B[r].add((p,d))  

                for p in self.regions:
                    for r in self.children[p]:
                        N[p,r], D[p,r] = set(), set()
                        for s in self.parents[p]:
                            N[p,r].add((s,p))
                        for d in self.descendants[p]:
                            for s in set(self.parents[d]) - {p} - set(self.descendants[p]):
                                N[p,r].add((s,d))
                        for s in set(self.parents[r]) - {p}:
                            D[p,r].add((s,r))
                        for d in self.descendants[r]:
                            for p1 in set(self.parents[d]) - {r} - set(self.descendants[r]):
                                D[p,r].add((p1,d))
                        cancel = N[p,r] & D[p,r]
                        N[p,r] = N[p,r] - cancel
                        D[p,r] = D[p,r] - cancel

                self.N, self.D, self.B = N, D, B

            else:
                # From Yedida et al. for fully saturated region graphs
                # for sending messages ru --> rd and computing beliefs B_r
                N,D,B = {}, {}, {}
                for r in regions:
                    B[r] = [(ru, r) for ru in self.parents[r]]
                    for rd in self.descendants[r]:
                        for ru in set(self.parents[rd])-self.downp[r]:
                            B[r].append((ru, rd))

                for ru in regions:
                    for rd in self.children[ru]:
                        fu, fd = self.downp[ru], self.downp[rd]
                        cond = lambda r: not r[0] in fu and r[1] in (fu-fd)
                        N[ru, rd] = [e for e in G.edges if cond(e)]
                        cond = lambda r: r[0] in (fu-fd) and r[1] in fd and r != (ru,rd)
                        D[ru, rd] = [e for e in G.edges if cond(e)]

                self.N, self.D, self.B = N, D, B


        self.messages = {}
        self.message_order = []
        for ru in sorted(regions, key=len): #nx.topological_sort(H): # should be G or H?
            for rd in self.children[ru]:
                self.message_order.append((ru,rd))
                self.messages[ru,rd] = Factor.zeros(self.domain.project(rd))
                self.messages[rd,ru] = Factor.zeros(self.domain.project(rd)) # only for hazan et al

    def generalized_belief_propagation(self, potentials, callback=None):
        # https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/current/4paper/4-2.pdf
        pot = {}
        for r in self.regions:
            if r in self.cliques: pot[r] = potentials[r]
            else: pot[r] = Factor.zeros(self.domain.project(r))
           
        for _ in range(self.iters):
            new = {}
            for ru, rd in self.message_order:
                # Yedida et al. strongly recommend using updated messages for LHS (denom in our case)
                #num = sum(pot[c] for c in self.downp[ru] if c != rd)
                num = pot[ru]
                num = num + sum(self.messages[r1,r2] for r1, r2 in self.N[ru, rd])
                denom = sum(new[r1,r2] for r1,r2 in self.D[ru, rd])
                diff = tuple(set(ru) - set(rd))
                new[ru,rd] = num.logsumexp(diff) - denom
                new[ru,rd] -= new[ru,rd].logsumexp()

            #self.messages = new
            for ru, rd in self.message_order:
                self.messages[ru,rd] = 0.5*self.messages[ru,rd] + 0.5*new[ru,rd]
            #print(self.messages[ru,rd].datavector())
            #ru, rd = self.message_order[0]
            #print(ru, rd, self.messages[ru,rd].values)

        marginals = {}
        for r in self.cliques:
            #belief = sum(potentials[c] for c in self.downp[r]) + sum(self.messages[r1,r2] for r1,r2 in self.B[r])
            belief = potentials[r] + sum(self.messages[r1,r2] for r1,r2 in self.B[r])
            belief += np.log(self.total) - belief.logsumexp()
            marginals[r] = belief.exp()
        #print(marginals[('A','B')].datavector())

        return CliqueVector(marginals)


    def hazan_peng_shashua(self, potentials, callback=None):
        # https://arxiv.org/pdf/1210.4881.pdf
        c0 = self.counting_numbers
        pot = {}
        for r in self.regions:
            if r in self.cliques: pot[r] = potentials[r]
            else: pot[r] = Factor.zeros(self.domain.project(r))

        messages = self.messages
        #for p in sorted(self.regions, key=len): #nx.topological_sort(H): # should be G or H?
        #    for r in self.children[p]:
        #        messages[p,r] = Factor.zeros(self.domain.project(r))
        #        messages[r,p] = Factor.zeros(self.domain.project(r))

        cc = {}
        for r in self.regions:
            for p in self.parents[r]:
                cc[p,r] = c0[p] / (c0[r] + sum(c0[p1] for p1 in self.parents[r]))

        for _ in range(self.iters):
            new = {}
            for r in self.regions:
                for p in self.parents[r]:
                    new[p,r] = (pot[p] + sum(messages[c,p] for c in self.children[p] if c!=r) - sum(messages[p,p1] for p1 in self.parents[p])) / c0[p]
                    new[p,r] = c0[p] * new[p,r].logsumexp(tuple(set(p)-set(r)))
                    new[p,r] -= new[p,r].logsumexp()

            for r in self.regions:
                for p in self.parents[r]:
                    new[r,p] = cc[p,r]*(pot[r] + sum(messages[c,r] for c in self.children[r]) + sum(messages[p1,r] for p1 in self.parents[r])) - messages[p,r]
                    #new[r,p] = cc[p,r]*(pot[r] + sum(messages[c,r] for c in self.children[r]) + sum(new[p1,r] for p1 in self.parents[r])) - new[p,r]
                    new[r,p] -= new[r,p].logsumexp()

            #messages = new
            # Damping is not described in paper, but is needed to get convergence for dense graphs
            rho = self.damping
            for p in self.regions:
                for r in self.children[p]:
                    messages[p,r] = rho*messages[p,r] + (1.0-rho)*new[p,r]
                    messages[r,p] = rho*messages[r,p] + (1.0-rho)*new[r,p]
            mu = {}
            for r in self.regions:
                belief = (pot[r] + sum(messages[c,r] for c in self.children[r]) - sum(messages[r,p] for p in self.parents[r])) / c0[r]
                belief += np.log(self.total) - belief.logsumexp()
                mu[r] = belief.exp()

            if callback is not None:
                callback(mu)

            if self.is_converged(mu):
                self.messages = messages
                return CliqueVector(mu)

        self.messages = messages        
        return CliqueVector(mu)
                            

    def wiegerinck(self, potentials, callback=None):
        c = self.counting_numbers
        m = {}
        for delta in self.regions:
            m[delta] = 0
            for alpha in self.ancestors[delta]:
                m[delta] += c[alpha]

        Q = {}
        for r in self.regions:
            if r in self.cliques:
                Q[r] = potentials[r] / c[r]
            else:
                Q[r] = Factor.zeros(self.domain.project(r))

        inner = [r for r in self.regions if len(self.parents[r]) > 0]
        diff = lambda r,s: tuple(set(r)-set(s))
        for _ in range(self.iters):
            for r in inner:
                A = c[r] / (m[r] + c[r])
                B = m[r] / (m[r] + c[r])
                Qbar = sum(c[s]*Q[s].logsumexp(diff(s,r)) for s in self.ancestors[r]) / m[r]
                Q[r] = Q[r]*A + Qbar*B
                Q[r] -= Q[r].logsumexp()
                for s in self.ancestors[r]:
                    Q[s] = Q[s] + Q[r] - Q[s].logsumexp(diff(s,r))
                    Q[s] -= Q[s].logsumexp()

            marginals = {}
            for r in self.regions:
                marginals[r] = (Q[r] + np.log(self.total) - Q[r].logsumexp()).exp()
            if callback is not None:
                callback(marginals)
                
        return CliqueVector(marginals) 

    def loh_wibisono(self, potentials, callback=None):
        # https://papers.nips.cc/paper/2014/file/39027dfad5138c9ca0c474d71db915c3-Paper.pdf
        pot = {}
        for r in self.regions:
            if r in self.cliques: pot[r] = potentials[r]
            else: pot[r] = Factor.zeros(self.domain.project(r))

        rho = self.counting_numbers

        for _ in range(self.iters):
            new = {}
            for s, r in self.message_order:
                diff = tuple(set(s) - set(r))
                num = pot[s]/rho[s]
                for v in self.parents[s]:
                    num += self.messages[v,s]*rho[v]/rho[s]
                for w in self.children[s]:
                    if w != r:
                        num -= self.messages[s,w]
                num = num.logsumexp(diff)
                denom = pot[r]/rho[r]
                for u in self.parents[r]:
                    if u != s:
                        denom += self.messages[u,r]*rho[u]/rho[r]
                for t in self.children[r]:
                    denom -= self.messages[r,t]
                
                new[s,r] = rho[r] / (rho[r]+rho[s]) * (num - denom)
                new[s,r] -= new[s,r].logsumexp()

            for ru, rd in self.message_order:
                self.messages[ru,rd] = 0.5*self.messages[ru,rd] + 0.5*new[ru,rd]

            #ru, rd = self.message_order[0]
            #print(ru, rd, self.messages[ru,rd].values)

            marginals = {}
            for r in self.regions:
                belief = pot[r]/rho[r]
                for s in self.parents[r]:
                    belief += self.messages[s,r]*rho[s]/rho[r]
                for t in self.children[r]:
                    belief -= self.messages[r,t]
                belief += np.log(self.total) - belief.logsumexp()
                marginals[r] = belief.exp()
            #print(marginals[('A','B')].datavector())
            if callback is not None:
                callback(marginals)

        return CliqueVector(marginals)

    def optimize_kikuchi(self, potentials, backend='cvxopt'):
        # Variables are mu[cl] for cl in cliques
        # Objective is -theta^T mu - bethe_entropy(mu)
        # Constraint is Local polytope
        potentials = potentials.copy()
        for r in self.regions:
            if not r in self.cliques:
                potentials[r] = Factor.zeros(self.domain.project(r))

        index = {}
        idx = 0
        for cl in self.regions:
            end = idx + self.domain.size(cl)
            index[cl] = (idx, end)
            idx = end
        clique_size = end
            
        potentials = CliqueVector(potentials)
        def to_vector(marginals):
            return np.concatenate([marginals[cl].datavector() for cl in self.regions])
        def to_cliquevector(vector):
            marginals = {}
            for cl in self.regions:
                start, end = index[cl]
                dom = self.domain.project(cl)
                marginals[cl] = Factor(dom, vector[start:end])
            return CliqueVector(marginals)
        def vector_loss(vector):
            marginals = to_cliquevector(vector)
            f, df = self.energy_functional(potentials, marginals)
            #print('Log', f)
            return f, to_vector(df)

        def make_new_factor(t):
            ans = 1
            for i in t:
                n = self.domain.size(i)
                d = Domain(['y%s'%i, i], [n, n])
                ans = ans * Factor(d, np.eye(n))
            return ans
        def get_query_matrix(r, t):
            m = self.domain.size(t)
            R = Factor.ones(self.domain.project(r))
            T = make_new_factor(t)
            u = tuple('y%s'%i for i in t)
            Qr = (T * R).transpose(u + r)
            return Qr.values.reshape(m,-1)
        def get_constraint(r, s):
            t = tuple(set(r) & set(s))
            if len(t) == 0: return None
            m = self.domain.size(t)
            Qr = get_query_matrix(r, t)
            Qs = get_query_matrix(s, t)
            C = np.zeros((m, clique_size))
            start, end = index[r]
            C[:, start:end] = Qr
            start, end = index[s]
            C[:, start:end] = -Qs
            return C[:-1] # remove last constraint

        def powerset(iterable):
            "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)))
        def enumerate_constraints():
            # Have to be careful to not introduce duplicate constraints, will mess up cvxopt
            ans = []
            for r in self.regions:
                for t in self.children[r]:
                    C = get_constraint(r,t)
                    ans.append(C)
            return np.vstack([C for C in ans if C is not None])

        A1 = enumerate_constraints()
        A2 = block_diag(*[np.ones(self.domain.size(cl)) for cl in self.regions])
        #A2 = np.zeros((len(self.cliques), clique_size))
        #for i, cl in enumerate(self.cliques):
        #    idx, end = index[cl]
        #    A2[i,idx:end] = 1.0
        b1 = np.zeros(A1.shape[0])
        b2 = np.ones(A2.shape[0])
        A = np.vstack([A1, A2])
        b = np.concatenate([b1, b2])
        A,b = reduce_row_echelon(A, b)
        print('CHECKPT', A.shape, np.linalg.matrix_rank(A))

        #P = sum(potentials.values())
        #P = (P - P.logsumexp()).exp()
        #unif = CliqueVector({cl : P.project(cl) for cl in potentials })
        unif = { cl : Factor.uniform(self.domain.project(cl)) for cl in self.regions }
        x0 = to_vector(unif)
       
        if backend == 'scipy': 
            constraint = optimize.LinearConstraint(A, b, b)
            bounds = [(0, None) for _ in range(clique_size)]
            ans = optimize.minimize(vector_loss,
                                    x0,
                                    method='SLSQP',
                                    jac=True,
                                    bounds=bounds,
                                    constraints=constraint)
            result = to_cliquevector(ans.x) * self.total

        elif backend == 'cvxopt':
            Z = np.eye(A.shape[1]) - np.linalg.pinv(A) @ A
            def F(x=None, z=None):
                if x is None:
                    return (0, matrix(x0))
                mu = np.array(x).flatten()
                if mu.min() <= 0: return None
                f, df = vector_loss(mu)
                if z is None:
                    return f, matrix(df).T
                H = self.kikuchi_entropy_hessian(to_cliquevector(mu))
                # Using Z^T H Z as hessian is a heuristic necessary to make CVXOPT happy
                # Likely it is necessary because energy functional is only convex *over the constraints*
                # and not everywhere.  Therefore, H is not positive semidefinite, but Z^T H Z is.
                #return f, matrix(df).T, z[0]*matrix(Z.T @ H @ Z) 
                return f, matrix(df).T, z[0]*matrix(H) 

            G = -matrix(np.eye(clique_size))
            h = matrix(np.zeros(clique_size))
            A = matrix(A)
            b = matrix(b)

            ans = solvers.cp(F, G, h, A=A, b=b)
            print(ans)
            result = to_cliquevector(np.array(ans['x']).flatten()) * self.total
        return result #{ cl : result[cl] for cl in self.cliques }

    def energy_functional(self, potentials, marginals):
        H, dH = self.kikuchi_entropy(marginals)
        f = -potentials.dot(marginals) - H
        df = -1*(potentials + dH)
        return f, df

    def kikuchi_entropy(self, marginals):
        """
        Return the Bethe Entropy and the gradient with respect to the marginals
        
        """
        weights = self.counting_numbers
        entropy = 0
        dmarginals = {}
        for cl in self.regions:
            mu = marginals[cl] / self.total
            entropy += weights[cl] * (mu * mu.log()).sum()
            dmarginals[cl] = weights[cl] * (1 + mu.log()) / self.total
            
        return -entropy, -1*CliqueVector(dmarginals)

    def mle(self, mu):
        return -1*self.kikuchi_entropy(mu)[1]

    def kikuchi_entropy_hessian(self, marginals):
        weights = self.counting_numbers
        hessian = {}
        attributes = set()
        for cl in self.regions:
            mu = marginals[cl] / self.total
            hessian[cl] = weights[cl] * np.diag(1.0 / mu.datavector())
            dom = self.domain.project(cl)
            
        return block_diag(*[hessian[cl] for cl in self.regions])           


def estimate_kikuchi_marginal(domain, total, marginals):
    marginals = dict(marginals)
    regions = set(marginals.keys())
    size = 0
    while len(regions) > size:
        size = len(regions)
        for r1, r2 in itertools.combinations(regions, 2):
            z = tuple(sorted(set(r1) & set(r2)))
            if len(z) > 0 and not z in regions:
                marginals[z] = marginals[r1].project(z)
                regions.update({z})

    G = nx.DiGraph()
    G.add_nodes_from(regions)
    for r1 in regions:
        for r2 in regions:
            if set(r2) < set(r1) and not \
                any(set(r2) < set(r3) and set(r3) < set(r1) for r3 in regions):
                G.add_edge(r1, r2)

    H1 = nx.transitive_closure(G.reverse())
    ancestors = { r : list(H1.neighbors(r)) for r in regions }
    moebius = {}
    def get_counting_number(r):
        if not r in moebius: 
            moebius[r] = 1 - sum(get_counting_number(s) for s in ancestors[r])
        return moebius[r]

    logP = Factor.zeros(domain)
    for r in regions:
        kr = get_counting_number(r)
        logP += kr * marginals[r].log()
    logP += np.log(total) - logP.logsumexp()
    return logP.exp()

# Consumes a *consistent* rectangular system of equations
# Produces a new compacted system of equations with linearly dependent rows removed
# Note: this algorithm computes the row echelon form of the augmented matrix
# This algorithm should be numerically robust
def reduce_row_echelon(B, y):
    A = np.concatenate([B, y[:,np.newaxis]], axis=1)
    m,n = A.shape
    c = 0
    for r in range(m):
        # find first non-zero column
        while c+1 < n and np.all(A[r:, c] == 0):
            c += 1
        if c+1 == n:
            return A[:r,:-1], A[:r,-1]
        # find row with largest value in that column
        i_max = r+np.argmax(np.abs(A[r:,c]))
        # swap those rows
        A[[i_max, r]] = A[[r, i_max]]
        A[r,:] /= A[r,c]
        # zero out column for every row below r
        for i in range(r+1, m):
            A[i,:] -= A[i,c]*A[r,:]
        # clamp small values to 0
        A[np.abs(A) < 1e-12] = 0
    return A[:,:-1], A[:,-1] 
