import numpy as np
from collections import defaultdict
from mbi import Domain, Factor, CliqueVector
from scipy.linalg import block_diag
from scipy import optimize
from functools import reduce
import itertools
from cvxopt import solvers, matrix
from multiprocessing import Pool
from scipy import sparse
solvers.options['show_progress'] = False

class FactorGraph():
    def __init__(self, domain, cliques, total = 1.0, convex = False, iters=25):
        self.domain = domain
        self.cliques = cliques
        self.total = total
        self.convex = convex
        self.iters = iters

        if convex:
            self.counting_numbers = self.get_counting_numbers()
            self.belief_propagation = self.convergent_belief_propagation
        else:
            counting_numbers = {}
            for cl in cliques:
                counting_numbers[cl] = 1.0
            for a in domain:
                counting_numbers[a] = 1.0 - len([cl for cl in cliques if a in cl])
            self.counting_numbers = None, None, counting_numbers
            self.belief_propagation = self.loopy_belief_propagation

        self.potentials = None
        self.marginals = None
        self.mu_n, self.mu_f = self.init_messages()
        self.beliefs = { i : Factor.zeros(domain.project(i)) for i in domain }

    def datavector(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def init_messages(self):
        mu_n = defaultdict(dict)
        mu_f = defaultdict(dict)
        for cl in self.cliques:
            for v in cl:
                mu_f[cl][v] = Factor.zeros(self.domain.project(v))
                mu_n[v][cl] = Factor.zeros(self.domain.project(v))
        return mu_n, mu_f

    def project(self,attrs):
        if type(attrs) is list:
            attrs = tuple(attrs)
        
        if self.marginals is not None:
            # we will average all ways to obtain the given marginal, 
            # since there may be more than one
            ans = Factor.zeros(self.domain.project(attrs))
            terminate = False
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    ans += self.marginals[cl].project(attrs)
                    terminate = True
            if terminate: return ans * (self.total / ans.sum())
        
        belief = sum(self.beliefs[i] for i in attrs)
        belief += np.log(self.total) - belief.logsumexp()
        return belief.transpose(attrs).exp()

      
    def loopy_belief_propagation(self, potentials, callback=None):
       
        mu_n, mu_f = self.mu_n, self.mu_f 
        self.potentials = potentials
        
        for i in range(self.iters):
            #factor to variable BP
            for cl in self.cliques:
                pre = sum(mu_n[c][cl] for c in cl)
                for v in cl:
                    complement = [var for var in cl if var is not v]
                    mu_f[cl][v] = potentials[cl] + pre - mu_n[v][cl]
                    mu_f[cl][v] = mu_f[cl][v].logsumexp(complement)
                    mu_f[cl][v] -= mu_f[cl][v].logsumexp()

            #variable to factor BP
            for v in self.domain:
                fac = [cl for cl in self.cliques if v in cl]
                pre = sum(mu_f[cl][v] for cl in fac)
                for f in fac:
                    complement = [var for var in fac if var is not f]
                    #mu_n[v][f] = Factor.zeros(self.domain.project(v))
                    mu_n[v][f] = pre - mu_f[f][v] #sum(mu_f[c][v] for c in complement)
                    #mu_n[v][f] += sum(mu_f[c][v] for c in complement)
                    #mu_n[v][f] -= mu_n[v][f].logsumexp()
            
            if callback is not None:
                mg = self.clique_marginals(mu_n, mu_f, potentials) 
                callback(mg)

        self.beliefs={v:sum(self.mu_f[cl][v] for cl in self.cliques if v in cl) for v in self.domain}
        self.mu_f, self.mu_n = mu_f, mu_n
        self.marginals = self.clique_marginals(mu_n, mu_f, potentials)
        return self.marginals

    def convergent_belief_propagation(self, potentials, callback=None):
        # Algorithm 11.2 in Koller & Friedman (modified to work in log space)

        v, vhat, k = self.counting_numbers
        sigma, delta = self.mu_n, self.mu_f
        #sigma, delta = self.init_messages()

        for it in range(self.iters):

            #pre = {}
            #for r in self.cliques:
            #    pre[r] = sum(sigma[j][r] for j in r)

            for i in self.domain:
                nbrs = [r for r in self.cliques if i in r]
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    delta[r][i] = potentials[r] + sum(sigma[j][r] for j in comp)
                    #delta[r][i] = potentials[r] + pre[r] - sigma[i][r]
                    delta[r][i] /= vhat[i,r]
                    delta[r][i] = delta[r][i].logsumexp(comp)
                belief = Factor.zeros(self.domain.project(i)) 
                belief += sum(delta[r][i]*vhat[i,r] for r in nbrs) / vhat[i]
                belief -= belief.logsumexp()
                self.beliefs[i] = belief
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    A = -v[i,r]/vhat[i,r]
                    B = v[r]
                    sigma[i][r] = A*(potentials[r] + sum(sigma[j][r] for j in comp))
                    #sigma[i][r] = A*(potentials[r] + pre[r] - sigma[i][r])
                    sigma[i][r] += B*(belief - delta[r][i])
            if callback is not None:
                mg = self.clique_marginals(sigma, delta, potentials)
                callback(mg)

        self.mu_n, self.mu_f = sigma, delta
        return self.clique_marginals(sigma, delta, potentials)

    def clique_marginals(self, mu_n, mu_f, potentials):
        if self.convex: v, _, _ = self.counting_numbers
        marginals = {}
        for cl in self.cliques:
            belief = potentials[cl] + sum(mu_n[n][cl] for n in cl)
            if self.convex: belief *= 1.0/v[cl]
            belief += np.log(self.total) - belief.logsumexp()
            marginals[cl] = belief.exp()
        return CliqueVector(marginals)

    def optimize_bethe(self, potentials, backend='cvxopt'):
        loss_fn = lambda mu: self.energy_functional(potentials, mu)
        return self.optimize(loss_fn, backend=backend, hessian=self.bethe_entropy_hessian)

    def optimize(self, loss_fn, backend='scipy', hessian=None):
        # Variables are mu[cl] for cl in cliques
        # Objective is -theta^T mu - bethe_entropy(mu)
        # Constraint is Local polytope
        index = {}
        idx = 0
        for cl in self.cliques:
            end = idx + self.domain.size(cl)
            index[cl] = (idx, end)
            idx = end
        clique_size = end
            
        def to_vector(marginals):
            return np.concatenate([marginals[cl].datavector() for cl in self.cliques])
        def to_cliquevector(vector):
            marginals = {}
            for cl in self.cliques:
                start, end = index[cl]
                dom = self.domain.project(cl)
                marginals[cl] = Factor(dom, vector[start:end])
            return CliqueVector(marginals)
        def vector_loss(vector):
            marginals = to_cliquevector(vector)
            f, df = loss_fn(marginals)
            #print(f)
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
            canonical = {}
            for r in self.cliques:
                for t in powerset(r):
                    t = tuple(sorted(t))
                    if not t in canonical:
                        canonical[t] = r
            ans = []                
            for r in self.cliques:
                for s in self.cliques:
                    if r == s: break
                    t = tuple(sorted(set(r) & set(s)))
                    if len(t)>0 and s == canonical[t]:
                        ans.append(get_constraint(r, s))
            return np.vstack([C for C in ans if C is not None])

        A1 = enumerate_constraints()
        A2 = block_diag(*[np.ones(self.domain.size(cl)) for cl in self.cliques])
        b1 = np.zeros(A1.shape[0])
        b2 = np.ones(A2.shape[0])*self.total 
        A = np.vstack([A1, A2])
        b = np.concatenate([b1, b2])
        A, b = reduce_row_echelon(A,b)

        print('CHECKPT', A.shape, np.linalg.matrix_rank(A))

        #unif = { cl : self.total*Factor.uniform(self.domain.project(cl)) for cl in self.cliques }
        unif = self.total*CliqueVector.uniform(self.domain, self.cliques)
        x0 = to_vector(unif)
       
        if backend == 'scipy': 
            #A = sparse.csr_matrix(A)
            constraint = optimize.LinearConstraint(A, b, b)
            bounds = [(0, None) for _ in range(clique_size)]
            ans = optimize.minimize(vector_loss,
                                    x0,
                                    method='SLSQP',
                                    jac=True,
                                    bounds=bounds,
                                    constraints=constraint)
            print(ans)
            return to_cliquevector(ans.x) * self.total

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
                # NOTE: This does not work with loss_fn
                H = hessian(to_cliquevector(mu))
                H = block_diag(*[H[cl] for cl in self.cliques])
                # Using Z^T H Z as hessian is a heuristic necessary to make CVXOPT happy
                # Likely it is necessary because energy functional is only convex *over the constraints*
                # and not everywhere.  Therefore, H is not positive semidefinite, but Z^T H Z is.
                #return f, matrix(df).T, z[0]*matrix(H) #matrix(Z.T @ H @ Z) 
                return f, matrix(df).T, z[0]*matrix(Z.T @ H @ Z) 

            G = -matrix(np.eye(clique_size))
            h = matrix(np.zeros(clique_size))
            A = matrix(A)
            b = matrix(b)

            ans = solvers.cp(F, G, h, A=A, b=b)
            print(ans)
            return to_cliquevector(np.array(ans['x']).flatten())# * self.total

    def mle(self, marginals):
        return -self.bethe_entropy(marginals)[1]

    def energy_functional(self, potentials, marginals):
        H, dH = self.bethe_entropy(marginals)
        f = -potentials.dot(marginals) - H
        df = -1*(potentials + dH)
        return f, df

    def bethe_entropy(self, marginals):
        """
        Return the Bethe Entropy and the gradient with respect to the marginals
        
        """
        _, _, weights = self.counting_numbers
        entropy = 0
        dmarginals = {}
        attributes = set()
        for cl in self.cliques:
            mu = marginals[cl] / self.total
            entropy += weights[cl] * (mu * mu.log()).sum()
            dmarginals[cl] = weights[cl] * (1 + mu.log()) / self.total
            for a in set(cl) - set(attributes):
                p = mu.project(a)
                entropy += weights[a] * (p * p.log()).sum()
                dmarginals[cl] += weights[a] * (1 + p.log()) / self.total
                attributes.update(a)
            
        return -entropy, -1*CliqueVector(dmarginals)

    def bethe_entropy_hessian(self, marginals):
        _, _, weights = self.counting_numbers
        hessian = {}
        attributes = set()
        for cl in self.cliques:
            mu = marginals[cl] / self.total
            hessian[cl] = weights[cl] * np.diag(1.0 / mu.datavector())
            dom = self.domain.project(cl)
            for a in set(cl) - set(attributes):
                p = mu.project(a)
                subs = [np.ones((n,n)) for n in dom.shape]
                subs[dom.attrs.index(a)] = weights[a]*np.diag(1.0 / p.datavector())
                H = reduce(np.kron, subs)
#                print(a, weights[a], p.datavector(), H)
                hessian[cl] += H
                attributes.update(a)
        return hessian
            

    def get_counting_numbers(self):
        index = {}
        idx = 0

        for i in self.domain:
            index[i] = idx
            idx += 1
        for r in self.cliques:
            index[r] = idx
            idx += 1
            
        for r in self.cliques:
            for i in r:
                index[r,i] = idx
                idx += 1
                    
        vectors = {}
        for r in self.cliques:
            v = np.zeros(idx)
            v[index[r]] = 1
            for i in r:
                v[index[r,i]] = 1
            vectors[r] = v
                

        for i in self.domain:
            v = np.zeros(idx)
            v[index[i]] = 1
            for r in self.cliques:
                if i in r:
                    v[index[r,i]] = -1
            vectors[i] = v
            
        constraints = []
        for i in self.domain:
            con = vectors[i].copy()
            for r in self.cliques:
                if i in r:
                    con += vectors[r]
            constraints.append(con)
        A = np.array(constraints)
        b = np.ones(len(self.domain))

        X = np.vstack([vectors[r] for r in self.cliques])
        y = np.ones(len(self.cliques))
        P = X.T @ X
        q = -X.T @ y
        G = -np.eye(q.size)
        h = np.zeros(q.size)
        minBound = 1.0 / len(self.domain)
        for r in self.cliques:
            h[index[r]] = -minBound

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        ans = solvers.qp(P, q, G, h, A, b)
        x = np.array(ans['x']).flatten()
        #for p in vectors: print(p, vectors[p] @ x)

        counting_v = {}
        for r in self.cliques:
            counting_v[r] = x[index[r]]
            for i in r:
                counting_v[i,r] = x[index[r,i]]
        for i in self.domain:
            counting_v[i] = x[index[i]]


        counting_vhat = {}
        counting_k = {}
        for i in self.domain:
            nbrs = [r for r in self.cliques if i in r]
            counting_vhat[i] = counting_v[i] + sum(counting_v[r] for r in nbrs)
            counting_k[i] = counting_v[i] - sum(counting_v[i,r] for r in nbrs)
            for r in nbrs:
                counting_vhat[i,r] = counting_v[r] + counting_v[i,r]
        for r in self.cliques:
            counting_k[r] = counting_v[r] + sum(counting_v[i,r] for i in r)

        return counting_v, counting_vhat, counting_k

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
