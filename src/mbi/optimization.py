import numpy as np
from cvxopt import matrix, solvers
from mbi import Domain, Factor, CliqueVector, FactorGraph, RegionGraph, FactoredInference
from scipy.linalg import block_diag
from scipy import optimize

class Optimizer(RegionGraph):
    def __init__(self, domain, cliques, total = 1.0, consistency=True, simplex=True):
        super(Optimizer, self).__init__(domain, cliques, total, convex=True, minimal=True)
        self.consistency = consistency
        self.simplex = simplex
        self.index = {}
        idx = 0
        for cl in self.regions:
            end = idx + self.domain.size(cl)
            self.index[cl] = (idx, end)
            idx = end
        self.clique_size = end

    def estimate(self, measurements, total=None, callback=None, backend='cvxopt', metric='L2'):
        engine = FactoredInference(self.domain, metric=metric, marginal_oracle='convex')
        engine._setup(measurements, total)
        self.total = engine.model.total
        hessian = None
        if backend == 'cvxopt':
            hessian = {}
            for r in self.regions:
                n = self.domain.size(r)
                hessian[r] = np.zeros((n,n))
            for Q, _, _, cl in measurements:
                hessian[cl] += Q.T @ Q
            hessian = lambda mu: hessian
        return self.optimize(engine._marginal_loss, 
                                backend=backend, 
                                hessian=hessian,
                                callback=callback)

    def to_vector(self, marginals):
        return np.concatenate([marginals[cl].datavector() for cl in self.regions])

    def to_cliquevector(self, vector):
        marginals = {}
        for cl in self.regions:
            start, end = self.index[cl]
            dom = self.domain.project(cl)
            marginals[cl] = Factor(dom, vector[start:end])
        return CliqueVector(marginals) 

    def vectorize(self, loss_fn):
        def ans(vector):
            marginals = self.to_cliquevector(vector)
            f, df = loss_fn(marginals)
            return f, self.to_vector(df)
        return ans

    def get_consistency_constraints(self):
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
            C = np.zeros((m, self.clique_size))
            start, end = self.index[r]
            C[:, start:end] = Qr
            start, end = self.index[s]
            C[:, start:end] = -Qs
            return C#[:-1] # remove last constraint
        def powerset(iterable):
            "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s,r) for r in range(1,len(s)))

        ans = []
        for r in self.regions:
            for t in self.children[r]:
                C = get_constraint(r,t)
                ans.append(C)
        A = np.vstack(ans)
        b = np.zeros(A.shape[0])
        return A, b

    def get_simplex_constraints(self):
        A = block_diag(*[np.ones(self.domain.size(cl)) for cl in self.regions])
        b = np.ones(A.shape[0])*self.total
        return A, b

    def get_constraints(self):
        A = []
        b = []
        if self.consistency:
            A1, b1 = self.get_consistency_constraints()
            A.append(A1)
            b.append(b1)
        if self.simplex:
            A2, b2 = self.get_simplex_constraints()
            A.append(A2)
            b.append(b2)
        if len(A) == 0:
            return None, None
        A = np.vstack(A)
        b = np.concatenate(b)
        A,b = reduce_row_echelon(A, b)
        return A, b

    def optimize_kikuchi(self, potentials, backend='cvxopt'):
        loss_fn = lambda mu: self.energy_functional(potentials, mu)
        return self.optimize(loss_fn, backend=backend, hessian=self.kikuchi)

    def optimize(self, loss_fn, backend='scipy', hessian=None, callback=None):
        def vector_loss(vector):
            marginals = self.to_cliquevector(vector)
            if callback is not None:
                callback(marginals)
            f, df = loss_fn(marginals)
            return f, self.to_vector(df)
        A, b = self.get_constraints()
        unif = self.total*CliqueVector.uniform(self.domain, self.regions)
        x0 = self.to_vector(unif)
        if backend == 'scipy':
            #A = sparse.csr_matrix(A)
            if A is not None:
                constraint = optimize.LinearConstraint(A, b, b)
            else:
                constraint = tuple()
            if self.simplex:
                bounds = [(0, None) for _ in range(self.clique_size)]
            else:
                bounds = None
            ans = optimize.minimize(vector_loss,
                                    x0,
                                    method='SLSQP',
                                    jac=True,
                                    bounds=bounds,
                                    constraints=constraint)
            #print(ans)
            return self.to_cliquevector(ans.x)# * self.total

        elif backend == 'cvxopt':
            #Z = np.eye(A.shape[1]) - np.linalg.pinv(A) @ A
            def F(x=None, z=None):
                if x is None:
                    return (0, matrix(x0))
                mu = np.array(x).flatten()
                if mu.min() <= 0: return None
                f, df = vector_loss(mu)
                if z is None:
                    return f, matrix(df).T
                H = hessian(self.to_cliquevector(mu))
                H = block_diag(*[H[cl] for cl in self.regions])
                # Using Z^T H Z as hessian is a heuristic necessary to make CVXOPT happy
                # Likely it is necessary because energy functional is only convex *over the constraints*
                # and not everywhere.  Therefore, H is not positive semidefinite, but Z^T H Z is.
                return f, matrix(df).T, z[0]*matrix(H) #matrix(Z.T @ H @ Z) 

            if self.simplex:
                G = -matrix(np.eye(self.clique_size))
                h = matrix(np.zeros(self.clique_size))
            else:
                G = h = None

            if A is not None:
                A = matrix(A)
                b = matrix(b)

            ans = solvers.cp(F, G, h, A=A, b=b)
            #print(ans)
            return self.to_cliquevector(np.array(ans['x']).flatten())# * self.total



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

    def kikuchi_entropy_hessian(self, marginals):
        weights = self.counting_numbers
        hessian = {}
        attributes = set()
        for cl in self.regions:
            mu = marginals[cl] / self.total
            hessian[cl] = weights[cl] * np.diag(1.0 / mu.datavector())
            dom = self.domain.project(cl)

        return block_diag(*[hessian[cl] for cl in self.regions])



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

