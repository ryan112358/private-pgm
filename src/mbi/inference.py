import numpy as np
from mbi import Domain, Factor, GraphicalModel, callbacks
from mbi.graphical_model import CliqueVector
from scipy.sparse.linalg import LinearOperator, eigsh, lsmr
from scipy import optimize

class FactoredInference:
    def __init__(self, domain, structural_zeros = None, metric='L2', log=False, iters=100, warm_start=False, elim_order=None):
        """
        Class for learning a GraphicalModel from  noisy measurements on a data distribution
        
        :param domain: The domain information (A Domain object)
        :param structural_zeros: An encoding of the known (structural) zeros in the distribution.
            Specified as a dictionary where 
                - each key is a subset of attributes of size r
                - each value is a list of r-tuples corresponding to impossible attribute settings
        :param metric: The optimization metric.  May be L1, L2 or a custom callable function
            - custom callable function must consume the marginals and produce the loss and gradient
            - see FactoredInference._marginal_loss for more information
        :param log: flag to log iterations of optimization
        :param iters: number of iterations to optimize for
        :param warm_start: initialize new model or reuse last model when calling infer multiple times
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.  
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.structural_zeros = structural_zeros
        self.metric = metric
        self.log = log
        self.iters = iters
        self.warm_start = warm_start
        self.history = []
        self.elim_order = elim_order

    def infer(self, measurements, total = None, engine='RDA', callback=None, options = {}):
        """ 
        Estimate a GraphicalModel from the given measurements

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param engine: the optimization algorithm to use, options include:
            MD - Mirror Descent with armijo line search
            MD2 - Mirror Descent with custom step size
            RDA - Regularized Dual Averaging
            LBFGS - Limited Memory BFGS 
            EM - Expectation Maximization
            IG - Interior Gradient
            LNNLS - Local Non-negative least squares
        :param callback: a function to be called after each iteration of optimization
        :param options: solver specific options passed as a dictionary
            { param_name : param_value }
        
        :return model: A GraphicalModel that best matches the measurements taken
        """      
        options['callback'] = callback
        if callback is None and self.log:
            options['callback'] = callbacks.Logger(self)
        if engine == 'MD':
            self.mirror_descent(measurements, total, **options)
        elif engine == 'MD2':
            self.mirror_descent2(measurements, total, **options)
        elif engine == 'RDA':
            self.dual_averaging(measurements, total, **options)
        elif engine == 'LBFGS':
            self.lbfgs(measurements, total, **options)
        elif engine == 'EM':
            self.expectation_maximization(measurements, total, **options)
        elif engine == 'IG':
            self.interior_gradient(measurements, total, **options)
        elif engine == 'LNNLS':
            self.local_nnls(measurements, total, **options)
        return self.model

    def local_nnls(self, measurements, total = None, callback=None):
        self._setup(measurements, total)

        model = self.model
        cliques, potentials = model.cliques, model.potentials

        init =  model.dict_to_vector(model.belief_prop_fast(potentials))

        def loss_and_grad(params):
            mu = model.vector_to_dict(params)
            if callback is not None:
                callback(mu)
            loss, dmu = self._marginal_loss(mu)
            dparams = model.dict_to_vector(dmu)
            return loss, dparams

        opts = { 'maxiter' : self.iters }
        bnds = [(0,None)]*len(init)
        res=optimize.minimize(loss_and_grad,init,method='L-BFGS-B',bounds=bnds,jac=True,options=opts)

        mu = model.vector_to_dict(res.x)
        model.potentials = model.mle(mu)
        model.marginals = model.belief_prop_fast(model.potentials)

    def interior_gradient(self, measurements, total, lipchitz = None,c = 1,sigma = 1,callback=None):
        """ Use the interior gradient algorithm to estimate the GraphicalModel
            See https://epubs.siam.org/doi/pdf/10.1137/S1052623403427823 for more information

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipchitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param c, sigma: parameters of the algorithm
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipchitz is not None,'lipchitz constant must be supplied'
        self._setup(measurements, total)
        # what are c and sigma?  For now using 1
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipchitz() if lipchitz is None else lipchitz
        if self.log:
            print('Lipchitz constant:', L)
    
        theta = model.potentials
        x = y = z = model.belief_prop_fast(theta)
        c0 = c
        l = sigma/L
        for k in range(1, self.iters+1):
            a = (np.sqrt((c*l)**2 + 4*c*l) - l*c) / 2
            y = (1 - a)*x + a*z
            c *= (1-a)
            _, g = self._marginal_loss(y) 
            theta = theta - a/c/total * g
            z = model.belief_prop_fast(theta)
            x = (1-a)*x + a*z
            if callback is not None:
                callback(x)

        model.marginals = x
        model.potentials = model.mle(x) 

    def dual_averaging(self, measurements, total = None, lipchitz = None, callback=None):
        """ Use the regularized dual averaging algorithm to estimate the GraphicalModel
            See https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipchitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipchitz is not None,'lipchitz constant must be supplied'
        self._setup(measurements, total)
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipchitz() if lipchitz is None else lipchitz
        print('Lipchitz constant:', L)
        if L == 0: return
 
        theta = model.potentials
        gbar = CliqueVector({ cl : Factor.zeros(domain.project(cl)) for cl in cliques })
        w = v = model.belief_prop_fast(theta)
        beta = 0

        for t in range(1, self.iters+1):
            c = 2.0 / (t + 1)
            u = (1-c)*w + c*v
            _, g = self._marginal_loss(u) # not interested in loss of this query point
            gbar = (1-c)*gbar + c*g
            theta = -t*(t+1)/(4*L+beta)/self.model.total * gbar 
            v = model.belief_prop_fast(theta)
            w = (1-c)*w + c*v
           
            if callback is not None:
                callback(w)

        model.marginals = w
        model.potentials = model.mle(w)

    def mirror_descent2(self, measurements, total = None, alpha0=1.0, callback=None):
        """ Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param alpha0: the initial learning rate
            - Uses alpha_t = alpha0 / t (square summable but not summable step sizes)
        :param callback: a function to be called after each iteration of optimization
        """
        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials

        mu = model.belief_prop_fast(theta)
        if callback is not None:
            callback(mu)

        for t in range(1, self.iters + 1):
            alpha = alpha0 / t
            l, dL = self._marginal_loss(mu)
            theta = theta - alpha*dL
            mu = model.belief_prop_fast(theta)
            if callback is not None:
                callback(mu)

        model.potentials = theta
        model.marginals = mu

        return l

    def mirror_descent(self, measurements, total = None, alpha=None, callback=None):
        """ Use the mirror descent algorithm with armijo line search to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf

        Note: this method requires the loss function to be smooth (e.g., L2)
        
        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param alpha: the initial learning rate
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'loss function not smooth, use mirror_descent2 (MD2) instead'
        
        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        if alpha is None:
            alpha = 1.0/model.total

        mu = model.belief_prop_fast(theta)
        if callback is not None:
            callback(mu)
        l, dL = self._marginal_loss(mu)

        def update(theta, dL, alpha):
            theta2 = theta - alpha*dL
            mu2 = model.belief_prop_fast(theta2)
            l2, dL2 = self._marginal_loss(mu2)
            return theta2, mu2, l2, dL2

        theta2, mu2, l2, dL2 = update(theta, dL, alpha)
        if l2 < l:
            while 0.5*dL.dot(mu2-mu) + l > l2:
                alpha *= 2.0
                theta2, mu2, l2, dL2 = update(theta, dL, alpha)
            theta, mu, l, dL = theta2, mu2, l2, dL2

        for t in range(1, self.iters + 1):
            theta2, mu2, l2, dL2 = update(theta, dL, alpha)
            while 0.5*dL.dot(mu2-mu) + l < l2:
                alpha *= 0.5
                theta2, mu2, l2, dL2 = update(theta, dL, alpha)
            theta, mu, l, dL = theta2, mu2, l2, dL2
            if callback is not None:
                callback(mu)
            if t%100 == 0:
                alpha *= 10

        model.potentials = theta
        model.marginals = mu

        return l

    def lbfgs(self, measurements, total = None, callback=None):
        """ Estimate model using LBFGS algorithm by solving the following problem

        theta* = argmin L(mu(theta))

        Note: this is not a convex objective so convergence to global optima is not guaranteed.
                In practice, it usually converges and is pretty efficient.

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        """
        self._setup(measurements, total)
        model = self.model
        cliques, potentials = model.cliques, model.potentials
       
        init =  model.dict_to_vector(potentials) 

        def loss_and_grad(params):

            theta = model.vector_to_dict(params)

            u, logZ, cache = model.belief_propagation(theta)
            loss, du = self._marginal_loss(u)
            dtheta = model.back_belief_propagation(du, cache)

            model.potentials = theta
            model.marginals = u
            if callback is not None:
                callback(u)

            dparams = model.dict_to_vector(dtheta)
            return loss, dparams

        opts = { 'maxiter' : self.iters }
        res = optimize.minimize(loss_and_grad, init, method='L-BFGS-B', jac=True, options=opts)
        return res.fun

    def expectation_maximization(self, measurements, total, alpha, callback=None):
        self._setup(measurements, total)
        model = self.model
        cliques, theta, domain = model.cliques, model.potentials, model.domain

        em_history = []
        emiter = 0

        for _ in range(self.iters // 500):
            n = model.belief_prop_fast(theta)
            for _ in range(500):
                loss, dL = self._marginal_loss(n)
                theta1 = theta - dL
                n1 = model.belief_prop_fast(theta1)
                n = (1-alpha)*n + alpha*n1
                if callback is not None:
                    callback(n)
            theta = model.mle(n)

        model.potentials = theta
        model.marginals = n

    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal 
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = { }

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                c = 1.0/noise
                mu2 = mu.project(proj)
                x = mu2.values.flatten()
                diff = c*(Q.dot(x) - y)
                if metric == 'L1':
                    loss += np.sum(np.abs(diff))
                    grad = c*Q.T.dot(np.sign(diff))
                else:
                    loss += 0.5*np.dot(diff, diff)
                    grad = c*Q.T.dot(diff)
                gradient[cl] += Factor(mu2.domain, grad)
        return loss, CliqueVector(gradient)

    def _setup(self, measurements, total):
        """ Perform necessary setup for running inference
       
        1. If total is None, find the minimum variance unbiased estimate for total and use that
        2. Construct the GraphicalModel 
            * If there are structural_zeros in the distribution, initialize factors appropriately
        3. Pre-process measurements into groups so that _marginal_loss may be evaluated efficiently
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, noise, proj in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]
                if np.allclose(Q.T.dot(v), o):
                    variances = np.append(variances, noise**2 * np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(0, estimate)

        if not self.warm_start or not hasattr(self, 'model'):
            cliques = [m[3] for m in measurements] 
            if self.structural_zeros is not None:
                cliques += list(self.structural_zeros.keys())
            self.model = GraphicalModel(self.domain,cliques,total,elimination_order=self.elim_order)
            if self.structural_zeros is not None:
                for cl in self.structural_zeros:
                    dom = self.domain.project(cl)
                    zeros = self.structural_zeros[cl]
                    fact = Factor.active(dom, zeros)
                    for cl2 in self.model.potentials:
                        if set(cl) <= set(cl2):
                            self.model.potentials[cl2] += fact
                            break
        
        cliques = self.model.cliques
        self.groups = { cl : [] for cl in cliques }
        for m in measurements:
            for cl in cliques:
                # (Q, y, noise, proj) tuple
                if set(m[3]) <= set(cl):
                    self.groups[cl].append(m)
                    break


    def _lipchitz(self):
        """ compute lipchitz constant for L2 loss 

            Note: must be called after _setup
        """
        # first convert Qs into a linear operator over the concatenated clique marginals
        def matvec(vector):
            ans = []
            idx = 0
            for cl in self.groups:
                dom = self.domain.project(cl)
                end = idx + dom.size()
                mu = Factor(dom, vector[idx:end])
                idx = end
                for Q, y, noise, proj in self.groups[cl]:
                    c = 1.0/noise
                    mu2 = mu.project(proj)
                    x = mu2.values.flatten()
                    ans.append(c*Q.dot(x))
            return np.concatenate(ans)
        def rmatvec(vector):
            ans = []
            idx = 0
            for cl in self.groups:
                dom = self.domain.project(cl)
                mu = Factor.zeros(dom)
                for Q, y, noise, proj in self.groups[cl]:
                    c = 1.0/noise
                    dom2 = dom.project(proj)
                    end = idx + Q.shape[0]
                    v = vector[idx:end]
                    idx = end
                    mu += Factor(dom2, c*Q.T.dot(v))
                ans.append(mu.values.flatten())
            return np.concatenate(ans)

        m, n = 0,0
        for cl in self.groups:
            n += np.prod(self.domain.project(cl).shape)
            for Q, _, _, _ in self.groups[cl]:
                m += Q.shape[0]
        
        if m == 0: return 0

        Q = LinearOperator((m, n), matvec, rmatvec)
        return eigsh(Q.H * Q, 1)[0][0]# * self.model.total

