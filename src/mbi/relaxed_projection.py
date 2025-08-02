import numpy as np
import pandas as pd
import itertools
from typing import Callable, List, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, vmap, lax
from jax.example_libraries import optimizers
from jax.example_libraries.optimizers import unpack_optimizer_state, pack_optimizer_state
from dataclasses import dataclass, field
from .dataset import Dataset




def initialize_synthetic_dataset(
    key: jnp.ndarray, num_generated_points, data_dimension
):
    shape = (num_generated_points, data_dimension)
    random_initial = random.uniform(key=key, shape=shape)
    return random_initial


@jax.jit
def sparsemax(logits):
    """forward pass for sparsemax
    this will process a 2d-array $logits, where axis 1 (each row) is assumed to be
    the logits-vector.
    """

    # sort logits
    z_sorted = jnp.sort(logits, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = jnp.cumsum(z_sorted, axis=1)
    k = jnp.arange(1, logits.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = logits.shape[1] - jnp.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(logits)
    tau_sum = z_cumsum[jnp.arange(0, logits.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return jnp.maximum(0, logits - tau_z)


@jax.jit
def sparsemax_project(D, feats_idx):
    return jnp.hstack(
        [sparsemax(D[:, q]) if len(q) > 1 else D[:, q] for q in feats_idx]  # after
    )


def run_sgd(
        D_prime: jnp.ndarray,
        opt_init,
        progress_loss_fn: Callable,
        update_fn: Callable,
        opt_lr: float,
        iters: int,
        rp_stop: float,
        sigmoid_0: float,
        sigmoid_double: int,
    ) -> jnp.ndarray:
        """
        key function used to update D
        Args:
            D_prime: the initial synthetic dataset as a JAX array
            opt_init: function to initialize the optimizer state, given D_prime
            update_fn: function performing one update step;
                    signature (opt_state, sigmoid_param, learning_rate)
                    -> (new_opt_state, new_D_prime, loss)
            opt_lr: learning rate for each SGD step
            iters: number of update iterations
            rp_stop, sigmoid_0, sigmoid_double: early stop param, not used
        """
        # Initialize optimizer state and best trackers
        opt_state = opt_init(D_prime)
        D_prime_best = D_prime
        best_loss = progress_loss_fn(D_prime_best)

        # Define the scan body
        def step(carry, _):
            opt_state, D, D_prime_best, best_loss, prev_loss = carry
            opt_state, D_new, loss = update_fn(opt_state, opt_lr)

            mask = loss < best_loss
            best_loss = jnp.where(mask, loss, best_loss)
            D_prime_best = jnp.where(mask[..., None], D_new, D_prime_best)

            carry = (
                opt_state,
                D_new,
                D_prime_best,
                best_loss,
                loss
            )
            return carry, None

        carry0 = (
            opt_state,
            D_prime,
            D_prime_best,
            best_loss,
            best_loss
        )

        carry_final, _ = lax.scan(step, carry0, None, length=iters)
        D_prime_best = carry_final[2]
        return D_prime_best

run_sgd = jax.jit(run_sgd, static_argnums=(1, 2, 3, 5))


# --- configuration --- 

@dataclass
class RAPppConfiguration():
    optimizer_learning_rate: List[float] = field(default_factory=lambda: [0.01])
    iterations: List[int] = field(default_factory=lambda: [50])
    rap_percent_stopping_condition: List[float] = field(
        default_factory=lambda: [0.0001]
    )
    clip_grad: float = 0.1
    sigmoid_0: List[float] = field(default_factory=lambda: [2])
    sigmoid_doubles: list = field(default_factory=lambda: [10])
    loss_type: list = field(default_factory=lambda: [2])
    truncate_sigmoid: list = field(default_factory=lambda: [True])






#######################################################
#                                                     #
#         Here is a the marginal class                #
#                                                     #
#######################################################

class MarginalStatistics():
    def __init__(self, domain, k, max_number_rows=5000):
        self.domain = domain
        self.K = min(k, len(self.domain.attrs))
        self.max_number_rows = max_number_rows

        self.workload = list(itertools.combinations(self.domain.attrs, self.K))
        self.max_size = max([self.domain.size(cl) for cl in self.workload])
        self.feature_indices_map = {}
        s = 0
        for col in self.domain.attrs:
            e = s + self.domain.size(col)
            self.feature_indices_map[col] = list(range(s, e))
            s = e

        queries = []
        for marginal in self.workload:
            positions = [] # columns
            for col in marginal:
                positions.append(self.feature_indices_map[col])
            indices = [] # unqiue values
            for tup in itertools.product(*positions):
                indices.append(tup)
            # K-way Index vector
            idx = []
            for k in range(self.K):
                idx.append(np.array([q[k] for q in indices]))
            idx = np.array(idx)
            queries.append(idx)
        queries = jnp.array(
            [
                jnp.hstack(
                    [jnp.array(query_idx), -jnp.ones((self.K, self.max_size - query_idx.shape[1]))]
                )
                for query_idx in queries
            ]
        ).astype(int)
        extra_zeroes = -jnp.ones((1, self.K, self.max_size), dtype=int)
        self.queries = jnp.vstack([queries, extra_zeroes])


    def compute_stat(self, D_zeroes, query_index):
        sub_marginal_queries = self.queries[query_index]
        D_temp = D_zeroes[:, sub_marginal_queries]
        answers = D_temp.prod(axis=1)
        stats = answers.sum(axis=0)
        return stats

    def get_num_queries(self):
        return self.queries.shape[0] - 1

    def get_workload_index(self, workload):
        try:
            idx = self.workload.index(workload)
        except:
            idx = -1
        return idx
    
    def get_exact_statistics_fn(self):
        compute_stats_vmap = vmap(self.compute_stat, in_axes=(None, 0))

        @jax.jit
        def compute_statistics_fn(D, queries_idx: jnp.ndarray):
            queries_idx = queries_idx.reshape(-1).astype(int)

            D_zeroes = jnp.hstack(
                [D, jnp.zeros((D.shape[0], 1))]
            )  # add a column of zeros
            D_zeroes = jnp.asarray(D_zeroes)
            rows = D_zeroes.shape[0]

            num_D_split = max(2, int(rows / self.max_number_rows + 0.5))
            D_splits = jnp.array_split(D_zeroes, num_D_split)
            stats = jnp.array(
                [compute_stats_vmap(D_sub, queries_idx) for D_sub in D_splits]
            ).sum(axis=0)
            stats_normed = stats / D.shape[0]

            return stats_normed

        return compute_statistics_fn

    def get_differentiable_statistics_fn(self):
        stat_fn = self.get_exact_statistics_fn()

        def compute_statistics_fn(D, queries_idx: jnp.ndarray):
            return stat_fn(D, queries_idx)

        return compute_statistics_fn


    





#######################################################
#                                                     #
#               Here is a the RP class                #
#                                                     #
#######################################################

class RelaxedProjection():
    def __init__(
        self,
        dataset: Dataset,
        args: RAPppConfiguration = None,
        stat_dim: List[int] = None,
        D_size: int = 1000
    ):
        """
        dataset: a Dataset class
        args: RAPppConfiguration, containing hyperparamter setting of the mechanism
        stat_dim: list of the dimension of marginals which will be used
        """
        self.dataset = dataset
        self.domain = dataset.domain

        if args is None:
            self.args = RAPppConfiguration(
                iterations=[1000],
                optimizer_learning_rate=[0.01]
            ) # default setting
        else:
            self.args = args
        
        self.stat_dim = stat_dim
        self.selected_query_index = [None] * len(stat_dim)
        self.measured_ans = [None] * len(stat_dim)
        self.contain_workload = [False] * len(stat_dim)
        self._initialize_synthetic(num_generated_points=D_size, seed=0)


    def _initialize_synthetic(self, num_generated_points: int, seed: int):
        """
        Create an initial synthetic dataset D_prime.
        Must be called before fit_workload if no D_prime exists yet.
        """
        key = random.PRNGKey(seed)
        data_dim = sum(self.domain.shape)
        self.D_prime = jnp.asarray(
            initialize_synthetic_dataset(key, num_generated_points, data_dim)
        )

    def store_marginals(self, marginals):
        """
        store marginal info and compile loss function
        """
        self.statistics = [MarginalStatistics(self.domain, dim) for dim in self.stat_dim]
        new_selected_query_index = [None] * len(self.stat_dim)
        new_measured_ans = [None] * len(self.stat_dim)

        for cl, measure in marginals:
            try:
                stat_id = self.stat_dim.index(len(cl))
                self.contain_workload[stat_id] = True

                if new_selected_query_index[stat_id] is None:
                    new_selected_query_index[stat_id] = [self.statistics[stat_id].workload.index(cl)]
                    new_measured_ans[stat_id] = [np.hstack([np.array(measure), np.zeros((self.statistics[stat_id].max_size - len(measure),))])]
                else:
                    new_selected_query_index[stat_id].append(self.statistics[stat_id].workload.index(cl))
                    new_measured_ans[stat_id].append(
                        np.hstack([np.array(measure), np.zeros((self.statistics[stat_id].max_size - len(measure),))])
                    )
            except:
                raise ValueError(f'Unsupported marginal dimension: {len(cl)}')

        for i in range(len(self.stat_dim)):
            if new_selected_query_index[i] is not None:
                if self.selected_query_index[i] is None:
                    self.selected_query_index[i] = jnp.array(new_selected_query_index[i]).reshape(-1)
                    self.measured_ans[i] = jnp.array(new_measured_ans[i])
                else:
                    self.selected_query_index[i] = jnp.vstack([self.selected_query_index[i], jnp.array(new_selected_query_index[i]).reshape(-1)])
                    self.measured_ans[i] = jnp.vstack([self.measured_ans[i], jnp.array(new_measured_ans[i])])

        self._compile_all_loss_update_comb()


    def _compile_all_loss_update(self):
        self.prog_fns = {}
        self.upt_fns = {}
        for i in range(len(self.stat_dim)):
            if not self.contain_workload[i]:
                continue

            exact_fn = self.statistics[i].get_exact_statistics_fn()
            diff_fn  = self.statistics[i].get_differentiable_statistics_fn()
            selected_query_index = self.selected_query_index[i]
            target = self.measured_ans[i]
            

            @jax.jit
            def progress_loss(D_prime, query_idx, target, _exact_fn=exact_fn):
                stats = _exact_fn(D_prime, query_idx)
                return jnp.linalg.norm(target - stats)**2
            prog_fn = jax.jit(lambda D: progress_loss(D, selected_query_index, target))

            @jax.jit
            def loss_fn(D_prime, query_idx, target, _diff_fn=diff_fn):
                stats = _diff_fn(D_prime, query_idx)
                return jnp.linalg.norm(target - stats)**2

            opt_init, opt_update, get_params = optimizers.adam(lambda x: x)
            feats_cum = jnp.array([0] + list(self.domain.shape)).cumsum()
            feats_idx = [list(range(feats_cum[i], feats_cum[i+1])) for i in range(len(feats_cum)-1)]

            @jax.jit
            def update_fn(state, lr, query_idx, target,
                          _loss_fn=loss_fn, _opt_update=opt_update, _get_params=get_params):
                D = _get_params(state)
                value, grads = value_and_grad(_loss_fn, argnums=0)(D, query_idx, target)
                state = _opt_update(lr, grads, state)

                unpack = unpack_optimizer_state(state)
                proj = sparsemax_project(unpack.subtree[0], feats_idx)
                clipped = jnp.clip(proj, 0, 1)
                unpack.subtree = (clipped, unpack.subtree[1], unpack.subtree[2])
                state = pack_optimizer_state(unpack)

                return state, _get_params(state), value
            upt_fn = jax.jit(lambda state, lr: update_fn(state, lr, selected_query_index, target))

            self.prog_fns[i] = prog_fn
            self.upt_fns[i]  = upt_fn

            dummy_state = opt_init(jnp.zeros((1, sum(self.domain.shape))))
            dummy_D = jnp.zeros((1, sum(self.domain.shape)))
            self.prog_fns[i](dummy_D).block_until_ready()
            _, dummy_D_out, _ = self.upt_fns[i](dummy_state, 0.0)
            dummy_D_out.block_until_ready()
    

    def _compile_all_loss_update_comb(self):
        exact_fn, diff_fn, selected_query_index, target = [None]*len(self.stat_dim), [None]*len(self.stat_dim), [None]*len(self.stat_dim), [None]*len(self.stat_dim)
        for i in range(len(self.stat_dim)):
            exact_fn[i] = self.statistics[i].get_exact_statistics_fn()
            diff_fn[i]  = self.statistics[i].get_differentiable_statistics_fn()
            selected_query_index[i] = self.selected_query_index[i]
            target[i] = self.measured_ans[i]
        

        @jax.jit
        def progress_loss(D_prime, query_idx, target, _exact_fn=exact_fn):
            loss = 0.0
            for i in range(len(self.stat_dim)):
                stats = _exact_fn[i](D_prime, query_idx[i])
                loss += jnp.linalg.norm(target[i] - stats)**2
            return loss
        prog_fn = jax.jit(lambda D: progress_loss(D, selected_query_index, target))

        @jax.jit
        def loss_fn(D_prime, query_idx, target, _diff_fn=diff_fn):
            loss = 0.0
            for i in range(len(self.stat_dim)):
                stats = _diff_fn[i](D_prime, query_idx[i])
                loss += jnp.linalg.norm(target[i] - stats)**2
            return loss

        opt_init, opt_update, get_params = optimizers.adam(lambda x: x)
        feats_cum = jnp.array([0] + list(self.domain.shape)).cumsum()
        feats_idx = [list(range(feats_cum[i], feats_cum[i+1])) for i in range(len(feats_cum)-1)]

        @jax.jit
        def update_fn(state, lr, query_idx, target,
                        _loss_fn=loss_fn, _opt_update=opt_update, _get_params=get_params):
            D = _get_params(state)
            value, grads = value_and_grad(_loss_fn, argnums=0)(D, query_idx, target)
            state = _opt_update(lr, grads, state)

            unpack = unpack_optimizer_state(state)
            proj = sparsemax_project(unpack.subtree[0], feats_idx)
            clipped = jnp.clip(proj, 0, 1)
            unpack.subtree = (clipped, unpack.subtree[1], unpack.subtree[2])
            state = pack_optimizer_state(unpack)

            return state, _get_params(state), value
        upt_fn = jax.jit(lambda state, lr: update_fn(state, lr, selected_query_index, target))

        self.prog_fn_comb = prog_fn
        self.upt_fn_comb  = upt_fn

        dummy_state = opt_init(jnp.zeros((1, sum(self.domain.shape))))
        dummy_D = jnp.zeros((1, sum(self.domain.shape)))
        self.prog_fn_comb(dummy_D).block_until_ready()
        _, dummy_D_out, _ = self.upt_fn_comb(dummy_state, 0.0)
        dummy_D_out.block_until_ready()


    def run_sep(self):
        """
        Run many projection round to update self.D_prime on 
        different statistics seperately
        
        This should be used with ``_compile_all_loss_update``
        """
        if self.D_prime is None:
            raise RuntimeError("Call initialize_synthetic(...) before fit_workload.")

        cfg = self.args

        # prepare loss/update functions bound to this module
        for i in range(len(self.stat_dim)):
            if not self.contain_workload[i]: 
                print(f'no {self.stat_dim[i]}-dim marginal stored')
                continue
            print(f'fit {self.stat_dim[i]}-dim marginals')

            prog_fn = self.prog_fns[i]
            upt_fn = self.upt_fns[i]

            progress_loss_start = prog_fn(self.D_prime)

            # one projection round
            self.D_prime = self.projection_mechanism(
                cfg,
                self.D_prime,
                prog_fn,
                upt_fn
            )

            progress_loss_final = prog_fn(self.D_prime)

            print(
                f'dim-{self.stat_dim[i]} marginal fitting results: start loss = {progress_loss_start}, end loss = {progress_loss_final}'
            )


    def run(self):
        """
        Run one projection round to update self.D_prime
        """
        if self.D_prime is None:
            raise RuntimeError("Call initialize_synthetic(...) before fit_workload.")

        cfg = self.args

        # prepare loss/update functions bound to this module
        prog_fn_comb = self.prog_fn_comb
        upt_fn_comb = self.upt_fn_comb

        progress_loss_start = prog_fn_comb(self.D_prime)

        # one projection round
        self.D_prime = self.projection_mechanism(
            cfg,
            self.D_prime,
            prog_fn_comb,
            upt_fn_comb
        )

        progress_loss_final = prog_fn_comb(self.D_prime)

        print(
            f'marginal fitting results: start loss = {progress_loss_start}, end loss = {progress_loss_final}'
        )

    def _get_workload_index(self, workload):
        for i, stat in enumerate(self.statistics):
            idx = stat.get_workload_index(workload)
            if idx >= 0:
                return i, idx 
        raise ValueError(f'Workload {workload} not found')
            

    def query_marginal(self, marginals: List[Tuple]) -> List[np.ndarray]:
        """
        Return the current synthetic answers for this workload.
        """
        if self.D_prime is None:
            raise RuntimeError("No synthetic data: run fit_workload first.")
        res = []
        stat_fn = [self.statistics[k].get_exact_statistics_fn() for k in range(len(self.stat_dim))]
        for cl in marginals:
            marg_size = self.domain.size(cl)
            k, idx = self._get_workload_index(cl)
            cl_res = np.array(stat_fn[k](self.D_prime, jnp.array([idx]))).reshape(-1)
            res.append(cl_res[:marg_size])
        return res


    def projection_mechanism(
        self,
        config: RAPppConfiguration,
        D_init: jnp.ndarray,
        progress_loss_fn: Callable,
        update_fn: Callable,
    ) -> jnp.ndarray:
        D_prime_l2_loss_min = np.inf

        D_prime = None
        for loss_type in config.loss_type:
            opt_init, opt_update, get_params = optimizers.adam(lambda x: x)
            """ Loss function for a single row of D_prime """

            for opt_lr, iters, rp_stop, sigmoid_0, sig_doubles in itertools.product(
                config.optimizer_learning_rate,
                config.iterations,
                config.rap_percent_stopping_condition,
                config.sigmoid_0,
                config.sigmoid_doubles,
            ):
                D_prime = jnp.copy(D_init)

                """ Use SGD with corresponding parameters to find a D' that minimized the loss function using the corresponding parameters  """
                D_prime = run_sgd(
                    D_prime,
                    opt_init,
                    progress_loss_fn,
                    update_fn,
                    opt_lr,
                    iters,
                    rp_stop,
                    sigmoid_0,
                    sig_doubles
                )
                D_prime_progress_loss = progress_loss_fn(D_prime)

                D_prime, D_prime_l2_loss_min = (
                    (D_prime, D_prime_progress_loss)
                    if D_prime_progress_loss < D_prime_l2_loss_min
                    else (D_prime, D_prime_l2_loss_min)
                )

        return D_prime


    def sample(self, num_samples):
        n, _ = self.D_prime.shape
        key = jax.random.PRNGKey(0)
        sampled_indices = jax.random.choice(key, n, shape=(num_samples,), replace=True)
        sampled_data = self.D_prime[sampled_indices]

        result = []
        key = jax.random.PRNGKey(0)
        cum_num_classes = np.cumsum([0] + list(self.domain.shape))
        for i in range(len(cum_num_classes)-1):
            s, e = cum_num_classes[i], cum_num_classes[i+1]
            p = sampled_data[:, s:e]
            key, sk = jax.random.split(key)
            ids = jax.random.categorical(sk, jnp.log(jnp.clip(p, a_min=1e-30, a_max=1.0)), axis=-1)
            result.append(ids)

        sampled_df = pd.DataFrame(np.stack([np.array(r) for r in result], axis=1), columns = list(self.domain.attrs))
        sampled_data = Dataset(sampled_df, self.domain)
        return sampled_data
    
    def _clip_array(self, array):
        return jnp.clip(array, 0, 1)
