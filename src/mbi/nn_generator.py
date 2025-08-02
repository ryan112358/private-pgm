import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import itertools
import functools
from typing import Sequence


#######################################################
#                                                     #
#           Here is a simple MLP generator            #
#                                                     #
#######################################################

class SimpleResidual(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        out = nn.Dense(self.out_dim)(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        return jnp.concatenate([out, x], axis=-1)


class Generator(nn.Module):
    embedding_dim: int
    gen_dims: Sequence[int]
    data_dim: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        for d in self.gen_dims:
            x = SimpleResidual(out_dim=d)(x, train)
        x = nn.Dense(self.data_dim)(x)
        return x




#######################################################
#                                                     #
#         Here is a the NN generator class            #
#                                                     #
#######################################################

class MargNN:
    def __init__(
            self, 
            domain: dict, 
            batch_size: int = 512,
            lr: float = 1e-3,
            iterations: int = 300, 
            embd_dims: int = None,
            gen_dims: tuple = None,
            seed: int = 0, 
            resample: bool = False
        ):
        """
        This is the class of neural network generator. The generator is constructed 
        on a MLP structure

        Args:
            domain: A dictionary of attributes and their domain size.
            batch_size: Training batch size. Default to 512.
            lr: Learning rate. Default (also recommend) to 1e-3.
            iterations: Number of training iterations each time execute train_model. 
                Default to 200.
            embd_dims: Dimensionalities for NN generator's input dim. Default to 
                data_dims, which is the output dim of NN.
            gen_dims: A tuple representing the dimensionalities of layers in the 
                generator.
            seed: Random seed for sampling.
            resample: A bool value, defining whether we need to resample input 
                each time we generate data.
        """
        self.column_dims = domain
        self.num_classes = np.array(list(domain.values()))
        self.column_names = np.array(list(domain.keys()))
        self.cum_num_classes = np.concatenate((np.array([0]), np.cumsum(self.num_classes)))
        self.batch_size = batch_size 
        self.resample = resample

        data_dims = self.cum_num_classes[-1]
        if embd_dims is None: embd_dims = data_dims
        if gen_dims is None: gen_dims = (data_dims, data_dims)

        self.key = jax.random.PRNGKey(seed)
        self.model = Generator(
            embedding_dim = embd_dims,
            gen_dims = gen_dims,
            data_dim = data_dims
        )
        dummy = jnp.zeros((self.batch_size, embd_dims))
        variables = self.model.init(self.key, dummy, train=True)
        self.params, self.batch_stats = variables['params'], variables['batch_stats']

        self.lr = lr
        self.iterations = iterations
        self.z = self._uniform_sample()


    def _find_query_index(self, marginals):
        index, answer, size, weight = [], [], [], []
        for marg, matrix, w in marginals:
            starts = [
                self.cum_num_classes[np.where(self.column_names == col)[0][0]]
                for col in marg
            ]
            ends = [s + self.column_dims[col] for s, col in zip(starts, marg)]

            index += list(itertools.product(*(range(s, e) for s, e in zip(starts, ends))))
            answer += matrix.tolist()
            size += [1/matrix.size] * matrix.size
            weight += [w] * matrix.size

        answer = jnp.array(answer, dtype=jnp.float64)
        size = jnp.array(size, dtype=jnp.float64)
        weight = jnp.array(weight, dtype=jnp.float64)
        return index, answer, size, weight


    def _merge_marginals(self, marginals):
        merged = {}
        for name, matrix, w in marginals:
            if name not in merged:
                merged[name] = (matrix * w, w)
            else:
                total_mat, total_w = merged[name]
                merged[name] = (total_mat + matrix * w, total_w + w)
        result = []
        for name, (total_mat, total_w) in merged.items():
            result.append((name, total_mat / total_w, total_w))
        return result


    def store_marginals(self, marginals):
        merged = self._merge_marginals(marginals)
        self.queries, self.real_answers, self.query_size, self.query_weight = self._find_query_index(merged)

        K, D = len(self.queries), int(self.cum_num_classes[-1])
        lengths = [len(q) for q in self.queries]
        rows = jnp.repeat(jnp.arange(K), jnp.array(lengths))
        cols = jnp.concatenate([jnp.array(q, dtype=jnp.int32) for q in self.queries])

        mask = jnp.zeros((K, D), dtype=jnp.float32)
        mask = mask.at[rows, cols].set(1.0)
        self.Q_mask = mask


    def _uniform_sample(self):
        if not hasattr(self, 'z') or self.resample:
            parts = []
            for i in range(len(self.cum_num_classes) - 1):
                s, e = self.cum_num_classes[i], self.cum_num_classes[i+1]
                self.key, subkey = jax.random.split(self.key)
                idxs = jax.random.randint(subkey, (self.batch_size,), 0, e - s)
                onehot = jax.nn.one_hot(idxs, e - s)
                onehot = jnp.clip(onehot, 1e-30, 1 - (e - s) * 1e-30)
                parts.append(onehot)
            return jnp.concatenate(parts, axis=1)
        else:
            return self.z


    def _predict_x(self, output):
        outs = []
        for i in range(len(self.cum_num_classes) - 1):
            s, e = self.cum_num_classes[i], self.cum_num_classes[i+1]
            logits = jax.nn.log_softmax(output[:, s:e], axis=-1)
            outs.append(jnp.clip(logits, -30.0, 0.0))
        return jnp.concatenate(outs, axis=1)


    def train_model(self):
        self.opt = optax.adam(self.lr)
        self.opt_state = self.opt.init(self.params)

        # --- capture static variables ---
        model = self.model
        predict_x = self._predict_x

        # --- core training function ---
        @jax.jit
        def train_step(variables, opt_state, z, Q_mask, real_answers, query_weight):
            def loss_fn(params, batch_stats, Q_mask, real_answers, query_weight):
                out, new_state = model.apply(
                    {'params': params, 'batch_stats': batch_stats},
                    z,
                    train=True,
                    mutable=['batch_stats'],
                )
                x_pred = predict_x(out)
                S      = x_pred @ Q_mask.T
                syn    = jnp.exp(S).mean(axis=0)
                loss   = jnp.sum(query_weight * (syn - real_answers) ** 2)
                return loss, new_state['batch_stats']
        
            (loss, new_batch_stats), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(variables['params'], variables['batch_stats'], Q_mask, real_answers, query_weight)

            updates, opt_state = self.opt.update(grads, opt_state)
            new_params = optax.apply_updates(variables['params'], updates)

            new_variables = {'params': new_params, 'batch_stats': new_batch_stats}
            return new_variables, opt_state, loss


        # --- main training process ---
        variables = {'params': self.params, 'batch_stats': self.batch_stats}
        for iter in range(self.iterations):
            z = self._uniform_sample()

            variables, self.opt_state, loss = train_step(
                variables, 
                self.opt_state, 
                z,
                self.Q_mask,
                self.real_answers,
                self.query_weight
            )
            self.params, self.batch_stats = variables['params'], variables['batch_stats']

            if iter == 0: print('Start Accumulated Loss:', loss)
            elif iter == self.iterations-1: print('End Accumulated Loss:', loss)




    def sample(self, num_samples):
        z = self._uniform_sample()
        out = self.model.apply(
            {'params': self.params, 'batch_stats': self.batch_stats},
            z,
            train=False
        )
        x_pred = jnp.exp(self._predict_x(out))

        self.key, subkey = jax.random.split(self.key)
        idx = jax.random.randint(subkey, (num_samples,), 0, x_pred.shape[0])
        batch = x_pred[idx]

        result = []
        for i in range(len(self.num_classes)):
            s, e = self.cum_num_classes[i], self.cum_num_classes[i+1]
            p = batch[:, s:e]
            self.key, sk = jax.random.split(self.key)
            ids = jax.random.categorical(sk, jnp.log(p), axis=-1) # categorical requires log prob
            result.append(ids)
        return np.stack([np.array(r) for r in result], axis=1)


    def obtain_sample_marginals(self, marginals):
        z = self._uniform_sample()
        out = self.model.apply(
            {'params': self.params, 'batch_stats': self.batch_stats},
            z,
            train=False
        )
        x_pred = jnp.exp(self._predict_x(out))

        return [self._map_to_marginal(x_pred, m) for m in marginals]
    

    def _map_to_marginal(self, x_pred, marginal):
        starts = [
            self.cum_num_classes[np.where(self.column_names == col)[0][0]]
            for col in marginal
        ]
        ends = [s + self.column_dims[col] for s, col in zip(starts, marginal)]
        splits = [x_pred[:, s:e] for s, e in zip(starts, ends)]

        input_dims = ','.join(f'b{chr(105 + i)}' for i in range(len(splits)))
        output_dims = ''.join(f'{chr(105 + i)}' for i in range(len(splits)))
        einsum_str = f'{input_dims}->b{output_dims}'

        joint = jnp.einsum(einsum_str, *splits).mean(axis=0)
        return np.array(joint).flatten()