"""Functions for computing marginals from log-space potentials.

The functions in this library should all produce numerically identical
outputs on well-behaved inputs, but may have different stability characteristics
on poorly-behaved inputs, and different rutnime/memory performance characteristics.

The currently supported marginal_oracles in this libarary are:
    * brute_force_marginals
    * einsum_marginals
    * message_passing_stable
    * message_passing_fast

We recommend using message_passing_stable with accelerated estimation algorithms like
Interior Gradient, but using message_passing_fast with mirror descent.  A marginal oracle
is a function that consumes a vector of log-space potentials over an arbitrary set of cliques
and returns a vector of marginals defined over those same set of cliques.
"""

import string
import jax
import jax.numpy as jnp
from mbi import CliqueVector, Domain, Factor, junction_tree
from mbi.marginal_loss import clique_mapping
import functools
import collections

_EINSUM_LETTERS = list(string.ascii_lowercase) + list(string.ascii_uppercase)


def sum_product(factors: list[Factor], dom: Domain) -> Factor:
    """Compute the sum-of-products of a list of Factors using einsum.

    Args:
        factors: A list of Factors.
        dom: The target domain of the output factor.

    Returns:
        sum_{S \ D} prod_i F_i,
        where
            * F_i = factors[i]
            * D = dom
            * S = union of domains of F_i
    """

    attrs = sorted(set.union(*[set(f.domain) for f in factors]).union(set(dom)))
    mapping = dict(zip(attrs, _EINSUM_LETTERS))
    convert = lambda d: "".join(mapping[a] for a in d.attributes)
    formula = ",".join(convert(f.domain) for f in factors) + "->" + convert(dom)
    #print(jnp.einsum_path(formula, *[f.values for f in factors]))
    values = jnp.einsum(
        formula,
        *[f.values for f in factors],
        optimize="dp",  # default setting broken in some cases
        precision=jax.lax.Precision.HIGHEST
    )
    return Factor(dom, values)


def logspace_sum_product(log_factors: list[Factor], dom: Domain) -> Factor:
    """Numerically stable algorithm for computing sum product in log space.

    This seems to be the most stable algorithm for doing this computation that doesn't
    require materializing sum(log_factors).  Materializing sum(log_factors) will
    in general give better numerical stability, but it comes at the cost of
    increased memory usage.

    TODO(ryan112358): consider using jax.lax.scan and/or jax.lax.map 
    to compute log sum product sequentially.  This will also have the dual
    benefit of using less memory than jnp.einsum in some hard cases.

    https://github.com/jax-ml/jax/issues/24915

    https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy

    Args:
        log_factors: a list of log-space factors.
        dom: The desired domain of the output factor.

    Returns:
        log sum_{S \ D} prod_i exp(F_i),
        where 
            * F_i = log_factors[i], 
            * D is the input domain,
            * S is the union of the domains of F_i
    """
    maxes = [f.max(f.domain.marginalize(dom).attributes) for f in log_factors]
    stable_factors = [(f - m).exp() for f, m in zip(log_factors, maxes)]
    return sum_product(stable_factors, dom).log() + sum(maxes)


def logspace_sum_product_very_stable(log_factors: list[Factor], dom: Domain) -> Factor:
    """More stable implementation of logspace_sum_product.
        
    This ipmlementation may (or may not) materialize a Factor over the domain
    of all elements of log_factors.  Without JIT, it will materialize this "super-factor".
    Under JIT, there may be some instances where the compiler can figure out
    that it does not need to materialize this intermediate to compute the final output.
    """
    summed = sum(log_factors)
    return summed.logsumexp(summed.domain.marginalize(dom).attributes)


def brute_force_marginals(potentials: CliqueVector, total: float = 1) -> CliqueVector:
    """Compute marginals from (log-space) potentials by materializing the full joint distribution."""
    P = sum(potentials.arrays.values()).normalize(total, log=True).exp()
    marginals = {cl: P.project(cl) for cl in potentials.cliques}
    return CliqueVector(potentials.domain, potentials.cliques, marginals)


def einsum_marginals(potentials: CliqueVector, total: float = 1) -> CliqueVector:
    """Compute marginals from (log-space) potentials by using einsum.

    This is a "brute-force" approach and is not recommended in practice.
    """
    inputs = list(potentials.arrays.values())
    return CliqueVector(
        potentials.domain,
        potentials.cliques,
        {
            cl: logspace_sum_product(inputs, potentials[cl].domain)
            .normalize(total, log=True)
            .exp()
            for cl in potentials.cliques
        },
    )


@jax.jit
def message_passing_stable(potentials: CliqueVector, total: float = 1) -> CliqueVector:
    """Compute marginals from (log-space) potentials using the message passing algorithm.

    This implementation operates completely in logspace, until the last step where it
    exponentiates the log-beliefs to get marginals.  It is very stable numerically,
    but in general could materialize factors defined over "super-cliques", which
    are the nodes in the junction tree implied by the cliques in potentials.
    Thus, it may require more memory than "message_passing_fast" below.

    Args:
        potentials: The (log-space) potentials of a graphical model.
        total: The normalization factor.

    Returns:
        The marginals of the graphical model, defined over the same set of cliques
        as the input potentials.  Each marginal is non-negative and sums to "total".
    """
    domain, cliques = potentials.domain, potentials.cliques

    jtree = junction_tree.make_junction_tree(domain, cliques)[0]
    message_order = junction_tree.message_passing_order(jtree)
    maximal_cliques = junction_tree.maximal_cliques(jtree)

    mapping = clique_mapping(maximal_cliques, cliques)
    beliefs = potentials.expand(maximal_cliques)

    messages = {}
    for i, j in message_order:
        sep = beliefs[i].domain.invert(tuple(set(i) & set(j)))
        if (j, i) in messages:
            tau = beliefs[i] - messages[(j, i)]
        else:
            tau = beliefs[i]
        messages[(i, j)] = tau.logsumexp(sep)
        beliefs[j] = beliefs[j] + messages[(i, j)]

    return beliefs.normalize(total, log=True).exp().contract(cliques)

@jax.jit
def message_passing_fast(potentials: CliqueVector, total: float = 1) -> CliqueVector:
    """Compute marginals from (log-space) potentials using the message passing algorithm.

    This implementation leverages the "einsum" primitve to compute clique marginals
    without materializing marginals over the super cliques first (nodes in the
    junction tree).  It can be much faster and more memory efficient than
    message_passing_stable, but there are some cases where this
    implementation is not as stable.

    See the stackoverflow thread for the key difficulty here.
    https://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy

    Args:
        potentials: The (log-space) potentials of a graphical model.
        total: The normalization factor.

    Returns:
        The marginals of the graphical model, defined over the same set of cliques
        as the input potentials.  Each marginal is non-negative and sums to "total".
    """
    domain, cliques = potentials.active_domain, potentials.cliques

    jtree = junction_tree.make_junction_tree(domain, cliques)[0]
    message_order = junction_tree.message_passing_order(jtree)
    # TODO: upstream this logic to message_passing_order function
    message_order = [(i, j) for i, j in message_order if len(set(i) & set(j)) > 0]
    maximal_cliques = junction_tree.maximal_cliques(jtree)

    mapping = clique_mapping(maximal_cliques, cliques)
    inverse_mapping = collections.defaultdict(list)
    incoming_messages = collections.defaultdict(list)
    potential_mapping = collections.defaultdict(list)

    for cl in cliques:
        potential_mapping[mapping[cl]].append(potentials[cl])
        inverse_mapping[mapping[cl]].append(cl)

    for i in range(len(message_order)):
        msg = message_order[i]
        for j in range(i):
            msg2 = message_order[j]
            if msg[0] == msg2[1] and msg[1] != msg2[0]:
                incoming_messages[msg].append(msg2)

    messages = {}
    for i, j in message_order:
        shared = domain.project(tuple(set(i) & set(j)))
        input_potentials = potential_mapping[i]
        input_messages = [messages[key] for key in incoming_messages[(i, j)]]
        inputs = input_potentials + input_messages
        messages[(i, j)] = logspace_sum_product(inputs, shared)

    beliefs = {}
    for cl in maximal_cliques:
        input_potentials = potential_mapping[cl]
        input_messages = [messages[key] for key in messages if key[1] == cl]
        inputs = input_potentials + input_messages
        for cl2 in inverse_mapping[cl]:
            beliefs[cl2] = (
                logspace_sum_product(inputs, domain.project(cl2))
                .normalize(total, log=True)
                .exp()
            )

    return CliqueVector(potentials.domain, cliques, beliefs)


def variable_elimination(
    potentials: CliqueVector, clique: tuple[str, ...], total: float = 1
) -> Factor:
    """Compute an out-of-model/unsupported marginal from the potentials.

    Args:
        potentials: The (log-space) potentials of a Graphical Model.
        clique: The subset of attributes whose marginal you want.
        total: The normalization factor.

    Returns:
        The marginal defined over the domain of the input clique, where
        each entry is non-negative and sums to the input total.
    """
    clique = tuple(clique)
    cliques = potentials.cliques + [clique]
    domain = potentials.active_domain
    elim = domain.invert(clique)
    elim_order, _ = junction_tree.greedy_order(domain, cliques, elim=elim)

    k = len(potentials.cliques)
    psi = dict(zip(range(k), potentials.arrays.values()))
    for z in elim_order:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        psi[k] = sum(psi2).logsumexp([z])
        k += 1
    # this expand covers the case when clique is not in the active domain
    newdom = potentials.domain.project(clique)
    return (
        sum(psi.values())
        .expand(newdom)
        .normalize(total, log=True)
        .exp()
        .project(clique)
    )
