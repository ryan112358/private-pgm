import string
import jax
import jax.numpy as jnp
from mbi import CliqueVector, Domain, Factor, junction_tree_new
from mbi.marginal_loss import clique_mapping
import functools
import collections

_EINSUM_LETTERS = list(string.ascii_lowercase) + list(string.ascii_uppercase)


def sum_product(factors: list[Factor], dom: Domain) -> Factor:
    """Compute the sum-product of a list of factors."""
    attrs = sorted(set.union(*[set(f.domain) for f in factors]).union(set(dom)))
    mapping = dict(zip(attrs, _EINSUM_LETTERS))
    convert = lambda d: "".join(mapping[a] for a in d.attributes)
    formula = ",".join(convert(f.domain) for f in factors) + "->" + convert(dom)
    values = jnp.einsum(
        formula, *[f.values for f in factors], precision=jax.lax.Precision.HIGHEST
    )
    return Factor(dom, values)


def logspace_sum_product(potentials: list[Factor], dom: Domain) -> Factor:
    maxes = [f.max(f.domain.marginalize(dom).attributes) for f in potentials]
    stable_potentials = [(f - m).exp() for f, m in zip(potentials, maxes)]
    return sum_product(stable_potentials, dom).log() + sum(maxes)


def brute_force_marginals(potentials: CliqueVector, total: float = 1) -> CliqueVector:
    P = sum(potentials.values()).normalize(total, log=True).exp()
    return CliqueVector({cl: P.project(cl) for cl in potentials})


def einsum_marginals(potentials: CliqueVector, total: float = 1) -> CliqueVector:
    inputs = list(potentials.values())
    return CliqueVector(
        {
            cl: logspace_sum_product(inputs, potentials[cl].domain)
            .normalize(total, log=True)
            .exp()
            for cl in potentials.keys()
        }
    )

def message_passing(
    potentials: CliqueVector,
    total: float = 1,
) -> CliqueVector:
  """Message passing marginal inference."""
  cliques = list(potentials.keys())
  domain = potentials.domain

  jtree = junction_tree_new.make_junction_tree(domain, cliques)[0]
  message_order = junction_tree_new.message_passing_order(jtree)
  maximal_cliques = junction_tree_new.maximal_cliques(jtree)

  mapping = clique_mapping(maximal_cliques, cliques)

  beliefs = CliqueVector.zeros(domain, maximal_cliques)
  for cl in cliques:
    beliefs[mapping[cl]] = beliefs[mapping[cl]] + potentials[cl]

  messages = {}
  for i, j in message_order:
    sep = beliefs[i].domain.invert(tuple(set(i) & set(j)))
    if (j, i) in messages:
      tau = beliefs[i] - messages[(j, i)]
    else:
      tau = beliefs[i]
    messages[(i, j)] = tau.logsumexp(sep)
    beliefs[j] = beliefs[j] + messages[(i, j)]

  beliefs = beliefs.normalize(total, log=True).exp()

  result = {}
  for cl in cliques:
    result[cl] = beliefs[mapping[cl]].project(cl)

  return CliqueVector(result)


def message_passing_new(
    potentials: CliqueVector,
    total: float = 1,
) -> CliqueVector:
  """Message passing marginal inference."""
  cliques = list(potentials.keys())
  domain = potentials.domain

  jtree = junction_tree_new.make_junction_tree(domain, cliques)[0]
  message_order = junction_tree_new.message_passing_order(jtree)
  maximal_cliques = junction_tree_new.maximal_cliques(jtree)

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

  return CliqueVector(beliefs)


def variable_elimination(potentials: CliqueVector, clique: tuple[str, ...], total: float=1) -> Factor:
  cliques = potentials.cliques + [clique]
  domain = potentials.domain
  elim = domain.invert(clique)
  elim_order, _ = junction_tree_new.greedy_order(domain, cliques, elim=elim)

  k = len(potentials)
  psi = dict(zip(range(k), potentials.values()))
  for z in elim_order:
    psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
    psi[k] = sum(psi2).logsumexp([z])
    k += 1
  return sum(psi.values()).normalize(total, log=True).exp().project(clique)


