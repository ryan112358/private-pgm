"""Approximate marginal oracles with convex counting numbers.

See the paper ["Relaxed Marginal Consistency for Differentially Private Query Answering"](https://arxiv.org/pdf/2109.06153) for more details.

This file implements one approximate marginal inference oracle: Convex-GBP
with fixed counting numbers of 1.0 for all regions.  We experimented with
others, but do not officially support them in this library.  If interested,
please see the following snapshot of this repository:

https://github.com/ryan112358/private-pgm/tree/approx-experiments-snapshot

Pull requests are welcome to add support for other approximate oracles.
"""

import networkx as nx
import itertools
from scipy.cluster.hierarchy import DisjointSet
from mbi import Domain, CliqueVector, Factor


def build_graph(domain: Domain, cliques: list[tuple[str, ...]]) -> ...:
    # Hard-code minimal=True, convex=True
    # Counting numbers = 1 for all regions
    # Alg 11.3 of Koller & Friedman
    regions = set(cliques)
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
            if set(r2) < set(r1) and not any(
                set(r2) < set(r3) and set(r3) < set(r1) for r3 in regions
            ):
                G.add_edge(r1, r2)

    H = G.reverse()
    G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

    children = {r: list(G.neighbors(r)) for r in regions}
    parents = {r: list(H.neighbors(r)) for r in regions}
    descendants = {r: list(G1.neighbors(r)) for r in regions}
    ancestors = {r: list(H1.neighbors(r)) for r in regions}
    forebears = {r: set([r] + ancestors[r]) for r in regions}
    downp = {r: set([r] + descendants[r]) for r in regions}

    min_edges = []
    for r in regions:
        ds = DisjointSet()
        for u in parents[r]:
            ds.add(u)
        for u, v in itertools.combinations(parents[r], 2):
            uv = set(ancestors[u]) & set(ancestors[v])
            if len(uv) > 0:
                ds.merge(u, v)
        canonical = set()
        for u in parents[r]:
            canonical.update({ds[u]})
        min_edges.extend([(u, r) for u in canonical])

    G = nx.DiGraph()
    G.add_nodes_from(regions)
    G.add_edges_from(min_edges)

    H = G.reverse()
    G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

    children = {r: list(G.neighbors(r)) for r in regions}
    parents = {r: list(H.neighbors(r)) for r in regions}

    messages = {}
    message_order = []
    for ru in sorted(regions, key=len):
        for rd in children[ru]:
            message_order.append((ru, rd))
            messages[ru, rd] = Factor.zeros(domain.project(rd))
            messages[rd, ru] = Factor.zeros(domain.project(rd))  # only for hazan et al

    return regions, cliques, messages, message_order, parents, children


def convex_generalized_belief_propagation(
    potentials: CliqueVector, total: float = 1, iters: int = 1, damping: float = 0.5
) -> CliqueVector:
    """Convex generalized belief propagation for approximmate marginal inference.

        The algorithms implements the Algorithm 2 in our paper
        ["Relaxed Marginal Consistency for Differentially Private Query Answering"](https://arxiv.org/pdf/2109.06153), which itself is based on the paper titled
        ["Tightening Fractional Covering Upper Bounds on the Partition
    Function for High-Order Region Graphs"](https://arxiv.org/pdf/1210.4881).

    Args:
        potentials: A CliqueVector object containing the potentials of the graphical model.
        total: The total number of records in the dataset.
        iters: The number of iterations to run the algorithm.
        damping: The damping factor for the messages.

    Returns:
        A CliqueVector of pseudo-marginals for the cliques in the graphical model.
    """
    domain, cliques = potentials.domain, potentials.cliques
    regions, cliques, messages, message_order, parents, children = build_graph(
        domain, cliques
    )

    # Hardcode assumption that counting numbers are 1.0 for all regions.
    pot = potentials.expand(regions)

    cc = {}
    for r in regions:
        for p in parents[r]:
            cc[p, r] = 1 / (1 + len(parents[r]))

    for _ in range(iters):
        new = {}
        for r in regions:
            for p in parents[r]:
                new[p, r] = (
                    (
                        pot[p]
                        + sum(messages[c, p] for c in children[p] if c != r)
                        - sum(messages[p, p1] for p1 in parents[p])
                    )
                    .project(r, log=True)
                    .normalize(log=True)
                )

        for r in regions:
            for p in parents[r]:
                new[r, p] = (
                    cc[p, r]
                    * (
                        pot[r]
                        + sum(messages[c, r] for c in children[r])
                        + sum(messages[p1, r] for p1 in parents[r])
                    )
                    - messages[p, r]
                ).normalize(log=True)

        # Damping is not described in paper, but is needed to get convergence for dense graphs
        rho = damping
        for p in regions:
            for r in children[p]:
                messages[p, r] = rho * messages[p, r] + (1.0 - rho) * new[p, r]
                messages[r, p] = rho * messages[r, p] + (1.0 - rho) * new[r, p]
        mu = {}
        for r in cliques:
            mu[r] = (
                (
                    pot[r]
                    + sum(messages[c, r] for c in children[r])
                    - sum(messages[r, p] for p in parents[r])
                )
                .normalize(total, log=True)
                .exp()
            )

    return CliqueVector(domain, cliques, mu)
