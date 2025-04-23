from typing import TypeAlias

Clique: TypeAlias = tuple[str, ...]


def reverse_clique_mapping(
    maximal_cliques: list[Clique], all_cliques: list[Clique]
) -> dict[Clique, list[Clique]]:
    """Creates a mapping from maximal cliques to a list of cliques they contain.

    Args:
      maximal_cliques: A list of maximal cliques.
      all_cliques: A list of all cliques.

    Returns:
      A mapping from maximal cliques to cliques they contain.

    Example doctest:
    >>> maximal_cliques = [('A', 'B', 'C'), ('C', 'D')]
    >>> all_cliques = [('A', 'B'), ('C',), ('D',), ('A', 'C')]
    >>> mapping = reverse_clique_mapping(maximal_cliques, all_cliques)
    >>> sorted(mapping.items())
    [(('A', 'B', 'C'), [('A', 'B'), ('C',), ('A', 'C')]), (('C', 'D'), [('D',)])]
    """
    mapping = {cl: [] for cl in maximal_cliques}
    for cl in all_cliques:
        for cl2 in maximal_cliques:
            if set(cl) <= set(cl2):
                mapping[cl2].append(cl)
                break
    return mapping


def maximal_subset(cliques: list[Clique]) -> list[Clique]:
    """Given a list of cliques, finds a maximal subset of non-nested cliques.

    A clique is considered nested in another if all its vertices are a subset
    of the other's vertices.

    Example Usage:
    >>> maximal_subset([('A', 'B'), ('B',), ('C',), ('B', 'A')])
    [('A', 'B'), ('C',)]

    Args:
      cliques: A list of cliques.

    Returns:
      A new list containing a maximal subset of non-nested cliques.
    """
    cliques = sorted(cliques, key=len, reverse=True)
    result = []
    for cl in cliques:
        if not any(set(cl) <= set(cl2) for cl2 in result):
            result.append(cl)
    return result


def clique_mapping(
    maximal_cliques: list[Clique], all_cliques: list[Clique]
) -> dict[Clique, Clique]:
    """Creates a mapping from cliques to their corresponding maximal clique.

    Example Usage:
    >>> maximal_cliques = [('A', 'B'), ('B', 'C')]
    >>> all_cliques = [('B', 'A'), ('B',), ('C',), ('B', 'C')]
    >>> mapping = clique_mapping(maximal_cliques, all_cliques)
    # Convert to sorted items for stable comparison in doctest
    >>> sorted(mapping.items())
    [(('B',), ('A', 'B')), (('B', 'A'), ('A', 'B')), (('B', 'C'), ('B', 'C')), (('C',), ('B', 'C'))]


    Args:
      maximal_cliques: A list of maximal cliques.
      all_cliques: A list of all cliques.

    Returns:
      A mapping from cliques to their maximal clique.

    """
    mapping = {}
    # Ensure maximal_cliques are unique and sorted for deterministic mapping if multiple contain a clique
    maximal_cliques = sorted(list(set(maximal_cliques)))
    for cl in all_cliques:
        for cl2 in maximal_cliques:
            if set(cl) <= set(cl2):
                mapping[cl] = cl2
                break
    return mapping
