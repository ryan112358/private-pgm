from collections.abc import Collection, Iterator
import functools

import attr


@attr.dataclass(frozen=True)
class Domain:
  """Dataclass for representing discrete domains."""
  attributes: tuple[str, ...] = attr.field(converter=tuple)
  shape: tuple[int, ...] = attr.field(converter=tuple)

  def __post_init__(self):
    if len(self.attributes) == len(self.shape):
      raise ValueError('Dimensions must be equal.')
    if len(self.attributes) != len(set(self.attributes)):
      raise ValueError('Attributes must be unique.')

  @functools.cached_property
  def config(self) -> dict[str, int]:
    """Returns a dictionary of { attr : size } values."""
    return dict(zip(self.attributes, self.shape))

  @staticmethod
  def fromdict(config: dict[str, int]) -> 'Domain':
    """Construct a Domain object from a dictionary of { attr : size } values.

    Example Usage:
    >>> print(Domain.fromdict({'a': 10, 'b': 20}))
    Domain(a: 10, b: 20)

    Args:
      config: a dictionary of { attr : size } values
    Returns:
      the Domain object
    """
    return Domain(config.keys(), config.values())

  def project(self, attributes: str | Collection[str]) -> 'Domain':
    """Project the domain onto a subset of attributes.

    Args:
      attributes: the attributes to project onto
    Returns:
      the projected Domain object
    """
    if isinstance(attributes, str):
      attributes = [attributes]
    if not set(attributes) <= set(self.attributes):
      raise ValueError(f'Cannot project {self} onto {attributes}.')
    shape = tuple(self.config[a] for a in attributes)
    return Domain(attributes, shape)

  def marginalize(self, attrs: Collection[str]) -> 'Domain':
    """Marginalize out some attributes from the domain (opposite of project).

    Example Usage:
    >>> D1 = Domain(['a','b'], [10,20])
    >>> print(D1.marginalize(['a']))
    Domain(b: 20)

    Args:
      attrs: the attributes to marginalize out.
    Returns:
      the marginalized Domain object
    """
    proj = [a for a in self.attributes if a not in attrs]
    return self.project(proj)

  def contains(self, other: 'Domain') -> bool:
    """Determine if this domain contains another."""
    return set(other.attributes) <= set(self.attributes)

  def canonical(self, attrs):
    """Return the canonical ordering of the attributes."""
    return tuple(a for a in self.attributes if a in attrs)

  def invert(self, attrs):
    """ returns the attributes in the domain not in the list """
    return [a for a in self.attributes if a not in attrs]

  def intersect(self, other: 'Domain') -> 'Domain':
    """Intersect this Domain object with another.

    Example Usage:
    >>> D1 = Domain(['a','b'], [10,20])
    >>> D2 = Domain(['b','c'], [20,30])
    >>> print(D1.intersect(D2))
    Domain(b: 20)

    Args:
      other: another Domain object
    Returns:
      the intersection of the two domains
    """
    return self.project([a for a in self.attributes if a in other.attributes])

  def axes(self, attrs: Collection[str]) -> tuple[int, ...]:
    """Return the axes tuple for the given attributes.

    Args:
      attrs: the attributes
    Returns:
      a tuple with the corresponding axes
    """
    return tuple(self.attributes.index(a) for a in attrs)

  def merge(self, other: 'Domain') -> 'Domain':
    """Merge this Domain object with another.

    :param other: another Domain object
    :return: a new domain object covering the full domain

    Example:
    >>> D1 = Domain(['a','b'], [10,20])
    >>> D2 = Domain(['b','c'], [20,30])
    >>> print(D1.merge(D2))
    Domain(a: 10, b: 20, c: 30)

    Args:
      other: another Domain object
    Returns:
      a new domain object covering the combined domain.
    """
    extra = other.marginalize(self.attributes)
    return Domain(self.attributes + extra.attributes, self.shape + extra.shape)

  def size(self, attributes: Collection[str] | None = None) -> int:
    """Return the total size of the domain.

    Example:
    >>> D1 = Domain(['a','b'], [10,20])
    >>> D1.size()
    200
    >>> D1.size(['a'])
    10

    Args:
      attributes: A subset of attributes whose total size should be returned.
    Returns:
      the total size of the domain
    """
    if attributes is None:
      return functools.reduce(lambda x, y: x * y, self.shape, 1)
    return self.project(attributes).size()

  @property
  def attrs(self):
    return self.attributes

  def __contains__(self, name: str) -> bool:
    """Check if the given attribute is in the domain."""
    return name in self.attributes

  def __getitem__(self, a: str) -> int:
    return self.config[a]

  def __iter__(self) -> Iterator[str]:
    return self.attributes.__iter__()

  def __len__(self) -> int:
    return len(self.attributes)

  def __str__(self) -> str:
    inner = ', '.join(['%s: %d' % x for x in zip(self.attributes, self.shape)])
    return 'Domain(%s)' % inner
