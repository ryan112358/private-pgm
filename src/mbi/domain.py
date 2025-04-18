from collections.abc import Sequence, Iterator
import functools

import attr


@attr.dataclass(frozen=True)
class Domain:
    """Represents the domain of a discrete dataset.

    A domain defines the attributes (variables) and their possible discrete values
    (represented by their counts or shapes). It is used to manage the structure
    of datasets and marginal tables.

    Attributes:
        attributes: A tuple of strings representing the names of the attributes
            (columns or variables) in the domain.
        shape: A tuple of integers representing the number of unique values (cardinality)
            for each corresponding attribute in `attributes`.
    """

    attributes: tuple[str, ...] = attr.field(converter=tuple)
    shape: tuple[int, ...] = attr.field(converter=tuple)

    def __attrs_post_init__(self):
        """Validates the Domain object after initialization.

        Ensures that the number of attributes matches the number of shape dimensions
        and that all attribute names are unique.

        Raises:
            ValueError: If the length of `attributes` and `shape` do not match,
                or if `attributes` contains duplicate names.
        """
        if len(self.attributes) != len(self.shape):
            raise ValueError("Dimensions must be equal.")
        if len(self.attributes) != len(set(self.attributes)):
            raise ValueError("Attributes must be unique.")

    @functools.cached_property
    def config(self) -> dict[str, int]:
        """Returns a dictionary mapping attribute names to their sizes (cardinalities).

        Returns:
            A dictionary where keys are attribute names (str) and values are their
            corresponding sizes (int).
        """
        return dict(zip(self.attributes, self.shape))

    @staticmethod
    def fromdict(config: dict[str, int]) -> "Domain":
        """Constructs a Domain object from a dictionary.

        Example Usage:
            >>> domain = Domain.fromdict({'a': 10, 'b': 20})
            >>> print(domain)
            Domain(a: 10, b: 20)

        Args:
            config: A dictionary mapping attribute names (str) to their sizes (int).

        Returns:
            A new Domain object created from the provided dictionary.
        """
        return Domain(config.keys(), config.values())

    def project(self, attributes: str | Sequence[str]) -> "Domain":
        """Projects the domain onto a subset of attributes.

        Creates a new Domain containing only the specified attributes and their
        corresponding shapes.

        Args:
            attributes: A single attribute name (str) or a sequence of attribute
                names (Sequence[str]) to project onto.

        Returns:
            A new Domain object representing the projection.

        Raises:
            ValueError: If any of the specified attributes are not present in this
                domain.
        """
        if isinstance(attributes, str):
            attributes = [attributes]
        if not set(attributes) <= set(self.attributes):
            raise ValueError(f"Cannot project {self} onto {attributes}.")
        shape = tuple(self.config[a] for a in attributes)
        return Domain(attributes, shape)

    def marginalize(self, attrs: Sequence[str]) -> "Domain":
        """Marginalizes out specified attributes from the domain.

        This is the opposite of `project`. It creates a new Domain containing all
        attributes *except* the ones specified.

        Example Usage:
            >>> d1 = Domain(('a', 'b'), (10, 20))
            >>> print(d1.marginalize(['a']))
        Domain(b: 20)

        Args:
            attrs: A sequence of attribute names (Sequence[str]) to remove.

        Returns:
            A new Domain object representing the domain after marginalization.
        """
        proj = [a for a in self.attributes if a not in attrs]
        return self.project(proj)

    def contains(self, other: "Domain") -> bool:
        """Checks if this domain completely contains another domain.

        Containment means that all attributes of the `other` domain are present
        in this domain. The shapes associated with common attributes are not checked.

        Args:
            other: The Domain object to check for containment.

        Returns:
            True if all attributes of `other` are present in this domain,
            False otherwise.
        """
        return set(other.attributes) <= set(self.attributes)

    def canonical(self, attrs: Sequence[str]) -> tuple[str, ...]:
        """Returns the canonical ordering of a subset of attributes.

        Provides the order of the given attributes as they appear in this domain's
        `attributes` tuple.

        Args:
            attrs: A sequence of attribute names (Sequence[str]) present in the domain.

        Returns:
            A tuple of attribute names in their canonical order based on this domain.
        """
        # Note: Added Sequence[str] and tuple[str, ...] type hints for clarity.
        return tuple(a for a in self.attributes if a in attrs)

    def invert(self, attrs: Sequence[str]) -> list[str]:
        """Returns the attributes in this domain that are *not* in the given list.

        Args:
            attrs: A sequence of attribute names (Sequence[str]).

        Returns:
            A list of attribute names from this domain that are not present in `attrs`.
        """
        # Note: Added Sequence[str] and list[str] type hints for clarity.
        return [a for a in self.attributes if a not in attrs]

    def intersect(self, other: "Domain") -> "Domain":
        """Computes the intersection of this domain with another domain.

        Creates a new Domain containing only the attributes that are present in
        *both* this domain and the `other` domain. The shape for the common
        attributes is taken from this domain.

        Example Usage:
            >>> d1 = Domain(('a', 'b'), (10, 20))
            >>> d2 = Domain(('b', 'c'), (20, 30))
            >>> print(d1.intersect(d2))
        Domain(b: 20)

        Args:
            other: The Domain object to intersect with.

        Returns:
            A new Domain object representing the intersection.
        """
        return self.project([a for a in self.attributes if a in other.attributes])

    def axes(self, attrs: Sequence[str]) -> tuple[int, ...]:
        """Returns the numerical axes corresponding to the given attributes.

        Provides the integer indices of the specified attributes within this domain's
        `attributes` tuple. This is useful for indexing operations (e.g., in NumPy).

        Args:
            attrs: A sequence of attribute names (Sequence[str]) present in the domain.

        Returns:
            A tuple of integers representing the axes (indices) of the attributes.
        """
        return tuple(self.attributes.index(a) for a in attrs)

    def merge(self, other: "Domain") -> "Domain":
        """Merges this domain with another domain.

        Combines the attributes and shapes of both domains. If an attribute exists
        in both, the shape from this domain is used. Attributes unique to `other`
        are appended.

        Example:
            >>> d1 = Domain(('a', 'b'), (10, 20))
            >>> d2 = Domain(('b', 'c'), (20, 30))
            >>> print(d1.merge(d2))
        Domain(a: 10, b: 20, c: 30)

        Args:
            other: The Domain object to merge with this one.

        Returns:
            A new Domain object representing the combined domain.
        """
        extra = other.marginalize(self.attributes)
        return Domain(self.attributes + extra.attributes, self.shape + extra.shape)

    def size(self, attributes: Sequence[str] | None = None) -> int:
        """Calculates the total size (number of possible configurations) of the domain.

        If specific attributes are provided, calculates the size of the projected
        domain defined by those attributes. Otherwise, calculates the size of the
        entire domain by multiplying the shapes of all attributes.

        Example:
            >>> d1 = Domain(('a', 'b'), (10, 20))
            >>> d1.size()
        200
        >>> d1.size(['a'])
        10

        Args:
            attributes: An optional sequence of attribute names (Sequence[str]).
                If None, calculates the size of the full domain.

        Returns:
            The total size (int) of the specified domain subset or the full domain.
        """
        if attributes is None:
            return functools.reduce(lambda x, y: x * y, self.shape, 1)
        return self.project(attributes).size()

    @property
    def attrs(self):
        """Provides access to the tuple of attribute names.

        Returns:
            A tuple containing the names of the attributes in the domain.
        """
        return self.attributes

    def __contains__(self, name: str) -> bool:
        """Check if the given attribute is in the domain."""
        return name in self.attributes

    def __getitem__(self, a: str) -> int:
        """Allows accessing the shape (size) of an attribute using dictionary-like lookup.

        Args:
            a: The name of the attribute (str).

        Returns:
            The size (int) of the specified attribute.

        Raises:
            KeyError: If the attribute `a` is not found in the domain.
        """
        return self.config[a]

    def __iter__(self) -> Iterator[str]:
        """Provides an iterator over the attribute names of the domain.

        Allows iterating directly over the Domain object to get its attribute names.

        Yields:
            Attribute names (str) in the order they are defined.
        """
        return self.attributes.__iter__()

    def __len__(self) -> int:
        """Returns the number of attributes in the domain.

        Returns:
            The count (int) of attributes defined in the domain.
        """
        return len(self.attributes)

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the Domain object.

        Returns:
            A string in the format "Domain(attr1: shape1, attr2: shape2, ...)".
        """
        inner = ", ".join(["%s: %d" % x for x in zip(self.attributes, self.shape)])
        return "Domain(%s)" % inner
