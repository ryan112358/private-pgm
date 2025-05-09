"""Prototype implementation of scan-based einsum implementation."""

import jax
import jax.numpy as jnp


def _axis_name_to_dim(axis_names: str, axis_name: str) -> int | None:
  """Returns the index of axis_name in axis_names, or None if not found."""
  if axis_name in axis_names:
    return axis_names.index(axis_name)
  # else: (implicit)
  return None


def _get_subarrays(
    arrays: tuple[jax.Array, ...], ax_dims: list[int | None], index: int
) -> tuple[jax.Array, ...]:
  """Return a slice of each array along the given axis at the given index.

  Example:
    >>> A = jnp.arange(6).reshape(2, 3)
    >>> B = jnp.arange(8).reshape(2, 4)
    >>> C = jnp.arange(12).reshape(3, 4)
    >>> X, Y, Z = _get_subarrays([A, B, C], [1, None, 0], 2)
    >>> jnp.allclose(A[:,2], X) & jnp.allclose(B, Y) & jnp.allclose(C[2,:], Z)
    Array(True, dtype=bool)

  Args:
    arrays: A list of arrays.
    ax_dims: A list of axis dimensions to slice along, or None if no slicing is
      needed.
    index: The index to slice at.
  Returns:
    A list of arrays, each of which is a slice of the corresponding input array.
  """
  ans = []
  for array, dim in zip(arrays, ax_dims):
    if dim is None:
      ans.append(array)
    else:
      idx = tuple(slice(None) if i != dim else index for i in range(array.ndim))
      ans.append(array[idx])
  return tuple(ans)


def _infer_shapes(
    input_axes_names: list[str], arrays: tuple[jax.Array]
) -> dict[str, int]:
  """Infer the shapes of the arrays given the input axes names."""
  ans = {}
  for axis_names, arr in zip(input_axes_names, arrays):
    ans.update(**dict(zip(axis_names, arr.shape)))
  return ans


def scan_einsum(
    formula: str, *arrays: jax.Array, sequential: str = ''
) -> jax.Array:
  """Einsum implementation that allows sequential execution across any axes.

  Can be more time and memory efficient than jnp.einsum when there are more
  than 2 operands, and even run in settings where jnp.einsum would not.

  Args:
    formula: The einsum formula.
    *arrays: The arrays to einsum.
    sequential: A string of axes to sequentially einsum across.  
    When sequential is empty, jnp.einsum is used.  When sequential is a single
      axis name, the smaller einsums are executed sequentially along that axis,
      and the results are accumulated.  When sequential is a string with
      multiple axis names, sequential einsums are executed along all axis 
      names given.

  Returns:
    The einsum result.
  """
  if not sequential:
    return jnp.einsum(formula, *arrays)

  ax = sequential[0]
  if ax not in formula:
      raise ValueError(f"Sequential axis '{ax}' not found.")

  input_axes, output_axes = formula.split('->')
  input_axes = input_axes.split(',')
  shapes = _infer_shapes(input_axes, arrays)

  ax_dims = [_axis_name_to_dim(names, ax) for names in input_axes]

  # We compute smaller einsums sequentially by looping over the ax dimension.
  loop = jnp.arange(shapes[ax])
  new_formula = formula.replace(ax, '')

  def small_einsum(i):
    new_arrays = _get_subarrays(arrays, ax_dims, i)
    return scan_einsum(new_formula, *new_arrays, sequential=sequential[1:])

  if ax in output_axes:
    # Each smaller einsum is independent.
    return jax.lax.map(small_einsum, loop).swapaxes(0, output_axes.index(ax))
  # else: (implicit)
  # Each smaller einsum contributes to the global einsum.
  init = jnp.zeros(tuple(shapes[i] for i in output_axes))
  return jax.lax.scan(
      lambda carry, i: (carry + small_einsum(i), ()), init, loop
  )[0]
    
