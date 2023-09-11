import jax.numpy as jnp
from jax import jit, vmap

class TransfiniteInterpolation:
    """
    Create a transfinite interpolator for a surface with a boundary described by four parametric curves.

    Parameters
    ----------
    C1_func : callable
        A function that takes a scalar or an array of scalars `v` and returns an array of shape (ndim, Nv)
        containing the coordinates of the west boundary.

    C2_func : callable
        A function that takes a scalar or an array of scalars `u` and returns an array of shape (ndim, Nu)
        containing the coordinates of the south boundary.

    C3_func : callable
        A function that takes a scalar or an array of scalars `v` and returns an array of shape (ndim, Nv)
        containing the coordinates of the east boundary.

    C4_func : callable
        A function that takes a scalar or an array of scalars `u` and returns an array of shape (ndim, Nu)
        containing the coordinates of the north boundary.

    P12 : ndarray
        A 1D array containing the coordinates of the point connecting the west and south boundaries.

    P23 : ndarray
        A 1D array containing the coordinates of the point connecting the south and east boundaries.

    P34 : ndarray
        A 1D array containing the coordinates of the point connecting the east and north boundaries.

    P41 : ndarray
        A 1D array containing the coordinates of the point connecting the north and west boundaries.

    Usage
    ----------
    # interpolator = jit_transfinite_interpolation(C1_func, C2_func, C3_func, C4_func, P12, P23, P34, P41)
    # result = interpolator(u, v)
    Returns: ndarray
        An array of shape (ndim, N) containing the coordinates of the surface at the points (u, v).
    """



    def __init__(self, C1_func, C2_func, C3_func, C4_func, P12, P23, P34, P41):
        self.C1_func = C1_func
        self.C2_func = C2_func
        self.C3_func = C3_func
        self.C4_func = C4_func
        self.P12 = jnp.array(P12)
        self.P23 = jnp.array(P23)
        self.P34 = jnp.array(P34)
        self.P41 = jnp.array(P41)

    def __call__(self, u, v):
        u, v = self._reshape_inputs(u, v)
        C1, C2, C3, C4 = vmap(self._vectorized_funcs)(u, v)

        term_1a = (1 - u) * C1 + u * C3
        term_1b = (1 - v) * C2 + v * C4
        term_2 = (1 - u) * (1 - v) * self.P12 + u * v * self.P34 + (1 - u) * v * self.P41 + u * (1 - v) * self.P23

        S = term_1a + term_1b - term_2

        return S

    def _reshape_inputs(self, u, v):
        if jnp.isscalar(u) and jnp.isscalar(v):
            u, v = jnp.array([u]), jnp.array([v])
        elif jnp.isscalar(u) and (v.ndim == 1):
            u = jnp.array([u])
        elif jnp.isscalar(v) and (u.ndim == 1):
            v = jnp.array([v])
        elif (u.ndim == 1) and (v.ndim == 1):
            assert u.shape[0] == v.shape[0], 'u and v must have the same size when they are one-dimensional arrays'
        else:
            raise Exception('The format of the u or v input is not supported')
        return u, v

    def _vectorized_funcs(self, u, v):
        return vmap(self.C1_func)(v), vmap(self.C2_func)(u), vmap(self.C3_func)(v), vmap(self.C4_func)(u)

# To enable GPU acceleration, you can use jit to compile the function:
jit_transfinite_interpolation = jit(TransfiniteInterpolation)


