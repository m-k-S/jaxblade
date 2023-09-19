import jax.numpy as jnp
from jax import jit, vmap
from scipy.integrate import fixed_quad

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

        # Declare input variables as instance variables
        self.C1_func = C1_func
        self.C2_func = C2_func
        self.C3_func = C3_func
        self.C4_func = C4_func
        self.P12 = P12
        self.P23 = P23
        self.P34 = P34
        self.P41 = P41


    def __call__(self, u, v):

        """ Evaluate the transfinite interpolator for input parameters ´(u,v)´ and return the surface coordinates ´S´

        Parameters
        ----------

        u: scalar or ndarray with shape (N,)
        Parameter in the west-east direction

        v: scalar or ndarray with shape (N,)
        Parameter in the south-north direction

        Returns
        -------
        S : ndarray with shape (ndim, N)
            Array containing the coordinates computed by transfinite interpolation | S(u,v)

        Notes
        -----
        The following formats are allowed for the input ´u´ and ´v´:

            - Both ´u´ and ´v´ are scalars
            - Both ´u´ and ´v´ are one-dimensional arrays with shape (N,)
            - ´u´ is a scalar and ´v´ is a one-dimensional array with shape (N,)
            - ´v´ is a scalar and ´u´ is a one-dimensional array with shape (N,)

        This function does not support ´u´ or ´v´ as two-dimensional arrays
        Use jnp.flatten() before calling this function if the (u,v) parametrization is in meshgrid format

        """

        # Rename instance variables
        C1_func = self.C1_func
        C2_func = self.C2_func
        C3_func = self.C3_func
        C4_func = self.C4_func
        P12 = self.P12
        P23 = self.P23
        P34 = self.P34
        P41 = self.P41

        # Adjust the data format depending on the shape of the (u,v) input
        if jnp.ndim(u) == 0 and jnp.ndim(v) == 0:
            N = 1
            u = jnp.asarray([u])
            v = jnp.asarray([v])

        elif jnp.ndim(u) == 0 and (v.ndim == 1):
            N = jnp.shape(v)[0]
            u = jnp.asarray([u])
            u = jnp.repeat(u, repeats=N, axis=0)

        elif jnp.ndim(v) == 0 and (u.ndim == 1):
            N = jnp.shape(u)[0]
            v = jnp.asarray([v])
            v = jnp.repeat(v, repeats=N, axis=0)

        elif (u.ndim == 1) and (v.ndim == 1):
            N = jnp.shape(u)[0]
            Nv = jnp.shape(v)[0]
            assert (N == Nv), 'u and v must have the same size when they are one-dimensional arrays'

        elif (u.ndim > 1) or (v.ndim > 1):
            raise Exception('u or v arrays with more than one dimension are not supported')

        else:
            raise Exception('Te format of the u or v input is not supported')

        # Evaluate the functions that define the boundaries (before reshaping u and v!)
        C1 = C1_func(v)
        C2 = C2_func(u)
        C3 = C3_func(v)
        C4 = C4_func(u)

        # Number of coordinate dimensions of the problem
        if jnp.isscalar(P12):
            P12 = jnp.asarray([P12])
            P23 = jnp.asarray([P23])
            P34 = jnp.asarray([P34])
            P41 = jnp.asarray([P41])

        n_dim = jnp.shape(P12)[0]

        # Reshape variables so that they are conformable for matrix multiplication
        u = jnp.repeat(u[jnp.newaxis, :], repeats=n_dim, axis=0)
        v = jnp.repeat(v[jnp.newaxis, :], repeats=n_dim, axis=0)
        P12 = jnp.repeat(P12[:, jnp.newaxis], repeats=N, axis=1)
        P23 = jnp.repeat(P23[:, jnp.newaxis], repeats=N, axis=1)
        P34 = jnp.repeat(P34[:, jnp.newaxis], repeats=N, axis=1)
        P41 = jnp.repeat(P41[:, jnp.newaxis], repeats=N, axis=1)

        # Linear interpolation in v between the C1(v) and C3(v) curves
        term_1a = (1 - u) * C1 + u * C3

        # Linear interpolation in u between the C2(v) and C4(v) curves
        term_1b = (1 - v) * C2 + v * C4

        # Bilinear interpolation between the four corners of the domain (P12, P23, P34, P41)
        term_2 = (1 - u) * (1 - v) * P12 + u * v * P34 + (1 - u) * v * P41 + u * (1 - v) * P23

        # Transfinite interpolation formula (also known as bilinearly blended Coons patch)
        # Note that the order of (u,v) is inverted with respect to the original formula
        S = term_1a + term_1b - term_2

        return S
class BilinearInterpolation:

    """ Create a bilinear interpolator for a two-dimensional function ´f´ on a regular ´(x,y)´ grid

    Parameters
    ----------
    x : array_like with shape (Nx,)
        Evenly spaced array containing the x-coordinates on the original grid

    y : array_like with shape (Ny,)
        Evenly spaced array containing the y-coordinates on the original grid

    f : array_line with shape (Nx, Ny)
        Array containing the function values on the original grid

    Example of use
    --------------
    When this class is instantiated it creates an interpolator using the original grid coordinates and function values
    This interpolator can be called to interpolate the function values at the input query points

        # Import statements
        import numpy as np
        from bilinear_interpolation import BilinearInterpolation

        # Compute the function values on the original grid
        a, b = 0.00, 1.00
        Nx, Ny = 101, 101
        x = jnp.linspace(a, b, Nx)
        y = jnp.linspace(a, b, Ny)
        [X, Y] = jnp.meshgrid(x, y)
        f = jnp.log(1 + X ** 2 + Y ** 2)

        # Define a new grid for interpolation
        a, b = 0.25, 0.75
        nx, ny = 51, 51
        xq = jnp.linspace(a, b, nx)
        yq = jnp.linspace(a, b, ny)

        # Interpolate the function values on the new grid
        f_interpolator = BilinearInterpolation(x, y, f)
        fq = f_interpolator(xq, yq)

    References
    ----------
    Numerical recipes. Section 3.6 - Interpolation on a Grid in Multidimensions
    W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery

    """


    def __init__(self, x, y, f):

        # Declare input variables as instance variables
        self.x = x
        self.y = y
        self.f = f


    def __call__ (self, xq, yq):

        """ Evaluate the bilinear interpolator at query points ´(xq,yq)´ and return the function values ´fq´

        Parameters
        ----------
        xq : array_like with shape (N,)
            Array containing x-coordinates at the query points

        yq : array_like with shape (N,)
            Array containing y-coordinates at the query points

        Returns
        -------
        fq : array_like with shape (N,)
            Array containing the function values at the query points

        """

        # Rename instance variables
        x = self.x
        y = self.y
        f = self.f

        # print(jnp.any(xq < jnp.amin(x)))
        # pdb.set_trace()

        # # Check for extrapolation
        # if jnp.any(xq < jnp.amin(x)) or jnp.any(xq > jnp.amax(x)) or \
        #    jnp.any(yq < jnp.amin(y)) or jnp.any(yq > jnp.amax(y)):
        #     raise ValueError('Extrapolation is not supported')

        # Check the input data
        Nx, Ny = x.size, y.size
        if f.shape != (Nx, Ny):
            raise ValueError('f is not set properly. f should have shape (Nx, Ny)')

        if (xq.ndim > 1) or (yq.ndim > 1):
            raise Exception('xq and yq must be one dimensional arrays')

        elif xq.size != yq.size:
            raise Exception('xq and yq must have the same number of elements')

        # Compute the indexes of query points neighbours (i and j have shape (N,))
        # This can be regarded as an explicit search algorithm for the case of a regular grid
        # This section of the code would be to be replaced by a search algorithm for the case of a non-regular grid
        i = jnp.floor(jnp.real((xq[:] - x[0]) / (x[-1] - x[0]) * (Nx - 1)))
        j = jnp.floor(jnp.real((yq[:] - y[0]) / (y[-1] - y[0]) * (Ny - 1)))
        i = jnp.asarray(i, dtype='int')
        j = jnp.asarray(j, dtype='int')
        # Using jnp.real() to find the indexes (i,j) is a trick required to avoid jnp.floor() of a complex number
        # This allows to find the "equivalent" index of a complex query point with a small imaginary part (complex step)

        # Avoid index out of bounds error when providing the upper limit of the interpolation
        i = i.at[i == Nx - 1].set(Nx - 2)
        j = j.at[j == Ny - 1].set(Ny - 2)

        # Bilinear interpolation formula
        u = (xq - x[i]) / (x[i + 1] - x[i])
        v = (yq - y[j]) / (y[j + 1] - y[j])
        fq = (1 - u) * (1 - v) * f[i, j] + u * (1 - v) * f[i + 1, j] + u * v * f[i + 1, j + 1] + (1 - u) * v * f[i, j + 1]

        # # Equivalent, but slower, alternative formula
        # Q1 = (x[i + 1] - xq) / (x[i + 1] - x[i]) * f[i, j] + (yq - x[i]) / (x[i + 1] - x[i]) * f[i + 1, j]
        # Q2 = (x[i + 1] - xq) / (x[i + 1] - x[i]) * f[i, j + 1] + (yq - x[i]) / (x[i + 1] - x[i]) *f[i + 1, j + 1]
        # fq = (y[j + 1] - xq) / (y[j + 1] - y[j]) * Q1 + (yq - y[j]) / (y[j + 1] - y[j]) * Q2

        return fq

def get_arc_length(C_func, t1, t2, n=100):

    """ Compute the arc length of a parametric curve ´C(t) = (x_0(t),..., x_n(t))´ using numerical integration

    Parameters
    ----------
    C_func : function returning ndarray with shape (ndim, N)
        Handle to a function that returns the parametric curve coordinates

    t1 : scalar
        Lower limit of integration for the arclength computation

    t2 : scalar
        Upper limit of integration for the arclength computation

    Returns
    -------
    L : scalar
        Arc length of curve C(t) in the interval [t1, t2]

    """

    # Compute the arc length differential using the central finite differences
    # Be careful with the step-size selected, accurary is not critical, but rounding error must not bloe up
    # It is not possible to use the complex step is the result of the arc-length computation is further differentiated
    def get_arc_length_differential(t, step=1e-3):
        # dCdt = jnp.imag(C_func(t + 1j * step)) / step              # dC/dt = (dx_0/dt, ..., dx_n/dt)
        dCdt = (C_func(t + step) - C_func(t - step))/(2*step)       # dC/dt = (dx_0/dt, ..., dx_n/dt)
        dLdt = jnp.sqrt(jnp.sum(dCdt**2, axis=0))                     # dL/dt = [(dx_0/dt)^2 + ... + (dx_n/dt)^2]^(1/2)
        return dLdt

    # Compute the arc length of C(t) in the interval [t1, t2] by numerical integration
    L = fixed_quad(get_arc_length_differential, t1, t2)[0]
    # x = jnp.linspace(t1, t2, 100)
    # y = get_arc_length_differential(x)
    # L = trapezoid(y, x)[0]

    return L