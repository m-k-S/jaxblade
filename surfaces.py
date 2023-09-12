from jax import jit
import jax.numpy as jnp
from interpolants import TransfiniteInterpolation, get_arc_length
from .parameterization.camber_thickness_params import Blade2DCamberThickness
from .parameterization.connecting_arc_params import Blade2DConnectingArcs

def make_meridional_channel(design_variable_functions, design_variable_control_points):

    """ Create the functions used to compute the (x,z) cooordinates at any point of the meridional channel

    The coordinates at the interior of the meridional channel are computed by transfinite interpolation from the
    coordinates at the boundaries (leading edge, lower surface, trailing edge and upper surface)

    The coordinates at any interior point can be evaluated as x(u,v) and z(u,v)
    """

    # Define a function to compute the x-coordinate of the meridional channel as x(u,v)
    get_meridional_channel_x = jit(TransfiniteInterpolation(design_variable_functions["x_leading"],
                                                        design_variable_functions["x_hub"],
                                                        design_variable_functions["x_trailing"],
                                                        design_variable_functions["x_shroud"],
                                                        design_variable_control_points["x_leading"][0],
                                                        design_variable_control_points["x_trailing"][0],
                                                        design_variable_control_points["x_trailing"][-1],
                                                        design_variable_control_points["x_leading"][-1]))

    # Define a function to compute the z-coordinate of the meridional channel as z(u,v)
    get_meridional_channel_z = jit(TransfiniteInterpolation(design_variable_functions["z_leading"],
                                                        design_variable_functions["z_hub"],
                                                        design_variable_functions["z_trailing"],
                                                        design_variable_functions["z_shroud"],
                                                        design_variable_control_points["z_leading"][0],
                                                        design_variable_control_points["z_trailing"][0],
                                                        design_variable_control_points["z_trailing"][-1],
                                                        design_variable_control_points["z_leading"][-1]))

    return get_meridional_channel_x, get_meridional_channel_z

def get_section_coordinates(u_section, v_section, parameterization="CONNECTING_ARCS"):

    """ Compute the coordinates of the current blade section

    Parameters
    ----------
    u_section : ndarray with shape (Nu,)
        Array containing the u-parameter used to evaluate section coordinates

    v_section : scalar
        Scalar containing the v-parameter of the current blade span

    Returns
    -------
    section_coordinates : ndarray with shape (3, Nu)
        Array containing the (x,y,z) coordinates of the current blade section
    """

    # Get the design variables of the current blade section
    section_variables = {}
    for k in DVs_names_2D:
        section_variables[k] = DVs_functions[k](v_section)

    # Compute the coordinates of a blade section with an unitary meridional chord
    if parameterization == 'CONNECTING_ARCS':
        section_coordinates = Blade2DConnectingArcs(section_variables).get_section_coordinates(u_section)
    elif parameterization == 'CAMBER_THICKNESS':
        section_coordinates = Blade2DCamberThickness(section_variables).get_section_coordinates(u_section)
    else:
        raise Exception('Choose a valid option for PARAMETRIZATION_TYPE: "CONNECTING_ARCS" or "CAMBER_THICKNESS"')

    # Rename the section coordinates
    x = section_coordinates[0, :]                   # x corresponds to the meridional direction
    y = section_coordinates[1, :]                   # y corresponds to the tangential direction

    # Ensure that the x-coordinates are between zero and one (actually between zero+eps and one-eps)
    uu_section = (x - jnp.amin(x) + 1e-12)/(jnp.amax(x) - jnp.amin(x) + 2e-12)

    # Obtain the x-z coordinates by transfinite interpolation of the meridional channel contour
    x = get_meridional_channel_x(uu_section, v_section)       # x corresponds to the axial direction
    z = get_meridional_channel_z(uu_section, v_section)       # x corresponds to the radial direction

    # Create a single-variable function with the coordinates of the meridional channel
    x_func = lambda u: get_meridional_channel_x(u, v_section)
    z_func = lambda u: get_meridional_channel_z(u, v_section)
    m_func = lambda u: jnp.concatenate((x_func(u), z_func(u)), axis=0)

    # Compute the arc length of the meridional channel (streamline)
    arc_length = get_arc_length(m_func, 0.0 + 1e-6, 1.0 - 1e-6)

    # Compute the y-coordinates of the current blade section by scaling the unitary blade
    y = DVs_functions["y_leading"](v_section) + y * arc_length

    # Transform the blade coordinates to cartesian coordinates
    if CASCADE_TYPE == "LINEAR":
        X = x
        Y = y
        Z = z
    elif CASCADE_TYPE == "ANNULAR":
        X = x
        Y = z*jnp.sin(y/z)
        Z = z*jnp.cos(y/z)
    else:
        raise Exception('Choose a valid cascade type: "LINEAR" or "ANNULAR"')

    # Piece together the X-Y-Z coordinates
    section_coordinates = jnp.concatenate((X, Y, Z), axis=0)

    return section_coordinates

def make_surface_interpolant(self, interp_method='bilinear'):

    """ Create a surface interpolant using the coordinates of several blade sections

    Set interp_method='bilinear' to create a bilinear interpolator
    Set interp_method='bicubic' to create a bicubic interpolator

    The interpolant created by this method is used by get_surface_coordinates() to evaluate the blade surface
    coordinates for any input (u,v) parametrization

    """

    # Update the geometry before creating the interpolant
    get_DVs_functions()            # First update the functions that return design variables
    make_meridional_channel()      # Then update the functions that return the meridional channel coordinates

    # Compute the coordinates of several blade sections
    # The (u,v) parametrization used to compute the blade sections must be regular (necessary for interpolation)
    num_points_section = 500
    u_interp = jnp.linspace(0.00, 1, num_points_section)
    v_interp = jnp.linspace(0.00, 1, N_SECTIONS)
    S_interp = jnp.zeros((3, num_points_section, N_SECTIONS), dtype=complex)
    for k in range(N_SECTIONS):
        S_interp[..., k] = get_section_coordinates(u_interp, v_interp[k])

    # Create the interpolator objects for the (x,y,z) coordinates
    if interp_method == 'bilinear':
        x_function = BilinearInterpolation(u_interp, v_interp, S_interp[0, ...])
        y_function = BilinearInterpolation(u_interp, v_interp, S_interp[1, ...])
        z_function = BilinearInterpolation(u_interp, v_interp, S_interp[2, ...])

    elif interp_method == 'bicubic':
        # There seems to be a problem with bicubic interpolation, the output is wiggly
        x_function = BicubicInterpolation(u_interp, v_interp, S_interp[0, ...])
        y_function = BicubicInterpolation(u_interp, v_interp, S_interp[1, ...])
        z_function = BicubicInterpolation(u_interp, v_interp, S_interp[2, ...])

    else:
        raise Exception('Choose a valid interpolation method: "bilinear" or "bicubic"')

    # Create the surface interpolant using a lambda function to combine the (x,y,z) interpolants
    surface_interpolant = lambda u, v: jnp.asarray((x_function(u, v), y_function(u, v), z_function(u, v)))


def make_meridional_channel(self):

    """ Create the functions used to compute the (x,z) cooordinates at any point of the meridional channel

    The coordinates at the interior of the meridional channel are computed by transfinite interpolation from the
    coordinates at the boundaries (leading edge, lower surface, trailing edge and upper surface)

    The coordinates at any interior point can be evaluated as x(u,v) and z(u,v)

    The functions created by this method are used by get_section_coordinates() to map the coordinates of
    the 2D unitary blades into the meridional channel

    """

    # Define a function to compute the x-coordinate of the meridional channel as x(u,v)
    get_meridional_channel_x = TransfiniteInterpolation(DVs_functions["x_leading"],
                                                                DVs_functions["x_hub"],
                                                                DVs_functions["x_trailing"],
                                                                DVs_functions["x_shroud"],
                                                                DVs_control_points["x_leading"][0],
                                                                DVs_control_points["x_trailing"][0],
                                                                DVs_control_points["x_trailing"][-1],
                                                                DVs_control_points["x_leading"][-1])

    # Define a function to compute the z-coordinate of the meridional channel as z(u,v)
    get_meridional_channel_z = TransfiniteInterpolation(DVs_functions["z_leading"],
                                                                DVs_functions["z_hub"],
                                                                DVs_functions["z_trailing"],
                                                                DVs_functions["z_shroud"],
                                                                DVs_control_points["z_leading"][0],
                                                                DVs_control_points["z_trailing"][0],
                                                                DVs_control_points["z_trailing"][-1],
                                                                DVs_control_points["z_leading"][-1])

    # Compute the arc length of the blade meanline (secondary computation required for the BladeFit class)
    x_func = lambda u: get_meridional_channel_x(u, v=0.50)
    z_func = lambda u: get_meridional_channel_z(u, v=0.50)
    m_func = lambda u: jnp.concatenate((x_func(u), z_func(u)), axis=0)
    meanline_length = get_arc_length(m_func, 0.0 + 1e-6, 1.0 - 1e-6)




def make_hub_surface(self):
    u_hub = jnp.linspace(0, 1, 100)
    hub_coordinates = get_extended_hub_coordinates(u_hub)

def get_extended_hub_coordinates(self, u):

    """ Compute the coordinates of the hub surface in the (x,z) plane

    The hub surface is is extended to the inlet and outlet regions by linear extrapolation (G1 continuity)
    The hub surface is extended one-fourth of the meridional channel arc length at midspan

    """

    # Sorting trick to retrieve order after concatenation
    my_order1 = jnp.argsort(u)
    my_order2 = jnp.argsort(my_order1)

    # Define the parameter for each arc (split the vector u in 3 pieces)
    # This is reordering, and we cannot reorder during matching. We retrieve the original order later
    u_inlet = jnp.sort((u[(u >= 0.00) & (u < 0.25)] - 0.00) / (0.25 - 0.00))
    u_main  = jnp.sort((u[(u >= 0.25) & (u < 0.75)] - 0.25) / (0.75 - 0.25))
    u_exit  = jnp.sort((u[(u >= 0.75) & (u <= 1.00)] - 0.75) / (1.00 - 0.75))

    # Get blade arc length at the mean section
    x_func = lambda uu: get_meridional_channel_x(uu, 0.50)
    z_func = lambda uu: get_meridional_channel_z(uu, 0.50)
    m_func = lambda uu: jnp.concatenate((x_func(uu), z_func(uu)), axis=0)
    arc_length = get_arc_length(m_func, 0.0 + 1e-6, 1.0 - 1e-6)

    # Tangent line extending the hub surface to the inlet region
    step = 1e-12
    dxdu = jnp.imag(DVs_functions["x_hub"](0 + step * 1j)[0, 0]) / step
    dzdu = jnp.imag(DVs_functions["z_hub"](0 + step * 1j)[0, 0]) / step
    slope_inlet = jnp.arctan2(dzdu,dxdu)
    x_inlet = DVs_control_points["x_hub"][0] + (u_inlet - 1) * jnp.cos(slope_inlet) * arc_length / 4
    z_inlet = DVs_control_points["z_hub"][0] + (u_inlet - 1) * jnp.sin(slope_inlet) * arc_length / 4

    # Tangent line extending the hub surface to the outlet region
    dxdu = -jnp.imag(DVs_functions["x_hub"](1 - step * 1j)[0, 0]) / step
    dzdu = -jnp.imag(DVs_functions["z_hub"](1 - step * 1j)[0, 0]) / step
    slope_exit = jnp.arctan2(dzdu,dxdu)
    x_exit = DVs_control_points["x_hub"][-1] + u_exit * jnp.cos(slope_exit) * arc_length / 4
    z_exit = DVs_control_points["z_hub"][-1] + u_exit * jnp.sin(slope_exit) * arc_length / 4

    # Region of the hub surface occupied by the blades
    x_main = DVs_functions["x_hub"](u_main).flatten()
    z_main = DVs_functions["z_hub"](u_main).flatten()

    # Concatenate the arcs to obtain the extended hub surface (and retrieve the original order)
    x_hub = jnp.concatenate((x_inlet, x_main, x_exit), axis=0)[my_order2]
    z_hub = jnp.concatenate((z_inlet, z_main, z_exit), axis=0)[my_order2]
    hub_coordinates = jnp.asarray((x_hub, z_hub))

    return hub_coordinates


# ---------------------------------------------------------------------------------------------------------------- #
# Compute the coordinates of the shroud
# ---------------------------------------------------------------------------------------------------------------- #
def make_shroud_surface(self):
    u_shroud = jnp.linspace(0, 1, 100)
    shroud_coordinates = get_extended_shroud_coordinates(u_shroud)

def get_extended_shroud_coordinates(self, u):

    """ Compute the coordinates of the hub surface in the (x,z) plane

    The shroud surface is is extended to the inlet and outlet regions by linear extrapolation (G1 continuity)
    The shroud surface is extended one-fourth of the meridional channel arc length at midspan

    """

    # Sorting trick to retrieve order after concatenation
    my_order1 = jnp.argsort(u)
    my_order2 = jnp.argsort(my_order1)

    # Define the parameter for each arc (split the vector u in 3 pieces)
    # This is reordering, and we cannot reorder during matching. We retrieve the original order later
    u_inlet = jnp.sort((u[(u >= 0.00) & (u < 0.25)] - 0.00) / (0.25 - 0.00))
    u_main = jnp.sort((u[(u >= 0.25) & (u < 0.75)] - 0.25) / (0.75 - 0.25))
    u_exit = jnp.sort((u[(u >= 0.75) & (u <= 1.00)] - 0.75) / (1.00 - 0.75))

    # Get blade arc length at the mean section
    x_func = lambda uu: get_meridional_channel_x(uu, 0.50)
    z_func = lambda uu: get_meridional_channel_z(uu, 0.50)
    m_func = lambda uu: jnp.concatenate((x_func(uu), z_func(uu)), axis=0)
    arc_length = get_arc_length(m_func, 0.0 + 1e-6, 1.0 - 1e-6)

    # Tangent line extending the hub surface to the inlet region
    step = 1e-12
    dxdu = jnp.imag(DVs_functions["x_shroud"](0 + step * 1j)[0, 0]) / step
    dzdu = jnp.imag(DVs_functions["z_shroud"](0 + step * 1j)[0, 0]) / step
    slope_inlet = jnp.arctan2(dzdu, dxdu)
    x_inlet = DVs_control_points["x_shroud"][0] + (u_inlet - 1) * jnp.cos(slope_inlet) * arc_length / 4
    z_inlet = DVs_control_points["z_shroud"][0] + (u_inlet - 1) * jnp.sin(slope_inlet) * arc_length / 4

    # Tangent line extending the hub surface to the outlet region
    dxdu = -jnp.imag(DVs_functions["x_shroud"](1 - step * 1j)[0, 0]) / step
    dzdu = -jnp.imag(DVs_functions["z_shroud"](1 - step * 1j)[0, 0]) / step
    slope_exit = jnp.arctan2(dzdu, dxdu)
    x_exit = DVs_control_points["x_shroud"][-1] + u_exit * jnp.cos(slope_exit) * arc_length / 4
    z_exit = DVs_control_points["z_shroud"][-1] + u_exit * jnp.sin(slope_exit) * arc_length / 4

    # Region of the hub surface occupied by the blades
    x_main = DVs_functions["x_shroud"](u_main).flatten()
    z_main = DVs_functions["z_shroud"](u_main).flatten()

    # Concatenate the arcs to obtain the extended hub surface (and retrieve the original order)
    x_hub = jnp.concatenate((x_inlet, x_main, x_exit), axis=0)[my_order2]
    z_hub = jnp.concatenate((z_inlet, z_main, z_exit), axis=0)[my_order2]
    shroud_coordinates = jnp.asarray((x_hub, z_hub))

    return shroud_coordinates