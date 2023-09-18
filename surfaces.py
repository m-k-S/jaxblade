from jax import jit
import jax.numpy as jnp
from interpolants import BilinearInterpolation, TransfiniteInterpolation, get_arc_length
from .parameterization.camber_thickness_params import Blade2DCamberThickness
from .parameterization.connecting_arc_params import Blade2DConnectingArcs

##############################################
# INTERPOLATING FUNCTIONS
##############################################

def make_meridional_channel(design_variables):

    """ Create the functions used to compute the (x,z) cooordinates at any point of the meridional channel

    The coordinates at the interior of the meridional channel are computed by transfinite interpolation from the
    coordinates at the boundaries (leading edge, lower surface, trailing edge and upper surface)

    The coordinates at any interior point can be evaluated as x(u,v) and z(u,v)
    """

    # Define a function to compute the x-coordinate of the meridional channel as x(u,v)
    meridional_channel_x = jit(TransfiniteInterpolation(design_variables.functions["x_leading"],
                                                        design_variables.functions["x_hub"],
                                                        design_variables.functions["x_trailing"],
                                                        design_variables.functions["x_shroud"],
                                                        design_variables.control_points["x_leading"][0],
                                                        design_variables.control_points["x_trailing"][0],
                                                        design_variables.control_points["x_trailing"][-1],
                                                        design_variables.control_points["x_leading"][-1]))

    # Define a function to compute the z-coordinate of the meridional channel as z(u,v)
    meridional_channel_z = jit(TransfiniteInterpolation(design_variables.functions["z_leading"],
                                                        design_variables.functions["z_hub"],
                                                        design_variables.functions["z_trailing"],
                                                        design_variables.functions["z_shroud"],
                                                        design_variables.control_points["z_leading"][0],
                                                        design_variables.control_points["z_trailing"][0],
                                                        design_variables.control_points["z_trailing"][-1],
                                                        design_variables.control_points["z_leading"][-1]))

    return meridional_channel_x, meridional_channel_z

def section_coordinates(u_section, v_section, design_variables, cascade_type="ANNULAR"):

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

    section_variables = {
        k: design_variables.functions[k](v_section)
        for k in design_variables.names
    }

    # Compute the coordinates of a blade section with an unitary meridional chord
    if design_variables.parameterization == 'CAMBER_THICKNESS':
        section_coordinates = Blade2DCamberThickness(section_variables).get_section_coordinates(u_section)
    elif design_variables.parameterization == 'CONNECTING_ARCS':
        section_coordinates = Blade2DConnectingArcs(section_variables).get_section_coordinates(u_section)
    else:
        raise Exception('Choose a valid option for PARAMETRIZATION_TYPE: "CONNECTING_ARCS" or "CAMBER_THICKNESS"')

    # Rename the section coordinates
    x = section_coordinates[0, :]                   # x corresponds to the meridional direction
    y = section_coordinates[1, :]                   # y corresponds to the tangential direction

    # Ensure that the x-coordinates are between zero and one (actually between zero+eps and one-eps)
    uu_section = (x - jnp.amin(x) + 1e-12)/(jnp.amax(x) - jnp.amin(x) + 2e-12)

    meridional_channel_x, meridional_channel_z = make_meridional_channel(design_variables)

    # Obtain the x-z coordinates by transfinite interpolation of the meridional channel contour
    x = meridional_channel_x(uu_section, v_section)       # x corresponds to the axial direction
    z = meridional_channel_z(uu_section, v_section)       # x corresponds to the radial direction

    # Create a single-variable function with the coordinates of the meridional channel
    x_func = lambda u: meridional_channel_x(u, v_section)
    z_func = lambda u: meridional_channel_z(u, v_section)
    m_func = lambda u: jnp.concatenate((x_func(u), z_func(u)), axis=0)

    # Compute the arc length of the meridional channel (streamline)
    arc_length = get_arc_length(m_func, 0.0 + 1e-6, 1.0 - 1e-6)

    # Compute the y-coordinates of the current blade section by scaling the unitary blade
    y = design_variables.functions["y_leading"](v_section) + y * arc_length

    # Transform the blade coordinates to cartesian coordinates
    if cascade_type == "ANNULAR":
        X = x
        Y = z*jnp.sin(y/z)
        Z = z*jnp.cos(y/z)
    elif cascade_type == "LINEAR":
        X = x
        Y = y
        Z = z
    else:
        raise Exception('Choose a valid cascade type: "LINEAR" or "ANNULAR"')

    # Piece together the X-Y-Z coordinates
    section_coordinates = jnp.concatenate((X, Y, Z), axis=0)

    return section_coordinates


##############################################
# MODELING SURFACES
#
# There are three main surfaces to model:
# - "Surface coordinates"
# - Hub coordinates
# - Shroud coordinates
##############################################


def build_surface_interpolant(u, v, num_points_section=500, n_sections=10):
    # Compute the coordinates of several blade sections
    # The (u,v) parametrization used to compute the blade sections must be regular (necessary for interpolation)
    num_points_section = 500
    u_interp = jnp.linspace(0.00, 1, num_points_section)
    v_interp = jnp.linspace(0.00, 1, n_sections)
    S_interp = jnp.zeros((3, num_points_section, n_sections), dtype=complex)
    for k in range(n_sections):
        S_interp[..., k] = section_coordinates(u_interp, v_interp[k])

    # Create the interpolator objects for the (x,y,z) coordinates
    x_function = BilinearInterpolation(u_interp, v_interp, S_interp[0, ...])
    y_function = BilinearInterpolation(u_interp, v_interp, S_interp[1, ...])
    z_function = BilinearInterpolation(u_interp, v_interp, S_interp[2, ...])

    return jnp.array([x_function(u, v), y_function(u, v), z_function(u, v)])

def build_hub_coordinates(u, design_variables):
    """ 
    Compute the coordinates of the hub surface in the (x,z) plane

    The hub surface is is extended to the inlet and outlet regions by linear extrapolation (G1 continuity)
    The hub surface is extended one-fourth of the meridional channel arc length at midspan
    """

    # Get the design variables
    DV_functions = design_variables.functions
    DV_control_points = design_variables.control_points

    # Sorting trick to retrieve order after concatenation
    my_order1 = jnp.argsort(u)
    my_order2 = jnp.argsort(my_order1)

    # Define the parameter for each arc (split the vector u in 3 pieces)
    # This is reordering, and we cannot reorder during matching. We retrieve the original order later
    u_inlet = jnp.sort((u[(u >= 0.00) & (u < 0.25)] - 0.00) / (0.25 - 0.00))
    u_main  = jnp.sort((u[(u >= 0.25) & (u < 0.75)] - 0.25) / (0.75 - 0.25))
    u_exit  = jnp.sort((u[(u >= 0.75) & (u <= 1.00)] - 0.75) / (1.00 - 0.75))

    meridional_channel_x, meridional_channel_z = make_meridional_channel(design_variables)

    # Get blade arc length at the mean section
    x_func = lambda uu: meridional_channel_x(uu, 0.50)
    z_func = lambda uu: meridional_channel_z(uu, 0.50)
    m_func = lambda uu: jnp.concatenate((x_func(uu), z_func(uu)), axis=0)
    arc_length = get_arc_length(m_func, 0.0 + 1e-6, 1.0 - 1e-6)

    # Tangent line extending the hub surface to the inlet region
    step = 1e-12
    dxdu = jnp.imag(DV_functions["x_hub"](0 + step * 1j)[0, 0]) / step
    dzdu = jnp.imag(DV_functions["z_hub"](0 + step * 1j)[0, 0]) / step
    slope_inlet = jnp.arctan2(dzdu,dxdu)
    x_inlet = DV_control_points["x_hub"][0] + (u_inlet - 1) * jnp.cos(slope_inlet) * arc_length / 4
    z_inlet = DV_control_points["z_hub"][0] + (u_inlet - 1) * jnp.sin(slope_inlet) * arc_length / 4

    # Tangent line extending the hub surface to the outlet region
    dxdu = -jnp.imag(DV_functions["x_hub"](1 - step * 1j)[0, 0]) / step
    dzdu = -jnp.imag(DV_functions["z_hub"](1 - step * 1j)[0, 0]) / step
    slope_exit = jnp.arctan2(dzdu,dxdu)
    x_exit = DV_control_points["x_hub"][-1] + u_exit * jnp.cos(slope_exit) * arc_length / 4
    z_exit = DV_control_points["z_hub"][-1] + u_exit * jnp.sin(slope_exit) * arc_length / 4

    # Region of the hub surface occupied by the blades
    x_main = DV_functions["x_hub"](u_main).flatten()
    z_main = DV_functions["z_hub"](u_main).flatten()

    # Concatenate the arcs to obtain the extended hub surface (and retrieve the original order)
    x_hub = jnp.concatenate((x_inlet, x_main, x_exit), axis=0)[my_order2]
    z_hub = jnp.concatenate((z_inlet, z_main, z_exit), axis=0)[my_order2]
    hub_coordinates = jnp.asarray((x_hub, z_hub))
    
    return hub_coordinates


#        self.u_hub = np.linspace(0, 1, 100)
#        self.u_shroud = np.linspace(0, 1, 100)
def get_extended_shroud_coordinates(u, design_variables):

    """ Compute the coordinates of the hub surface in the (x,z) plane

    The shroud surface is is extended to the inlet and outlet regions by linear extrapolation (G1 continuity)
    The shroud surface is extended one-fourth of the meridional channel arc length at midspan

    """

    # Get the design variables
    DV_functions = design_variables.functions
    DV_control_points = design_variables.control_points

    # Sorting trick to retrieve order after concatenation
    my_order1 = jnp.argsort(u)
    my_order2 = jnp.argsort(my_order1)

    # Define the parameter for each arc (split the vector u in 3 pieces)
    # This is reordering, and we cannot reorder during matching. We retrieve the original order later
    u_inlet = jnp.sort((u[(u >= 0.00) & (u < 0.25)] - 0.00) / (0.25 - 0.00))
    u_main = jnp.sort((u[(u >= 0.25) & (u < 0.75)] - 0.25) / (0.75 - 0.25))
    u_exit = jnp.sort((u[(u >= 0.75) & (u <= 1.00)] - 0.75) / (1.00 - 0.75))

    meridional_channel_x, meridional_channel_z = make_meridional_channel(design_variables)

    # Get blade arc length at the mean section
    x_func = lambda uu: meridional_channel_x(uu, 0.50)
    z_func = lambda uu: meridional_channel_z(uu, 0.50)
    m_func = lambda uu: jnp.concatenate((x_func(uu), z_func(uu)), axis=0)
    arc_length = get_arc_length(m_func, 0.0 + 1e-6, 1.0 - 1e-6)

    # Tangent line extending the hub surface to the inlet region
    step = 1e-12
    dxdu = jnp.imag(DV_functions["x_shroud"](0 + step * 1j)[0, 0]) / step
    dzdu = jnp.imag(DV_functions["z_shroud"](0 + step * 1j)[0, 0]) / step
    slope_inlet = jnp.arctan2(dzdu, dxdu)
    x_inlet = DV_control_points["x_shroud"][0] + (u_inlet - 1) * jnp.cos(slope_inlet) * arc_length / 4
    z_inlet = DV_control_points["z_shroud"][0] + (u_inlet - 1) * jnp.sin(slope_inlet) * arc_length / 4

    # Tangent line extending the hub surface to the outlet region
    dxdu = -jnp.imag(DV_functions["x_shroud"](1 - step * 1j)[0, 0]) / step
    dzdu = -jnp.imag(DV_functions["z_shroud"](1 - step * 1j)[0, 0]) / step
    slope_exit = jnp.arctan2(dzdu, dxdu)
    x_exit = DV_control_points["x_shroud"][-1] + u_exit * jnp.cos(slope_exit) * arc_length / 4
    z_exit = DV_control_points["z_shroud"][-1] + u_exit * jnp.sin(slope_exit) * arc_length / 4

    # Region of the hub surface occupied by the blades
    x_main = DV_functions["x_shroud"](u_main).flatten()
    z_main = DV_functions["z_shroud"](u_main).flatten()

    # Concatenate the arcs to obtain the extended hub surface (and retrieve the original order)
    x_hub = jnp.concatenate((x_inlet, x_main, x_exit), axis=0)[my_order2]
    z_hub = jnp.concatenate((z_inlet, z_main, z_exit), axis=0)[my_order2]
    shroud_coordinates = jnp.asarray((x_hub, z_hub))

    return shroud_coordinates