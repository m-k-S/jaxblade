import jax.numpy as jnp

from surfaces import build_surface_interpolant, build_hub_coordinates, build_shroud_coordinates
from design_variables import DesignVariables

if __name__ == "__main__":
    from example_params.example_connecting_arcs import config_params

    cascade_type = config_params['CASCADE_TYPE']
    n_sections = config_params['N_SECTIONS']
    design_variables = DesignVariables(config_params)

    h = 1e-5

    Nu = 500
    Nv = 25

    u = jnp.linspace(0.00+h, 1.00-h, Nu)
    v = jnp.linspace(0.00+h, 1.00-h, Nv)

    [u, v] = jnp.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    N_points = Nu*Nv

    u_surfaces = jnp.linspace(0, 1, 100)

    surface = build_surface_interpolant(
        u, v, 
        design_variables, 
        n_sections=n_sections,
        cascade_type=cascade_type,
    )
    hub = build_hub_coordinates(u_surfaces, design_variables)
    shroud = build_shroud_coordinates(u_surfaces, design_variables)
