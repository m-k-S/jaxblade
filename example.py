import jax.numpy as jnp

from surfaces import build_surface_interpolant, build_hub_coordinates, build_shroud_coordinates
from design_variables import DesignVariables

def run():
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
    return surface, hub, shroud

if __name__ == "__main__":
    import numpy as np
    import pyvista as pv

    ALPHA = 2.0
    OUTPUT_FILENAME = 'blade.vtk'

    surface, _, _= run()
    print('Converting pointcloud to 3D triangulated mesh...')
    cloud = pv.PolyData(np.asarray(surface.T))
    vol = cloud.delaunay_3d(alpha=ALPHA)
    shell = vol.extract_geometry()
    shell.save(OUTPUT_FILENAME)
    print(f'Successfully wrote mesh output to {OUTPUT_FILENAME}.')
    shell.plot()
