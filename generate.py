import jax.numpy as jnp
from design_variables import DesignVariables
from surfaces import build_surface_interpolant, build_hub_coordinates, build_shroud_coordinates


N_BLADES               = 1
NDIM                   = 3
N_SECTIONS             = 50
CASCADE_TYPE           = "ANNULAR"

config_params = {
    "x_leading": [0.00, 0.20, 0.10],
    "y_leading": [0.00, 0.05, 0.00],
    "z_leading": [2.00, 2.80, 3.50],
    "x_trailing": [1.00, 1.10, 0.90],
    "z_trailing": [2.20, 3.00, 3.70],
    "x_hub": [0.25, 0.75],
    "z_hub": [2.00, 2.20],
    "x_shroud": [0.30, 0.70],
    "z_shroud": [3.50, 3.70],
    "parameterization_type": "CONNECTING_ARCS",
    "stagger": [-48, -50, -48],
    "theta_in": [0.00, 5.00, 0.00],
    "theta_out": [-65, -70, -70],
    "wedge_in": [25, 20, 20],
    "wedge_out": [5, 5, 5],
    "radius_in": [0.15, 0.12, 0.10],
    "radius_out": [0.02, 0.02, 0.01],
    "dist_1": [0.35],
    "dist_2": [0.30],
    "dist_3": [0.30],
    "dist_4": [0.30],
}

design_variables = DesignVariables(
    config_params
)

h = 1e-5

Nu = 500
Nv = 25

# Define a default (u,v) parametrization from a meshgrid
if NDIM == 2:
    Nu, Nv = Nu, 1
    u = jnp.linspace(0.00+h, 1.00-h, Nu)
    v = 0.50
elif NDIM == 3:
    Nu, Nv = Nu, Nv
    u = jnp.linspace(0.00+h, 1.00-h, Nu)
    v = jnp.linspace(0.00+h, 1.00-h, Nv)
else:
    raise Exception('The number of dimensions must be "2" or "3"')

[u, v] = jnp.meshgrid(u, v)
u = u.flatten()
v = v.flatten()
N_points = Nu*Nv

u_surfaces = jnp.linspace(0, 1, 100)

build_surface_interpolant(u, v, design_variables)
build_hub_coordinates(u_surfaces, design_variables)
build_shroud_coordinates(u_surfaces, design_variables)