import jax.numpy as jnp
from .parameterization.bspline import BSplineCurve

# Array of variable names for the meridional channel
MERIDIONAL_CHANNEL_DESIGN_VARIABLES =  ['x_leading',
                                        'y_leading',
                                        'z_leading',
                                        'x_trailing',
                                        'z_trailing',
                                        'x_hub',
                                        'z_hub',
                                        'x_shroud',
                                        'z_shroud']

# Array of variable names for blade sections (camber thickness parametrization)
BLADE_SECTION_CAMBER_THICKNESS_DESIGN_VARIABLES =  ['stagger',
                                                    'theta_in',
                                                    'theta_out',
                                                    'radius_in',
                                                    'radius_out',
                                                    'dist_in',
                                                    'dist_out',
                                                    'thickness_upper_1',
                                                    'thickness_upper_2',
                                                    'thickness_upper_3',
                                                    'thickness_upper_4',
                                                    'thickness_upper_5',
                                                    'thickness_upper_6',
                                                    'thickness_lower_1',
                                                    'thickness_lower_2',
                                                    'thickness_lower_3',
                                                    'thickness_lower_4',
                                                    'thickness_lower_5',
                                                    'thickness_lower_6']

# Array of variable names for blade sections (connecting arcs parametrization)
BLADE_SECTION_CONNECTING_ARCS_DESIGN_VARIABLES =   ['stagger',
                                                    'theta_in',
                                                    'theta_out',
                                                    'wedge_in',
                                                    'wedge_out',
                                                    'radius_in',
                                                    'radius_out',
                                                    'dist_1',
                                                    'dist_2',
                                                    'dist_3',
                                                    'dist_4']

# Initialize the design variable names based on PARAMETRIZATION_TYPE
def initialize_DVs_names(PARAMETRIZATION_TYPE):
    if PARAMETRIZATION_TYPE == "CONNECTING_ARCS":
        DVs_names = MERIDIONAL_CHANNEL_DESIGN_VARIABLES + BLADE_SECTION_CONNECTING_ARCS_DESIGN_VARIABLES
    elif PARAMETRIZATION_TYPE == "CAMBER_THICKNESS":
        DVs_names = MERIDIONAL_CHANNEL_DESIGN_VARIABLES + BLADE_SECTION_CAMBER_THICKNESS_DESIGN_VARIABLES
    else:
        raise Exception('Choose a valid option for PARAMETRIZATION_TYPE: "CONNECTING_ARCS" or "CAMBER_THICKNESS"')
    return DVs_names

def update_design_variable_control_points(param_config, parameterization="CONNECTING_ARCS"):
    # This method should eventually be removed; just perform these computations
    # when defining the design variables in the first place

    design_variables = {name: param_config[name] for name in initialize_DVs_names(parameterization)}

    # Convert angle design variables from degrees to radians (be careful about converting the angle twice!)
    for dv in design_variables.keys():
        if dv in ['theta_in', 'theta_out', 'stagger', 'wedge_in', 'wedge_out']:
            for i in range(len(design_variables[dv])):  
                design_variables[dv][i] = design_variables[dv][i] * jnp.pi / 180
        
    # Adjust the hub and shroud control points so that the shared points match exactly
    design_variables['x_hub'] = [param_config['x_leading'][0]] + param_config['x_hub'] + [param_config['x_trailing'][0]]
    design_variables['z_hub'] = [param_config['z_leading'][0]] + param_config['z_hub'] + [param_config['z_trailing'][0]]
    design_variables['x_shroud'] = [param_config['x_leading'][-1]] + param_config['x_shroud'] + [param_config['x_trailing'][-1]]
    design_variables['z_shroud'] = [param_config['z_leading'][-1]] + param_config['z_shroud'] + [param_config['z_trailing'][-1]]

    return design_variables

# Define a function for creating the design variable functions
def get_DVs_functions(design_variables):
    # Design variable functions based on a BSplineCurve parametrization
    def DV_function(P, p, U):
        nn = P.shape[1]
        n = nn - 1
        p = min(n, 3)
        U = jnp.concatenate((jnp.zeros(p), jnp.linspace(0, 1, n - p + 2), jnp.ones(p)))
        return BSplineCurve(U, P, p)
    return {name: DV_function(P=jnp.array([design_variables[name]]), p=None, U=None) for name in design_variables}
