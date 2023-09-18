import jax.numpy as jnp
from .parameterization.bspline import BSplineCurve

class DesignVariables:
    def __init__(self, config_params, parameterization_type="CONNECTING_ARCS"):
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
    
        if parameterization_type == "CONNECTING_ARCS":
            DV_names = MERIDIONAL_CHANNEL_DESIGN_VARIABLES + BLADE_SECTION_CONNECTING_ARCS_DESIGN_VARIABLES
        elif parameterization_type == "CAMBER_THICKNESS":
            DV_names = MERIDIONAL_CHANNEL_DESIGN_VARIABLES + BLADE_SECTION_CAMBER_THICKNESS_DESIGN_VARIABLES
        else:
            raise Exception('Choose a valid option for PARAMETRIZATION_TYPE: "CONNECTING_ARCS" or "CAMBER_THICKNESS"')
        
        self.parameterization = parameterization_type
        self.names = DV_names

        DV_control_points = {name: config_params[name] for name in DV_names}

        for dv in DV_control_points:
            if dv in ['theta_in', 'theta_out', 'stagger', 'wedge_in', 'wedge_out']:
                for i in range(len(DV_control_points[dv])):  
                    DV_control_points[dv][i] = DV_control_points[dv][i] * jnp.pi / 180

        # Adjust the hub and shroud control points so that the shared points match exactly
        DV_control_points['x_hub'] = [config_params['x_leading'][0]] + config_params['x_hub'] + [config_params['x_trailing'][0]]
        DV_control_points['z_hub'] = [config_params['z_leading'][0]] + config_params['z_hub'] + [config_params['z_trailing'][0]]
        DV_control_points['x_shroud'] = [config_params['x_leading'][-1]] + config_params['x_shroud'] + [config_params['x_trailing'][-1]]
        DV_control_points['z_shroud'] = [config_params['z_leading'][-1]] + config_params['z_shroud'] + [config_params['z_trailing'][-1]]

        # Define the design variable functions
        self.functions = self.get_DVs_functions(DV_control_points)  
        self.control_points = DV_control_points

    # Define a function for creating the design variable functions
    def get_DVs_functions(self, design_variables_control_points):

        # Design variable functions based on a BSplineCurve parametrization
        def DV_function(P, p, U):
            nn = P.shape[1]
            n = nn - 1
            p = min(n, 3)
            U = jnp.concatenate((jnp.zeros(p), jnp.linspace(0, 1, n - p + 2), jnp.ones(p)))
            return BSplineCurve(U, P, p)
        
        return {name: DV_function(P=jnp.array([design_variables_control_points[name]]), p=None, U=None) for name in design_variables_control_points}
