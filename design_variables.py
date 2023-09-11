import numpy as np

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

def update_design_variable_control_points(design_variables):
    # This method should eventually be removed; just perform these computations
    # when defining the design variables in the first place

    # Convert angle design variables from degrees to radians (be careful about converting the angle twice!)
    for dv in design_variables.keys():
        if dv in ['theta_in', 'theta_out', 'stagger', 'wedge_in', 'wedge_out']:
            for i in range(len(design_variables[dv])):  
                design_variables[dv][i] = design_variables[dv][i] * np.pi / 180
        
    # Adjust the hub and shroud control points so that the shared points match exactly
    design_variables['x_hub'] = [design_variables['x_leading'][0]] + design_variables['x_hub'] + [design_variables['x_trailing'][0]]
    design_variables['z_hub'] = [design_variables['z_leading'][0]] + design_variables['z_hub'] + [design_variables['z_trailing'][0]]
    design_variables['x_shroud'] = [design_variables['x_leading'][-1]] + design_variables['x_shroud'] + [design_variables['x_trailing'][-1]]
    design_variables['z_shroud'] = [design_variables['z_leading'][-1]] + design_variables['z_shroud'] + [design_variables['z_trailing'][-1]]

    return design_variables