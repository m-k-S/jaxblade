config_params = {
    # Number of blade sections used to create the blade OPTIONS :: integer
    # The value must be at least 2 (even for 2D cases)
    # Increase this value depending on the span-variation complexity of the blade
    "N_SECTIONS": 10,

    # Type of cascade
    # Set CASCADE_TYPE = ANNULAR for an a annular cascade of blades (axisymmetric)
    # Set CASCADE_TYPE = LINEAR for a linear cascade of blades
    "CASCADE_TYPE": "ANNULAR",

    # Parameterized based on connecting arcs or camberline/thickness 
    # OPTIONS: CONNECTING_ARCS or CAMBER_THICKNESS        
    "PARAMETERIZATION_TYPE": "CONNECTING_ARCS",

    # Design variables for the meridional channel
    # Set a straight horizontal line for axial flow cascades
    # Set a straight vertical line for radial flow cascades
    # Set an arbitrary variation for mixed flow cascade
    "x_leading": [0.00, 0.20, 0.10],
    "y_leading": [0.00, 0.05, 0.00],
    "z_leading": [2.00, 2.80, 3.50],
    "x_trailing": [1.00, 1.10, 0.90],
    "z_trailing": [2.20, 3.00, 3.70],
    "x_hub": [0.25, 0.75],
    "z_hub": [2.00, 2.20],
    "x_shroud": [0.30, 0.70],
    "z_shroud": [3.50, 3.70],
    
    # Design variables for a 2D section parametrization based on connecting arcs
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