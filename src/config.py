"""
Configuration file for piezo-viscoelastic layered medium (PINN)
Compatible with residual_layer1_piezo, residual_layer2_piezo, residual_layer3_air
"""

CONFIG = {

    # --------------------------------------------------
    # 🔹 Layer 1 (Upper Piezo-viscoelastic Layer)
    # --------------------------------------------------
    "LAYER1": {
        "C44R1": 7.511e9,
        "C44I1": 9.7252e9,

        "e15R1": -0.0919,
        "e15I1": -0.000285,

        "tauR1": -0.3041e-9,
        "tauI1": -0.02e-12,

        "rho1": 3949.0,
        "sigma1": 1e11,

        "h1": 8.0
    },

    # --------------------------------------------------
    # 🔹 Layer 2 (Lower Piezo-viscoelastic Layer)
    # --------------------------------------------------
    "LAYER2": {
        "C44R2": 8.172e9,
        "C44I2": 32.239e9,

        "e15R2": -0.0968,
        "e15I2": -0.0003,

        "tauR2": -0.4187e-9,
        "tauI2": -2.55e-12,

        "rho2": 3259.0,
        "sigma2": 1e11,

        "h2": 2.0
    },

    # --------------------------------------------------
    # 🔹 Layer 3 (Air / Vacuum)
    # --------------------------------------------------
    "AIR": {
        "h3": 30.0,
        "tau_0": 8.854e-12
    },

    # --------------------------------------------------
    # 🔹 Interface Properties
    # --------------------------------------------------
    "INTERFACE": {
        "F": 1e9,        # Interface stiffness
        "delta": 0.2     # Sliding parameter
    },

    # --------------------------------------------------
    # 🔹 Geometry
    # --------------------------------------------------
    "GEOMETRY": {
        "h1": 8.0,
        "h2": 2.0,
        "h3": 30.0
    },
    # 🔹 Domain definition
    # --------------------------------------------------
    "DOMAIN": {
        "AIR":    [-30.0, -8.0],   # [-h1-h3, -h1]
        "LAYER1": [-8.0,   0.0],   # [-h1, 0]
        "LAYER2": [ 0.0,   2.0]    # [0, h2]
    },

    # --------------------------------------------------
    # 🔹 Wavenumber sweep
    # --------------------------------------------------
    "WAVENUMBER": {
        "k_min": 0.3308,
        "k_max": 1.1995,
        "num_k": 8
    },

    # --------------------------------------------------
    # 🔹 Training parameters
    # --------------------------------------------------
    "TRAINING": {
        "epochs": 20,
        "learning_rate": 5e-4,

        "loss_weights": {
            "pde": 10.0,
            "air": 1.0,
           "bc": 5.0,
         "interface": 10.0,
         "normalization": 1.0
        }
    },

    # --------------------------------------------------
    # 🔹 PINN setup
    # --------------------------------------------------
    "PINN": {
        "input_dim": 1,
        "output_dim": 4,   # [U_r, U_i, Phi_r, Phi_i]
        "hidden_layers": 3,
        "neurons": 64,
        "activation": "tanh"
    },
    
    # --------------------------------------------------
    # 🔹 Initial guess for phase velocity
    # --------------------------------------------------
    "INITIAL": {
        "c0": 1000.0   # Initial guess for phase velocity
    }
}