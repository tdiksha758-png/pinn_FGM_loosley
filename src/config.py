"""
Configuration file for piezomagnetic layered medium (PINN)
Compatible with residual_layer1_piezo, residual_layer2_piezo
(two-layer piezomagnetic structure, imperfect sliding interface — no air layer)

Material data and geometry taken directly from the MATLAB dispersion script.
"""

CONFIG = {

    # --------------------------------------------------
    # 🔹 Layer 1 (Upper Piezomagnetic Layer)
    # --------------------------------------------------
    "LAYER1": {
        "C44_1": 4.53e10,
        "q15_1": 550.0,
        "mu11_1": 5.9e-4,

        "rho1": 5350.0,
        "sigma_1": 1e11,

        "h1": 6.0
    },

    # --------------------------------------------------
    # 🔹 Layer 2 (Lower Piezomagnetic Layer)
    # --------------------------------------------------
    "LAYER2": {
        "C44_2": 15.87e9,
        "q15_2": 166.29,
        "mu11_2": 2.865e-6,

        "rho2": 9100.0,
        "sigma_2": 1e11,

        "h2": 2.0
    },

    # --------------------------------------------------
    # 🔹 Interface Properties
    # --------------------------------------------------
    "INTERFACE": {
        "F": 1e9,        # Interface stiffness
        "delta": 0.02    # Sliding parameter
    },

    # --------------------------------------------------
    # 🔹 Geometry
    # --------------------------------------------------
    "GEOMETRY": {
        "h1": 6.0,
        "h2": 2.0
    },

    # --------------------------------------------------
    # 🔹 Domain definition
    #     Layer 1: [-h1, 0]   (top surface at x = -h1, interface at x = 0)
    #     Layer 2: [0, h2]    (interface at x = 0, bottom surface at x = h2)
    # --------------------------------------------------
    "DOMAIN": {
        "layer1": [-6.0, 0.0],
        "layer2": [0.0, 2.0]
    },

    # --------------------------------------------------
    # 🔹 Wavenumber sweep
    #     k = x / h1, with x in [2.52, 5] from the MATLAB grid
    # --------------------------------------------------
    "WAVENUMBER": {
        "k_min": 0.4576,
        "k_max": 0.8333,
        "num_k": 12
    },

    # --------------------------------------------------
    # 🔹 Training parameters
    # --------------------------------------------------
    "TRAINING": {
        "epochs": 20,
        "learning_rate": 5e-4,

        "loss_weights": {
            "pde": 2.0,
            "bc": 4.0,
            "interface": 1.0,
            "normalization": 1.0
        }
    },

    # --------------------------------------------------
    # 🔹 PINN setup
    # --------------------------------------------------
    "PINN": {
        "input_dim": 2,     # [x, k]
        "output_dim": 2,    # [w, psi]
        "hidden_layers": 3,
        "neurons": 64,
        "activation": "tanh"
    },

    # --------------------------------------------------
    # 🔹 Initial guess for phase velocity
    #     c = y * beta1, beta1 = sqrt(C44_1/rho1), y in [1.85, 2.05] from MATLAB grid
    # --------------------------------------------------
    "INITIAL": {
        "c0": 5674.0
    }
}