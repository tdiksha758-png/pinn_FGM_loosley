"""
Configuration file for SH-wave dispersion in a functionally graded
Kelvin-Voigt viscoelastic layered medium using PINNs
"""

CONFIG = {

    # --------------------------------------------------
    # Functionally Graded Viscoelastic Layer (Linear variation)
    # --------------------------------------------------
    "LAYER": {
        "Ge_0": 4.34e10,          # Elastic shear modulus (Pa)
        "Gv_0": 6.77e10/50,       # Viscous shear modulus (Pa.s)

        "mu_0": 4.34e10,          # (Retained for compatibility)
        "rho_0": 2217.0,          # Density (kg/m^3)
        "P_0": 5.0e10,            # Initial stress (Pa)

        "alpha": 1.0,            # Linear grading parameter

        "s": 100.0,

        "L": 0.5                 # Layer thickness
    },

    # --------------------------------------------------
    # Functionally Graded Viscoelastic Substrate
    # (Quadratic variation)
    # --------------------------------------------------
    "SUBSTRATE": {
        "Ge_0": 6.77e10,          # Elastic shear modulus (Pa)
        "Gv_0": 4.34e10/60,       # Viscous shear modulus (Pa.s)

        "mu_0": 6.77e10,          # (Retained for compatibility)
        "rho_0": 3333.0,
        "P_0": 1.0e10,

        "alpha": 1.0
    },

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------
    "GEOMETRY": {
        "L": 0.5,
        "H_trunc": 20.0
    },

    # --------------------------------------------------
    # Wavenumber Sweep
    # --------------------------------------------------
    "WAVENUMBER": {
        "k_min": 0.194468,
        "k_max": 2.00,
        "num_k": 30
    },

    # --------------------------------------------------
    # Training Parameters
    # --------------------------------------------------
    "TRAINING": {
        "epochs": 20000,
        "learning_rate": 5e-4,

        "loss_weights": {
            "pde": 10.0,
            "bc": 1.0,
            "interface": 0.01,
            "far": 0.01
        }
    },

    # --------------------------------------------------
    # Reference Quantities
    # --------------------------------------------------
    "REFERENCE": {
        "beta_l": None,
        "beta_h": None
    }

}