"""
Configuration file for SH-wave dispersion in a functionally graded
layered medium using PINNs
"""

CONFIG = {

    # --------------------------------------------------
    # Functionally Graded Layer (Linear variation)
    # --------------------------------------------------
    "LAYER": {
        "mu_0": 0.3e11,         # Shear modulus at reference (Pa)
        "rho_0": 7500.0,          # Density (kg/m^3)
        "P_0": 1e7,               # Initial stress along x1 (Pa)
        "alpha": 0.2,             # Grading parameter along thickness
        "s": 100.0,               # Raw stiffness of imperfect interface
        "L": 2.0,                 # Layer thickness
    },

    # --------------------------------------------------
    # Functionally Graded Substrate (Quadratic variation)
    # --------------------------------------------------
    "SUBSTRATE": {
        "mu_0": 0.28e11,          # Shear modulus at reference (Pa)
        "rho_0": 2800.0,          # Density (kg/m^3)
        "P_0": 1e7,               # Initial stress along x1 (Pa)
        "alpha": 0.9,             # Quadratic grading parameter
    },

    
    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------
    "GEOMETRY": {
        "L": 2.0,                 # Layer thickness
        "H_trunc": 30.0,          # Truncated depth for substrate
    },

    # --------------------------------------------------
    # Wavenumber sweep (non-dimensional)
    # --------------------------------------------------
    "WAVENUMBER": {
        "k_min": 0.052,
        "k_max": 0.25,
        "num_k": 16
    },

    # --------------------------------------------------
    # Training parameters
    # --------------------------------------------------
    "TRAINING": {
        "epochs": 20000,
        "learning_rate": 5e-4,
        "loss_weights": {
            "pde": 10.0,         # Weight for PDE residual
            "bc": 1.0,           # Weight for boundary conditions
            "interface": 0.01,   # Weight for imperfect interface
            "far": 0.01          # Weight for far-field decay
        }
    },

    # --------------------------------------------------
    # PINN normalization / reference
    # --------------------------------------------------
    "REFERENCE": {
        "beta_l": None,           # Can compute from mu_0/rho_0 for layer
        "beta_h": None            # Can compute from mu_0/rho_0 for substrate
    }

}