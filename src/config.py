"""
Configuration file for SH-wave dispersion in a piezomagnetic
layered medium using PINNs
"""

CONFIG = {

    # --------------------------------------------------
    # Piezomagnetic Upper Layer
    # --------------------------------------------------
    "LAYER": {
        "c44_l": 45.3e9,          # Shear modulus (Pa)
        "h15_l": 550.0,           # Piezomagnetic coefficient
        "mu11_l": 157e-6,         # Magnetic permeability
        "rho_l": 5.3e3,           # Density (kg/m^3)
        "P1": 1e10,              # Initial stress along x1 (Pa)
        "s": 1000.0,            # Interface parameter
        "h1": 2.0,               # Layer thickness
    },

    # --------------------------------------------------
    # Piezomagnetic Lower Half-Space
    # --------------------------------------------------
    "SUBSTRATE": {
        "c44_h": 15.87e9,         # Shear modulus (Pa)
        "h15_h": 166.29,          # Piezomagnetic coefficient
        "mu11_h": 2.865e-6,       # Magnetic permeability
        "rho_h": 9100.0,          # Density (kg/m^3)
        "P2": 1e10,              # Initial stress along x1
    },


    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------
    "GEOMETRY": {
        "h1": 2.0,               # Layer thickness
        "H_trunc": 30.0,        # Truncated depth for half-space
    },


    # --------------------------------------------------
    # Wavenumber sweep
    # --------------------------------------------------
    "WAVENUMBER": {
        "k_min": 0.00867383,
        "k_max": 0.05,
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
    # Reference velocity
    # --------------------------------------------------
    "REFERENCE": {
        "beta_l": None
    }

}