import torch
from .utils import gradients


# ==========================================================
# Top Surface Boundary Condition
#
# tau_23 = 0 at z = -H
# ==========================================================

def top_surface_bc(model_layer, z_top, k, c, material_params):

    z_top = z_top.clone().detach().requires_grad_(True)

    V = model_layer(z_top)

    # ------------------------------------------------------
    # Separate real and imaginary displacement
    # ------------------------------------------------------

    V_real = V[:, 0:1]
    V_imag = V[:, 1:2]

    # ------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------

    V_real_z = gradients(V_real, z_top)
    V_imag_z = gradients(V_imag, z_top)

    # ------------------------------------------------------
    # Material properties
    # ------------------------------------------------------

    Ge0 = material_params["Ge_0"]
    Gv0 = material_params["Gv_0"]
    alpha1 = material_params["alpha"]

    # ------------------------------------------------------
    # Angular frequency
    # ------------------------------------------------------

    omega = k * c

    # ------------------------------------------------------
    # Complex Kelvin-Voigt modulus
    # ------------------------------------------------------

    G_star = Ge0 - 1j * omega * Gv0

    # ------------------------------------------------------
    # Functionally graded modulus
    # ------------------------------------------------------

    G_layer = G_star * (
        1.0 + alpha1 * z_top
    )

    # ------------------------------------------------------
    # Complex displacement derivative
    # ------------------------------------------------------

    V_z = torch.complex(
        V_real_z,
        V_imag_z
    )

    # ------------------------------------------------------
    # Complex shear stress
    # tau_23 = G_layer * dV/dz
    # ------------------------------------------------------

    tau =  V_z

    # ------------------------------------------------------
    # Separate real and imaginary parts
    # ------------------------------------------------------

    tau_real = torch.real(tau)
    tau_imag = torch.imag(tau)

    return tau_real, tau_imag


# ==========================================================
# Imperfect Interface Boundary Condition
#
# z = 0
#
# tau_23 = K (V_half - V_layer)
# ==========================================================

def imperfect_interface_bc(
    model_layer,
    model_half,
    z_int,
    k,
    c,
    params_layer,
    params_half
):

    z_int = z_int.clone().detach().requires_grad_(True)

    # ------------------------------------------------------
    # Network outputs
    # ------------------------------------------------------

    V_layer = model_layer(z_int)
    V_half = model_half(z_int)

    # ------------------------------------------------------
    # Separate real and imaginary parts
    # ------------------------------------------------------

    V_layer_real = V_layer[:, 0:1]
    V_layer_imag = V_layer[:, 1:2]

    V_half_real = V_half[:, 0:1]
    V_half_imag = V_half[:, 1:2]

    # ------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------

    V_layer_real_z = gradients(
        V_layer_real,
        z_int
    )

    V_layer_imag_z = gradients(
        V_layer_imag,
        z_int
    )

    V_half_real_z = gradients(
        V_half_real,
        z_int
    )

    V_half_imag_z = gradients(
        V_half_imag,
        z_int
    )

    # ------------------------------------------------------
    # Angular frequency
    # ------------------------------------------------------

    omega = k * c

    # ======================================================
    # Layer material
    # ======================================================

    Ge_l = params_layer["Ge_0"]
    Gv_l = params_layer["Gv_0"]

    Gstar_l = Ge_l - 1j * omega * Gv_l

    G_layer = Gstar_l * (
        1.0 + params_layer["alpha"] * z_int
    )

    # ======================================================
    # Half-space material
    # ======================================================

    Ge_h = params_half["Ge_0"]
    Gv_h = params_half["Gv_0"]

    Gstar_h = Ge_h - 1j * omega * Gv_h

    G_half = Gstar_h * (
        1.0 + params_half["alpha"] * z_int
    )**2

    # ======================================================
    # Interface stiffness
    # ======================================================

    K = (
        Ge_l
        / (
            params_layer["s"]
            * params_layer["L"]
        )
    )

    # ======================================================
    # Complex displacement fields
    # ======================================================

    V_layer_complex = torch.complex(
        V_layer_real,
        V_layer_imag
    )

    V_half_complex = torch.complex(
        V_half_real,
        V_half_imag
    )

    # ======================================================
    # Complex displacement derivatives
    # ======================================================

    V_layer_z = torch.complex(
        V_layer_real_z,
        V_layer_imag_z
    )

    V_half_z = torch.complex(
        V_half_real_z,
        V_half_imag_z
    )

    # ======================================================
    # Complex stresses
    # ======================================================

    tau_layer = G_layer * V_layer_z

    tau_half = G_half * V_half_z

    # ======================================================
    # Imperfect interface condition
    #
    # tau_layer = K (V_half - V_layer)
    # ======================================================

    res_interface = (
        (tau_layer
        - K * (
            V_half_complex
            - V_layer_complex
        ))/ G_layer
    )

    # ======================================================
    # Stress continuity
    # ======================================================

    res_stress = (
        (tau_layer
        - tau_half)/ G_layer
    )

    # ======================================================
    # Real and imaginary residuals
    # ======================================================

    res_interface_real = torch.real(
        res_interface
    )

    res_interface_imag = torch.imag(
        res_interface
    )

    res_stress_real = torch.real(
        res_stress
    )

    res_stress_imag = torch.imag(
        res_stress
    )

    return (
        res_interface_real,
        res_interface_imag,
        res_stress_real,
        res_stress_imag
    )


# ==========================================================
# Far-field Boundary Condition
#
# V -> 0 as z -> infinity
# ==========================================================

def halfspace_far_field_bc(
    model_half,
    z_far
):

    z_far = z_far.clone().detach().requires_grad_(True)

    V_far = model_half(z_far)

    # ------------------------------------------------------
    # Separate real and imaginary parts
    # ------------------------------------------------------

    V_far_real = V_far[:, 0:1]
    V_far_imag = V_far[:, 1:2]

    return V_far_real, V_far_imag