import torch
from .utils import gradients


# --------------------------------------------------
# Top surface boundary condition (z = -h1)
# --------------------------------------------------
def top_surface_bc(model_layer, z_top, material_params):
    """
    Top surface boundary conditions:

        tau_23^(l) = 0
        Phi^(l) = 0
    """

    z_top = z_top.clone().detach().requires_grad_(True)

    out = model_layer(z_top)

    U_l = out[:, 0:1]
    Phi_l = out[:, 1:2]

    # Derivatives
    U_l_z = gradients(U_l, z_top)
    Phi_l_z = gradients(Phi_l, z_top)

    # Material parameters
    c44_l = material_params["c44_l"]
    h15_l = material_params["h15_l"]

    # Shear stress:
    #
    # tau_23^(l) =
    # c44_l * dU_l/dz
    # + h15_l * dPhi_l/dz

    tau_l = (
        c44_l * U_l_z
        + h15_l * Phi_l_z
    )/c44_l

    # Magnetic potential condition
    res_phi = Phi_l

    return tau_l, res_phi


# --------------------------------------------------
# Imperfect interface boundary conditions (z = 0)
# --------------------------------------------------
def imperfect_interface_bc(
    model_layer,
    model_half,
    z_int,
    params_layer,
    params_half
):
    """
    Imperfect interface conditions at z = 0:

        tau_23^(l) = tau_23^(h)

        -tau_23^(h)
        = km [U^(h) - U^(l)]

        Phi^(l) = Phi^(h)

        B_3^(l) = B_3^(h)
    """

    z_int = z_int.clone().detach().requires_grad_(True)

    # --------------------------------------------------
    # Model outputs
    # --------------------------------------------------

    out_l = model_layer(z_int)
    out_h = model_half(z_int)

    U_l = out_l[:, 0:1]
    Phi_l = out_l[:, 1:2]

    U_h = out_h[:, 0:1]
    Phi_h = out_h[:, 1:2]

    # --------------------------------------------------
    # Derivatives
    # --------------------------------------------------

    U_l_z = gradients(U_l, z_int)
    Phi_l_z = gradients(Phi_l, z_int)

    U_h_z = gradients(U_h, z_int)
    Phi_h_z = gradients(Phi_h, z_int)

    # --------------------------------------------------
    # Material parameters
    # --------------------------------------------------

    c44_l = params_layer["c44_l"]
    h15_l = params_layer["h15_l"]
    mu11_l = params_layer["mu11_l"]

    c44_h = params_half["c44_h"]
    h15_h = params_half["h15_h"]
    mu11_h = params_half["mu11_h"]

    # --------------------------------------------------
    # Interface stiffness
    #
    # km = c44cap / (s*h1)
    # --------------------------------------------------

    c44cap = (
        c44_h
        + (h15_h**2 / mu11_h)
    )/c44_h

    s = params_layer["s"]
    h1 = params_layer["h1"]

    km = c44cap / (s * h1)

    # --------------------------------------------------
    # Shear stresses
    # --------------------------------------------------

    tau_l = (
        c44_l * U_l_z
        + h15_l * Phi_l_z
    )/c44_l

    tau_h = (
        c44_h * U_h_z
        + h15_h * Phi_h_z
    )/c44_h

    # --------------------------------------------------
    # Magnetic inductions B_3
    #
    # B_3 =
    # h15 * dU/dz - mu11 * dPhi/dz
    # --------------------------------------------------

    B3_l = (
        h15_l * U_l_z
        - mu11_l * Phi_l_z
    )/h15_l

    B3_h = (
        h15_h * U_h_z
        - mu11_h * Phi_h_z
    )/h15_h

    # --------------------------------------------------
    # Interface residuals
    # --------------------------------------------------

    # (17) Stress continuity
    res_stress = tau_l - tau_h

    # (18) Imperfect mechanical interface
    #
    # -tau_h = km * (U_h - U_l)
    #
    res_disp = (
        -tau_h
        - km * (U_h - U_l)
    )

    # (19) Magnetic potential continuity
    res_phi = Phi_l - Phi_h

    # (20) Magnetic induction continuity
    res_B = B3_l - B3_h

    return (
        res_disp,
        res_stress,
        res_phi,
        res_B
    )


# --------------------------------------------------
# Far-field boundary condition (z -> infinity)
# --------------------------------------------------
def halfspace_far_field_bc(model_half, z_far):
    """
    Far-field conditions for the lower piezomagnetic half-space:

        U^(h) -> 0
        Phi^(h) -> 0
    """

    z_far = z_far.clone().detach().requires_grad_(True)

    out_h = model_half(z_far)

    U_h = out_h[:, 0:1]
    Phi_h = out_h[:, 1:2]

    return U_h, Phi_h