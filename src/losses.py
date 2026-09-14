import torch
import torch.nn as nn

from .pde_residual import residual_layer, residual_halfspace
from .boundary_conditions import (
    top_surface_bc,
    halfspace_far_field_bc,
    imperfect_interface_bc
)

mse = nn.MSELoss()


# --------------------------------------------------
# PDE loss
# --------------------------------------------------
def compute_pde_loss(
    model_layer,
    model_half,
    z_layer,
    z_half,
    params_layer,
    params_half,
    k,
    c
):
    """
    PDE residual loss for piezomagnetic layer
    and piezomagnetic half-space.

    Each model gives two outputs:
        U   -> SH-wave displacement
        Phi -> magnetic potential
    """

    # Upper piezomagnetic layer
    rL1, rL2 = residual_layer(
        model_layer,
        z_layer,
        k,
        c,
        params_layer
    )

    # Lower piezomagnetic half-space
    rH1, rH2 = residual_halfspace(
        model_half,
        z_half,
        k,
        c,
        params_half
    )

    loss_pde = (
        mse(rL1, torch.zeros_like(rL1))
        + mse(rL2, torch.zeros_like(rL2))
        + mse(rH1, torch.zeros_like(rH1))
        + mse(rH2, torch.zeros_like(rH2))
    )

    return loss_pde


# --------------------------------------------------
# Top surface boundary loss
# --------------------------------------------------
def compute_top_surface_loss(
    model_layer,
    z_top,
    params_layer
):
    """
    Top surface boundary conditions.

    tau_23^(l) = 0
    Phi^(l) = 0
    """

    res_stress, res_phi = top_surface_bc(
        model_layer,
        z_top,
        params_layer
    )

    loss_stress = mse(
        res_stress,
        torch.zeros_like(res_stress)
    )

    loss_phi = mse(
        res_phi,
        torch.zeros_like(res_phi)
    )

    return loss_stress + loss_phi


# --------------------------------------------------
# Interface loss
# --------------------------------------------------
def compute_interface_loss(
    model_layer,
    model_half,
    z_int,
    params_layer,
    params_half,
    w_disp=1.0,
    w_stress=1.0,
    w_phi=1.0,
    w_B=1.0
):
    """
    Imperfect interface conditions:

        tau_23^(l) = tau_23^(h)

        -tau_23^(h)
        = km [U^(h) - U^(l)]

        Phi^(l) = Phi^(h)

        B_3^(l) = B_3^(h)
    """

    (
        res_disp,
        res_stress,
        res_phi,
        res_B
    ) = imperfect_interface_bc(
        model_layer,
        model_half,
        z_int,
        params_layer,
        params_half
    )

    loss_disp = mse(
        res_disp,
        torch.zeros_like(res_disp)
    )

    loss_stress = mse(
        res_stress,
        torch.zeros_like(res_stress)
    )

    loss_phi = mse(
        res_phi,
        torch.zeros_like(res_phi)
    )

    loss_B = mse(
        res_B,
        torch.zeros_like(res_B)
    )

    loss_interface = (
        w_disp * loss_disp
        + w_stress * loss_stress
        + w_phi * loss_phi
        + w_B * loss_B
    )

    return loss_interface


# --------------------------------------------------
# Far-field loss
# --------------------------------------------------
def compute_far_field_loss(
    model_half,
    z_far
):
    """
    Half-space far-field conditions:

        U^(h) -> 0
        Phi^(h) -> 0
    """

    res_U, res_Phi = halfspace_far_field_bc(
        model_half,
        z_far
    )

    loss_U = mse(
        res_U,
        torch.zeros_like(res_U)
    )

    loss_Phi = mse(
        res_Phi,
        torch.zeros_like(res_Phi)
    )

    return loss_U + loss_Phi


# --------------------------------------------------
# Total loss
# --------------------------------------------------
def total_loss(
    model_layer,
    model_half,
    z_layer,
    z_half,
    z_top,
    z_int,
    z_far,
    params_layer,
    params_half,
    k,
    c,
    w_pde=10.0,
    w_bc=1.0,
    w_int=0.01,
    w_far=0.01,
    w_amp=100.0
):
    """
    Total PINN loss for piezomagnetic SH-wave
    dispersion analysis.
    """

    # --------------------------------------------------
    # Amplitude fixing
    # --------------------------------------------------
    pred_top = model_layer(z_top)

    U_top = pred_top[:, 0:1]

    amp_loss = mse(
        U_top,
        torch.ones_like(U_top)
    )

    # --------------------------------------------------
    # Compute all losses
    # --------------------------------------------------
    loss_pde = compute_pde_loss(
        model_layer,
        model_half,
        z_layer,
        z_half,
        params_layer,
        params_half,
        k,
        c
    )

    loss_bc = compute_top_surface_loss(
        model_layer,
        z_top,
        params_layer
    )

    loss_int = compute_interface_loss(
        model_layer,
        model_half,
        z_int,
        params_layer,
        params_half
    )

    loss_far = compute_far_field_loss(
        model_half,
        z_far
    )

    # --------------------------------------------------
    # Total weighted loss
    # --------------------------------------------------
    loss_total = (
        w_pde * loss_pde
        + w_bc * loss_bc
        + w_int * loss_int
        + w_far * loss_far
        + w_amp * amp_loss
    )

    return loss_total, {
        "pde": loss_pde.item(),
        "bc_top": loss_bc.item(),
        "interface": loss_int.item(),
        "far": loss_far.item(),
        "amp": amp_loss.item()
    }