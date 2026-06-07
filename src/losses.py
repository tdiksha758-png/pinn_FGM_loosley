import torch
import torch.nn as nn

from .pde_residual import (
    residual_layer1_piezo,
    residual_layer2_piezo,
    residual_layer3_air
)

from .boundary_conditions import (
    top_surface_bc,
    bottom_surface_bc,
    imperfect_interface_bc
)

mse = nn.MSELoss()


# ==================================================
# RESIDUAL NORMALIZATION HELPER
# ==================================================
def normalize_residuals(residuals, scale_factor=1e9):
    """
    Normalize residuals by dividing by a characteristic scale
    to avoid large coefficient magnitudes overwhelming the loss.
    
    Args:
        residuals: list of residual tensors
        scale_factor: typical magnitude of material parameters
    """
    normalized = []
    for res in residuals:
        # Normalize by scale factor to prevent overflow
        # Keep numerical stability by clamping
        norm_res = res / (scale_factor + 1e-12)
        normalized.append(norm_res)
    return normalized


# ==================================================
# PDE LOSS (3 LAYERS) - WITH NORMALIZATION
# ==================================================
# ==================================================
# PDE LOSS (3 LAYERS)
# ==================================================
def compute_pde_loss(
    model_L1, model_L2, model_L3,
    x_L1, x_L2, x_L3,
    params_L1, params_L2, params_L3,
    k, c
):

    # --------------------------------------------------
    # Layer 1
    # --------------------------------------------------
    R1r_L1, R1i_L1, R2r_L1, R2i_L1 = residual_layer1_piezo(
        model_L1, x_L1, k, c, params_L1
    )

    # --------------------------------------------------
    # Layer 2
    # --------------------------------------------------
    R1r_L2, R1i_L2, R2r_L2, R2i_L2 = residual_layer2_piezo(
        model_L2, x_L2, k, c, params_L2
    )

    # --------------------------------------------------
    # Air Layer
    # --------------------------------------------------
    R3r, R3i, Dx3r, Dx3i = residual_layer3_air(
        model_L3,
        x_L3,
        k,
        params_L3["tau_0"]
    )

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------
    scale = 1e4

    # --------------------------------------------------
    # COMBINED PDE LOSS (ALL LAYERS)
    # --------------------------------------------------
    loss_pde = (

        mse(R1r_L1 / scale, torch.zeros_like(R1r_L1)) +
        mse(R1i_L1 / scale, torch.zeros_like(R1i_L1)) +
        mse(R2r_L1 / scale, torch.zeros_like(R2r_L1)) +
        mse(R2i_L1 / scale, torch.zeros_like(R2i_L1)) +

        mse(R1r_L2 / scale, torch.zeros_like(R1r_L2)) +
        mse(R1i_L2 / scale, torch.zeros_like(R1i_L2)) +
        mse(R2r_L2 / scale, torch.zeros_like(R2r_L2)) +
        mse(R2i_L2 / scale, torch.zeros_like(R2i_L2)) +

        mse(R3r, torch.zeros_like(R3r)) +
        mse(R3i, torch.zeros_like(R3i)) +
        mse(Dx3r / scale, torch.zeros_like(Dx3r)) +
        mse(Dx3i / scale, torch.zeros_like(Dx3i))

    )

    return loss_pde


# ==================================================
# TOP SURFACE (x = -h1)
# ==================================================
def compute_top_surface_loss(
    model_L1,
    model_L3,
    x_top,
    params_L1,
    params_L3,
    k,
    c
):

    sigma_r, sigma_i, phi_r, phi_i, Dx_r, Dx_i = top_surface_bc(
        model_L1,
        model_L3,
        x_top,
        params_L1,
        params_L3["tau_0"],
        k,
        c
    )

    scale = 1e4

    loss = (

        mse(sigma_r / scale,
            torch.zeros_like(sigma_r))

        +

        mse(sigma_i / scale,
            torch.zeros_like(sigma_i))

        +

        mse(phi_r / scale,
            torch.zeros_like(phi_r))

        +

        mse(phi_i / scale,
            torch.zeros_like(phi_i))

        +

        mse(Dx_r / scale,
            torch.zeros_like(Dx_r))

        +

        mse(Dx_i / scale,
            torch.zeros_like(Dx_i))
    )

    return loss

# ==================================================
# BOTTOM SURFACE (x = h2)
# ==================================================
def compute_bottom_surface_loss(model_L2, x_bot, params_L2, k, c):
    sigma_r, sigma_i, Dx_r, Dx_i = bottom_surface_bc(
        model_L2, x_bot, params_L2, k, c
    )

    # ✅ Apply same scaling as PDE for magnitude balance
    scale = 1e4
    
    loss = (
        mse(sigma_r / scale, torch.zeros_like(sigma_r)) +
        mse(sigma_i / scale, torch.zeros_like(sigma_i)) +
        mse(Dx_r / scale, torch.zeros_like(Dx_r)) +
        mse(Dx_i / scale, torch.zeros_like(Dx_i))
    )

    return loss


# ==================================================
# INTERFACE (x = 0)
# ==================================================
def compute_interface_loss(
    model_L1,
    model_L2,
    x_int,
    k,
    c,
    params_L1,
    params_L2,
    params_int
):

    eq1_r, eq1_i, eq2_r, eq2_i, eq3_r, eq3_i, eq4_r, eq4_i = imperfect_interface_bc(
        model_L1, model_L2, x_int,
        params_L1, params_L2, params_int,
        k, c
    )

    # ✅ Apply same scaling as PDE for magnitude balance
    scale = 1e5
    
    # Combine all 4 interface equations (stress, displacement, potential, E-displacement)
    loss = (
        mse(eq1_r / scale, torch.zeros_like(eq1_r)) +
        mse(eq1_i / scale, torch.zeros_like(eq1_i)) +
        mse(eq2_r / scale, torch.zeros_like(eq2_r)) +
        mse(eq2_i / scale, torch.zeros_like(eq2_i)) +
        mse(eq3_r / scale, torch.zeros_like(eq3_r)) +
        mse(eq3_i / scale, torch.zeros_like(eq3_i)) +
        mse(eq4_r / scale, torch.zeros_like(eq4_r)) +
        mse(eq4_i / scale, torch.zeros_like(eq4_i))
    )

    return loss



# ==================================================
# TOTAL LOSS
# ==================================================
def total_loss(
    model_L1,
    model_L2,
    model_L3,
    x_L1,
    x_L2,
    x_L3,
    x_top,
    x_int,
    x_bot,
    params_L1,
    params_L2,
    params_L3,
    params_int,
    k,
    c,
    u_pde=10.0,
    u_bc=5.0,
    u_int=5.0
):

    loss_pde = compute_pde_loss(
        model_L1, model_L2, model_L3,
        x_L1, x_L2, x_L3,
        params_L1, params_L2, params_L3,
        k, c
    )

    loss_top = compute_top_surface_loss(model_L1, model_L3, x_top, params_L1, params_L3, k, c)
    loss_bot = compute_bottom_surface_loss(model_L2, x_bot, params_L2, k, c)
    loss_int = compute_interface_loss(model_L1, model_L2, x_int, k, c, params_L1, params_L2, params_int)
    # BUG 7 FIX: pass x_int as second normalization anchor
    loss_total = (
        u_pde * loss_pde +
        u_bc  * (loss_top + loss_bot) +
        u_int * loss_int
    )

    return loss_total, {
        "pde":       loss_pde.item(),
        "bc_top":    loss_top.item(),
        "bc_bottom": loss_bot.item(),
        "interface": loss_int.item()
    }