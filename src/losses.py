import torch
import torch.nn as nn

from .pde_residual import (
    residual_layer1_piezo,
    residual_layer2_piezo
)

from .boundary_conditions import (
    top_surface_bc,
    bottom_surface_bc,
    imperfect_interface_bc
)

mse = nn.MSELoss()


# ==================================================
# PDE LOSS (2 LAYERS — no air layer)
# ==================================================
def compute_pde_loss(
    model_L1, model_L2,
    x_L1, x_L2,
    params_L1, params_L2,
    k, c
):

    # --------------------------------------------------
    # Layer 1
    # --------------------------------------------------
    R1_L1, R2_L1 = residual_layer1_piezo(
        model_L1, x_L1, k, c, params_L1
    )

    # --------------------------------------------------
    # Layer 2
    # --------------------------------------------------
    R1_L2, R2_L2 = residual_layer2_piezo(
        model_L2, x_L2, k, c, params_L2
    )

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------
    scale = 1

    # --------------------------------------------------
    # COMBINED PDE LOSS (BOTH LAYERS)
    # --------------------------------------------------
    loss_pde = (
        mse(R1_L1 / scale, torch.zeros_like(R1_L1)) +
        mse(R2_L1 / scale, torch.zeros_like(R2_L1)) +

        mse(R1_L2 / scale, torch.zeros_like(R1_L2)) +
        mse(R2_L2 / scale, torch.zeros_like(R2_L2))
    )

    return loss_pde


# ==================================================
# TOP SURFACE (x = -h1)
#   BC1: C44_1*dw1/dx + q15_1*dpsi1/dx = 0
#   BC2: psi1 = 0
# ==================================================
def compute_top_surface_loss(model_L1, x_top, params_L1, k):
    bc1, bc2 = top_surface_bc(
        model_L1, x_top, params_L1, k
    )

    # Apply same scaling as PDE for magnitude balance
    scale = 1

    loss = (
        mse(bc1 / scale, torch.zeros_like(bc1)) +
        mse(bc2, torch.zeros_like(bc2))
    )

    return loss


# ==================================================
# BOTTOM SURFACE (x = h2)
#   BC3: C44_2*dw2/dx + q15_2*dpsi2/dx = 0
#   BC4: q15_2*dw2/dx - mu11_2*dpsi2/dx = 0
# ==================================================
def compute_bottom_surface_loss(model_L2, x_bot, params_L2, k):
    bc3, bc4 = bottom_surface_bc(
        model_L2, x_bot, params_L2, k
    )

    # Apply same scaling as PDE for magnitude balance
    scale = 1

    loss = (
        mse(bc3 / scale, torch.zeros_like(bc3)) +
        mse(bc4 / scale, torch.zeros_like(bc4))
    )

    return loss


# ==================================================
# INTERFACE (x = 0)
#   BC5, BC6, BC7, BC8 — imperfect sliding contact
# ==================================================
def compute_interface_loss(
    model_L1,
    model_L2,
    x_int,
    k,
    params_L1,
    params_L2,
    params_int
):

    bc5, bc6, bc7, bc8 = imperfect_interface_bc(
        model_L1, model_L2, x_int,
        params_L1, params_L2, params_int,
        k
    )

    # Apply same scaling as PDE for magnitude balance
    scale = 1e5

    # Combine all 4 interface equations
    loss = (
        mse(bc5 / scale, torch.zeros_like(bc5)) +
        mse(bc6 / scale, torch.zeros_like(bc6)) +
        mse(bc7, torch.zeros_like(bc7)) +
        mse(bc8 / scale, torch.zeros_like(bc8))
    )

    return loss


# ==================================================
# TOTAL LOSS
# ==================================================
def total_loss(
    model_L1,
    model_L2,
    x_L1,
    x_L2,
    x_top,
    x_int,
    x_bot,
    params_L1,
    params_L2,
    params_int,
    k,
    c,
    u_pde=10.0,
    u_bc=5.0,
    u_int=10.0
):

    loss_pde = compute_pde_loss(
        model_L1, model_L2,
        x_L1, x_L2,
        params_L1, params_L2,
        k, c
    )

    loss_top = compute_top_surface_loss(model_L1, x_top, params_L1, k)
    loss_bot = compute_bottom_surface_loss(model_L2, x_bot, params_L2, k)
    loss_int = compute_interface_loss(model_L1, model_L2, x_int, k, params_L1, params_L2, params_int)

    loss_total = (
        u_pde * loss_pde +
        u_bc  * (loss_top + loss_bot) +
        u_int * loss_int
    )

    return loss_total, {
        "pde": loss_pde.item(),
        "bc_top": loss_top.item(),
        "bc_bottom": loss_bot.item(),
        "interface": loss_int.item()
    }