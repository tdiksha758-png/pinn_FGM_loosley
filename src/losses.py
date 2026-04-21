import torch
import torch.nn as nn
from .pde_residual import residual_layer_FGM, residual_halfspace_FGM
from .boundary_conditions import top_surface_bc, halfspace_far_field_bc, imperfect_interface_bc

mse = nn.MSELoss()

# --------------------------------------------------
# PDE loss
# --------------------------------------------------
def compute_pde_loss(model_layer, model_half, z_layer, z_half, params_layer, params_half, k, c):
    """
    PDE residual loss for layer and half-space
    """

    # Residuals (real displacement only)
    rL = residual_layer_FGM(model_layer, z_layer, k, c, params_layer)
    rH = residual_halfspace_FGM(model_half, z_half, k, c, params_half)

    loss_pde = mse(rL, torch.zeros_like(rL)) + mse(rH, torch.zeros_like(rH))
    return loss_pde


# --------------------------------------------------
# Top surface boundary loss
# --------------------------------------------------
def compute_top_surface_loss(model_layer, z_top, params_layer):
    """
    Stress-free top surface: tau_23 = 0
    """
    tau = top_surface_bc(model_layer, z_top, params_layer)
    loss_bc = mse(tau, torch.zeros_like(tau))
    return loss_bc



# --------------------------------------------------
# Interface loss (layer ↔ half-space)
# --------------------------------------------------
def compute_interface_loss(
    model_layer,
    model_half,
    z_int,
    params_layer,
    params_half,
    w_disp=1.0,
    w_stress=1.0
):
    """
    Interface loss using imperfect interface BC
    """

    # Get residuals from BC
    res_interface, res_stress = imperfect_interface_bc(
        model_layer, model_half, z_int, params_layer, params_half
    )

    # Loss terms
    loss_disp = mse(res_interface, torch.zeros_like(res_interface))
    loss_stress = mse(res_stress, torch.zeros_like(res_stress))

    # Total interface loss
    return w_disp * loss_disp + w_stress * loss_stress

# --------------------------------------------------
# Far-field loss
# --------------------------------------------------
def compute_far_field_loss(model_half, z_far):
    """
    Half-space decay condition: V -> 0
    """
    V = halfspace_far_field_bc(model_half, z_far)
    loss_far = mse(V, torch.zeros_like(V))
    return loss_far


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
    w_pde = 10.0,
    w_bc  = 1.0,
    w_int = 10,
    w_far = 0.1,
    w_amp = 100
):
    """
    Total PINN loss for SH-wave dispersion analysis
    """
    # Amplitude fixing at top surface
    pred_top = model_layer(z_top)
    V_top = pred_top
    amp_loss = mse(V_top, torch.ones_like(V_top))

    # Compute all losses
    loss_pde = compute_pde_loss(model_layer, model_half, z_layer, z_half, params_layer, params_half, k, c)
    loss_bc  = compute_top_surface_loss(model_layer, z_top, params_layer)
    loss_int = compute_interface_loss(model_layer, model_half, z_int, params_layer, params_half)
    loss_far = compute_far_field_loss(model_half, z_far)

    # Total weighted loss
    loss_total = w_pde * loss_pde + w_bc * loss_bc + w_int * loss_int + w_far * loss_far + w_amp * amp_loss

    return loss_total, {
        "pde": loss_pde.item(),
        "bc_top": loss_bc.item(),
        "interface": loss_int.item(),
        "far": loss_far.item(),
        "amp": amp_loss.item()
    }