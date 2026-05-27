import torch
import torch.nn as nn

from .pde_residual import (
    pde_residual,
    air_pde_residual
)

from .boundary_conditions import (
    top_bc,
    interface_bc,
    far_bc
)

mse = nn.MSELoss()


# ==================================================
# PDE LOSS
# ==================================================
def compute_pde_loss(
    model_air,
    model_layer,
    model_half,
    z_air,
    z_layer,
    z_half,
    params_air,
    params_L,
    params_H,
    k,
    c
):

    # ==================================================
    # LAYER PDE
    # ==================================================
    (
        R1_r_L,
        R1_i_L,
        R2_r_L,
        R2_i_L

    ) = pde_residual(

        model_layer,

        z_layer,

        k,
        c,

        params_L
    )

    # ==================================================
    # HALF-SPACE PDE
    # ==================================================
    (
        R1_r_H,
        R1_i_H,
        R2_r_H,
        R2_i_H

    ) = pde_residual(

        model_half,

        z_half,

        k,
        c,

        params_H
    )

    # ==================================================
    # AIR / VACUUM PDE
    # ==================================================
    (
        R_air_r,
        R_air_i

    ) = air_pde_residual(

        model_air,

        z_air,

        k,
        c,

        params_air
    )

    # ==================================================
    # LAYER PDE LOSS
    # ==================================================
    loss_L = (

        mse(R1_r_L, torch.zeros_like(R1_r_L))

        +

        mse(R1_i_L, torch.zeros_like(R1_i_L))

        +

        mse(R2_r_L, torch.zeros_like(R2_r_L))

        +

        mse(R2_i_L, torch.zeros_like(R2_i_L))
    )

    # ==================================================
    # HALF-SPACE PDE LOSS
    # ==================================================
    loss_H = (

        mse(R1_r_H, torch.zeros_like(R1_r_H))

        +

        mse(R1_i_H, torch.zeros_like(R1_i_H))

        +

        mse(R2_r_H, torch.zeros_like(R2_r_H))

        +

        mse(R2_i_H, torch.zeros_like(R2_i_H))
    )

    # ==================================================
    # AIR PDE LOSS
    # ==================================================
    loss_air = (

        mse(R_air_r, torch.zeros_like(R_air_r))

        +

        mse(R_air_i, torch.zeros_like(R_air_i))
    )

    # ==================================================
    # RETURN
    # ==================================================
    return (

        loss_L + loss_H,

        loss_air
    )


# ==================================================
# TOP SURFACE BC LOSS
# ==================================================
def compute_top_surface_loss(
    model_air,
    model_layer,
    z_air,
    z_top,
    params_L,
    k,
    c
):

    loss_bc = top_bc(

        model_air,

        model_layer,

        z_air,

        z_top,

        params_L,

        k,
        c
    )

    return loss_bc


# ==================================================
# INTERFACE LOSS
# ==================================================
def compute_interface_loss(
    model_layer,
    model_half,
    z_int,
    params_L,
    params_H,
    k,
    c,
    kappa
):

    loss_int = interface_bc(

        model_layer,

        model_half,

        z_int,

        params_L,

        params_H,

        k,
        c,

        kappa
    )

    return loss_int


# ==================================================
# FAR FIELD LOSS
# ==================================================
def compute_far_field_loss(
    model_half,
    z_far
):

    loss_far = far_bc(

        model_half,

        z_far
    )

    return loss_far


# ==================================================
# TOTAL LOSS
# ==================================================
def total_loss(
    model_air,
    model_layer,
    model_half,
    z_air,
    z_layer,
    z_half,
    z_top,
    z_int,
    z_far,
    params_air,
    params_L,
    params_H,
    k,
    c,
    kappa,
    w_pde=10.0,
    w_air=1.0,
    w_bc=1.0,
    w_int=10.0,
    w_far=0.1,
):

    # ==================================================
    # PDE LOSS
    # ==================================================
    loss_pde, loss_air = compute_pde_loss(

        model_air,

        model_layer,

        model_half,

        z_air,

        z_layer,

        z_half,

        params_air,

        params_L,

        params_H,

        k,
        c
    )

    # ==================================================
    # TOP SURFACE BC
    # ==================================================
    loss_bc = compute_top_surface_loss(

        model_air,

        model_layer,

        z_air,

        z_top,

        params_L,

        k,
        c
    )

    # ==================================================
    # INTERFACE LOSS
    # ==================================================
    loss_int = compute_interface_loss(

        model_layer,

        model_half,

        z_int,

        params_L,

        params_H,

        k,
        c,

        kappa
    )

    # ==================================================
    # FAR FIELD LOSS
    # ==================================================
    loss_far = compute_far_field_loss(

        model_half,

        z_far
    )

    # ==================================================
    # TOTAL LOSS
    # ==================================================
    loss_total = (

        w_pde * loss_pde

        +

        w_air * loss_air

        +

        w_bc * loss_bc

        +

        w_int * loss_int

        +

        w_far * loss_far
    )

    # ==================================================
    # LOGGING
    # ==================================================
    logs = {

        "pde": loss_pde.item(),

        "air": loss_air.item(),

        "bc_top": loss_bc.item(),

        "interface": loss_int.item(),

        "far": loss_far.item()
    }

    return loss_total, logs