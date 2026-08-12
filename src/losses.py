import torch
import torch.nn as nn

from .pde_residual import (
    residual_layer_FGM,
    residual_halfspace_FGM
)

from .boundary_conditions import (
    top_surface_bc,
    halfspace_far_field_bc,
    imperfect_interface_bc
)

mse = nn.MSELoss()


# ==========================================================
# PDE Loss
# ==========================================================

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

    # ------------------------------------------------------
    # Layer residuals
    # ------------------------------------------------------

    rL_real, rL_imag = residual_layer_FGM(
        model_layer,
        z_layer,
        k,
        c,
        params_layer
    )

    # ------------------------------------------------------
    # Half-space residuals
    # ------------------------------------------------------

    rH_real, rH_imag = residual_halfspace_FGM(
        model_half,
        z_half,
        k,
        c,
        params_half
    )

    # ------------------------------------------------------
    # Real and imaginary PDE losses
    # ------------------------------------------------------

    loss_L_real = mse(
        rL_real,
        torch.zeros_like(rL_real)
    )

    loss_L_imag = mse(
        rL_imag,
        torch.zeros_like(rL_imag)
    )

    loss_H_real = mse(
        rH_real,
        torch.zeros_like(rH_real)
    )

    loss_H_imag = mse(
        rH_imag,
        torch.zeros_like(rH_imag)
    )

    # ------------------------------------------------------
    # Total PDE loss
    # ------------------------------------------------------

    loss_pde = (
        loss_L_real
        + loss_L_imag
        + loss_H_real
        + loss_H_imag
    )

    return loss_pde


# ==========================================================
# Top Surface Loss
# ==========================================================

def compute_top_surface_loss(
    model_layer,
    z_top,
    params_layer,
    k,
    c
):

    # ------------------------------------------------------
    # Complex stress-free condition
    # ------------------------------------------------------

    tau_real, tau_imag = top_surface_bc(
        model_layer,
        z_top,
        k,
        c,
        params_layer
    )

    # ------------------------------------------------------
    # Real stress residual
    # ------------------------------------------------------

    loss_tau_real = mse(
        tau_real,
        torch.zeros_like(tau_real)
    )

    # ------------------------------------------------------
    # Imaginary stress residual
    # ------------------------------------------------------

    loss_tau_imag = mse(
        tau_imag,
        torch.zeros_like(tau_imag)
    )

    # ------------------------------------------------------
    # Total top-surface loss
    # ------------------------------------------------------

    loss_bc = (
        loss_tau_real
        + loss_tau_imag
    )

    return loss_bc


# ==========================================================
# Interface Loss
# ==========================================================

def compute_interface_loss(
    model_layer,
    model_half,
    z_int,
    params_layer,
    params_half,
    k,
    c,
    w_disp=1.0,
    w_stress=1.0
):

    (
        res_interface_real,
        res_interface_imag,
        res_stress_real,
        res_stress_imag
    ) = imperfect_interface_bc(

        model_layer,
        model_half,
        z_int,
        k,
        c,
        params_layer,
        params_half
    )

    # ------------------------------------------------------
    # Imperfect interface: real part
    # ------------------------------------------------------

    loss_interface_real = mse(
        res_interface_real,
        torch.zeros_like(res_interface_real)
    )

    # ------------------------------------------------------
    # Imperfect interface: imaginary part
    # ------------------------------------------------------

    loss_interface_imag = mse(
        res_interface_imag,
        torch.zeros_like(res_interface_imag)
    )

    # ------------------------------------------------------
    # Stress continuity: real part
    # ------------------------------------------------------

    loss_stress_real = mse(
        res_stress_real,
        torch.zeros_like(res_stress_real)
    )

    # ------------------------------------------------------
    # Stress continuity: imaginary part
    # ------------------------------------------------------

    loss_stress_imag = mse(
        res_stress_imag,
        torch.zeros_like(res_stress_imag)
    )

    # ------------------------------------------------------
    # Displacement/interface condition
    # ------------------------------------------------------

    loss_disp = (
        loss_interface_real
        + loss_interface_imag
    )

    # ------------------------------------------------------
    # Stress continuity
    # ------------------------------------------------------

    loss_stress = (
        loss_stress_real
        + loss_stress_imag
    )

    # ------------------------------------------------------
    # Total interface loss
    # ------------------------------------------------------

    loss_interface = (
        w_disp * loss_disp
        + w_stress * loss_stress
    )

    return loss_interface


# ==========================================================
# Far-field Loss
# ==========================================================

def compute_far_field_loss(
    model_half,
    z_far
):

    V_real, V_imag = halfspace_far_field_bc(
        model_half,
        z_far
    )

    # ------------------------------------------------------
    # Real displacement decay
    # ------------------------------------------------------

    loss_real = mse(
        V_real,
        torch.zeros_like(V_real)
    )

    # ------------------------------------------------------
    # Imaginary displacement decay
    # ------------------------------------------------------

    loss_imag = mse(
        V_imag,
        torch.zeros_like(V_imag)
    )

    # ------------------------------------------------------
    # Total far-field loss
    # ------------------------------------------------------

    loss_far = (
        loss_real
        + loss_imag
    )

    return loss_far


# ==========================================================
# Total Loss
# ==========================================================

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

    w_pde=0.5,
    w_bc=0.1,
    w_int=0.2,
    w_far=0.1,
    w_amp=0.1

):

    # ======================================================
    # Amplitude normalization
    #
    # V(-H) = 1 + i*0
    # ======================================================

    pred_top = model_layer(z_top)

    V_top_real = pred_top[:, 0:1]
    V_top_imag = pred_top[:, 1:2]

    # ------------------------------------------------------
    # Real amplitude condition
    # ------------------------------------------------------

    amp_real_loss = mse(
        V_top_real,
        torch.ones_like(V_top_real)
    )

    # ------------------------------------------------------
    # Imaginary amplitude condition
    # ------------------------------------------------------

    amp_imag_loss = mse(
        V_top_imag,
        torch.zeros_like(V_top_imag)
    )

    # ------------------------------------------------------
    # Total amplitude loss
    # ------------------------------------------------------

    amp_loss = (
        amp_real_loss
        + amp_imag_loss
    )


    # ======================================================
    # PDE
    # ======================================================

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


    # ======================================================
    # Top surface
    # ======================================================

    loss_bc = compute_top_surface_loss(

        model_layer,

        z_top,

        params_layer,

        k,
        c

    )


    # ======================================================
    # Interface
    # ======================================================

    loss_int = compute_interface_loss(

        model_layer,
        model_half,

        z_int,

        params_layer,
        params_half,

        k,
        c

    )


    # ======================================================
    # Far-field
    # ======================================================

    loss_far = compute_far_field_loss(

        model_half,

        z_far

    )


    # ======================================================
    # Total weighted loss
    # ======================================================

    loss_total = (

        w_pde * loss_pde

        + w_bc * loss_bc

        + w_int * loss_int

        + w_far * loss_far

        + w_amp * amp_loss

    )


    # ======================================================
    # Return
    # ======================================================

    return loss_total, {

        "pde": loss_pde.item(),

        "bc_top": loss_bc.item(),

        "interface": loss_int.item(),

        "far": loss_far.item(),

        "amp": amp_loss.item()

    }