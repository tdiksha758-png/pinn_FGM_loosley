import torch
from .utils import gradients


# ==================================================
# COMPUTE τ23 AND D3
# ==================================================
def compute_fields(model, z, params, k, c):

    z = z.clone().detach().requires_grad_(True)

    out = model(z)

    # ==================================================
    # OUTPUTS
    # ==================================================
    # [U_r, U_i, Psi_r, Psi_i]
    # ==================================================

    U_r = out[:, 0:1]
    U_i = out[:, 1:2]

    Psi_r = out[:, 2:3]
    Psi_i = out[:, 3:4]

    # ==================================================
    # DERIVATIVES
    # ==================================================

    U_r_z = gradients(U_r, z)
    U_i_z = gradients(U_i, z)

    Psi_r_z = gradients(Psi_r, z)
    Psi_i_z = gradients(Psi_i, z)

    # ==================================================
    # FREQUENCY
    # ==================================================

    omega = k * c

    # ==================================================
    # MATERIAL PARAMETERS
    # ==================================================

    c44_r = params["c44_star"]
    c44_i = -omega * params["c44_dash"]

    e15_r = params["e15_star"]
    e15_i = -omega * params["e15_dash"]

    a11_r = params["a11_star"]
    a11_i = -omega * params["a11_dash"]

    # ==================================================
    # τ23
    # ==================================================

    tau_r = (

        (c44_r * U_r_z
        - c44_i * U_i_z

        + e15_r * Psi_r_z
        - e15_i * Psi_i_z)/c44_r

    )

    tau_i = (

        (c44_r * U_i_z
        + c44_i * U_r_z

        + e15_r * Psi_i_z
        + e15_i * Psi_r_z)/c44_r

    )

    # ==================================================
    # D3
    # ==================================================

    D3_r = (

        (e15_r * U_r_z
        - e15_i * U_i_z

        - (
            a11_r * Psi_r_z
            - a11_i * Psi_i_z
        ))/e15_r
    )

    D3_i = (

        (e15_r * U_i_z
        + e15_i * U_r_z

        - (
            a11_r * Psi_i_z
            + a11_i * Psi_r_z
        ))/e15_r
        
    )

    return (

        tau_r,
        tau_i,

        D3_r,
        D3_i,

        U_r,
        U_i,

        Psi_r,
        Psi_i
    )


# ==================================================
# TOP SURFACE BC
#
# τ23 = 0
# ψ = ψ_air
# D3 = D_air
# ==================================================
def top_bc(
    model_air,
    model_layer,
    z_air,
    z_top,
    params_layer,
    k,
    c
):

    # ==================================================
    # AIR
    # ==================================================

    z_air = z_air.clone().detach().requires_grad_(True)

    out_air = model_air(z_air)

    Psi_air_r = out_air[:, 0:1]
    Psi_air_i = out_air[:, 1:2]

    Psi_air_r_z = gradients(Psi_air_r, z_air)
    Psi_air_i_z = gradients(Psi_air_i, z_air)

    eps0 = 8.854e-12

    D_air_r = -eps0 * Psi_air_r_z
    D_air_i = -eps0 * Psi_air_i_z

    # ==================================================
    # SOLID
    # ==================================================

    (
        tau_r,
        tau_i,

        D3_r,
        D3_i,

        _,
        _,

        Psi_r,
        Psi_i

    ) = compute_fields(
        model_layer,
        z_top,
        params_layer,
        k,
        c
    )

    # ==================================================
    # τ23 = 0
    # ==================================================

    res_tau = tau_r**2 + tau_i**2

    # ==================================================
    # D3 = D_air
    # ==================================================

    res_D = (

        (D3_r - D_air_r)**2

        +

        (D3_i - D_air_i)**2
    )

    # ==================================================
    # ψ = ψ_air
    # ==================================================

    res_psi = (

        (Psi_r - Psi_air_r)**2

        +

        (Psi_i - Psi_air_i)**2
    )

    return torch.mean(

        res_tau

        +

        res_D

        +

        res_psi
    )


# ==================================================
# INTERFACE CONDITIONS
# ==================================================
def interface_bc(
    model_layer,
    model_half,
    z_int,
    params_L,
    params_H,
    k,
    c,
    kappa
):

    # ==================================================
    # LAYER
    # ==================================================

    (
        tau_L_r,
        tau_L_i,

        D3_L_r,
        D3_L_i,

        U_L_r,
        U_L_i,

        Psi_L_r,
        Psi_L_i

    ) = compute_fields(
        model_layer,
        z_int,
        params_L,
        k,
        c
    )

    # ==================================================
    # HALF-SPACE
    # ==================================================

    (
        tau_H_r,
        tau_H_i,

        D3_H_r,
        D3_H_i,

        U_H_r,
        U_H_i,

        Psi_H_r,
        Psi_H_i

    ) = compute_fields(
        model_half,
        z_int,
        params_H,
        k,
        c
    )

    # ==================================================
    # STRESS CONTINUITY
    # ==================================================

    res_stress = (

        (tau_L_r - tau_H_r)**2

        +

        (tau_L_i - tau_H_i)**2
    )

    # ==================================================
    # IMPERFECT INTERFACE
    # ==================================================

    res_spring = (

        ((tau_H_r / kappa) - (U_H_r - U_L_r))**2

        +

        ((tau_H_i / kappa) - (U_H_i - U_L_i))**2
    )

    # ==================================================
    # PSI CONTINUITY
    # ==================================================

    res_psi = (

        (Psi_L_r - Psi_H_r)**2

        +

        (Psi_L_i - Psi_H_i)**2
    )

    # ==================================================
    # D3 CONTINUITY
    # ==================================================

    res_D3 = (

        (D3_L_r - D3_H_r)**2

        +

        (D3_L_i - D3_H_i)**2
    )

    return torch.mean(

        res_stress

        +

        res_spring

        +

        res_psi

        +

        res_D3
    )


# ==================================================
# FAR FIELD
# ==================================================
def far_bc(model_half, z_far):

    out = model_half(z_far)

    return torch.mean(out**2)