import torch

# ==================================================
# Gradient helper
# ==================================================
def grad_res(u, x):

    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]


# ==================================================
# SOLID PDE RESIDUAL
# (Layer / Half-space)
# ==================================================
def pde_residual(
    model,
    z,
    k,
    c,
    params
):

    z = z.clone().detach().requires_grad_(True)

    out = model(z)

    # ==================================================
    # OUTPUTS
    # ==================================================
    # [U_r, U_i, Psi_r, Psi_i]
    # ==================================================

    U_r   = out[:, 0:1]
    U_i   = out[:, 1:2]

    Psi_r = out[:, 2:3]
    Psi_i = out[:, 3:4]

    # ==================================================
    # SECOND DERIVATIVES
    # ==================================================

    def d2(u):

        return grad_res(
            grad_res(u, z),
            z
        )

    U_r_zz = d2(U_r)

    U_i_zz = d2(U_i)

    Psi_r_zz = d2(Psi_r)

    Psi_i_zz = d2(Psi_i)

    # ==================================================
    # omega = k*c
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

    rho = params["rho"]

    sigma = params["sigma"]

    # ==================================================
    # EFFECTIVE COEFFICIENTS
    # ==================================================

    A_r = c44_r + sigma - rho * c**2

    A_i = c44_i

    # ==================================================
    # MECHANICAL PDE
    # ==================================================

    R1_r = (

        (c44_r * U_r_zz
        - c44_i * U_i_zz

        + e15_r * Psi_r_zz
        - e15_i * Psi_i_zz

        - k**2 * (A_r * U_r - A_i * U_i)

        - k**2 * (e15_r * Psi_r - e15_i * Psi_i))/c44_r

    )

    R1_i = (

        (c44_r * U_i_zz
        + c44_i * U_r_zz

        + e15_r * Psi_i_zz
        + e15_i * Psi_r_zz

        - k**2 * (A_r * U_i + A_i * U_r)

        - k**2 * (e15_r * Psi_i + e15_i * Psi_r))/c44_r

    )

    # ==================================================
    # ELECTRIC PDE
    # ==================================================

    R2_r = (

        (e15_r * U_r_zz
        - e15_i * U_i_zz

        - (
            a11_r * Psi_r_zz
            - a11_i * Psi_i_zz
        )

        - k**2 * (
            e15_r * U_r
            - e15_i * U_i
        )

        + k**2 * (
            a11_r * Psi_r
            - a11_i * Psi_i
        ))/e15_r

    )

    R2_i = (

        (e15_r * U_i_zz
        + e15_i * U_r_zz

        - (
            a11_r * Psi_i_zz
            + a11_i * Psi_r_zz
        )

        - k**2 * (
            e15_r * U_i
            + e15_i * U_r
        )

        + k**2 * (
            a11_r * Psi_i
            + a11_i * Psi_r
        ))/e15_r

    )

    # ==================================================
    # RETURN SOLID PDE RESIDUALS
    # ==================================================

    return (

        R1_r,
        R1_i,

        R2_r,
        R2_i
    )


# ==================================================
# AIR / VACUUM PDE
# ==================================================
def air_pde_residual(
    model_air,
    z_air,
    k,
    c,
    params_air
):

    z_air = z_air.clone().detach().requires_grad_(True)

    out = model_air(z_air)

    # ==================================================
    # OUTPUTS
    # ==================================================

    Psi_air_r = out[:, 0:1]

    Psi_air_i = out[:, 1:2]

    # ==================================================
    # SECOND DERIVATIVES
    # ==================================================

    def d2(u):

        return grad_res(
            grad_res(u, z_air),
            z_air
        )

    Psi_air_r_zz = d2(Psi_air_r)

    Psi_air_i_zz = d2(Psi_air_i)

    # ==================================================
    # AIR PDE
    #
    # d²Psi/dz² - k²Psi = 0
    # ==================================================

    R_air_r = (

        Psi_air_r_zz
        - k**2 * Psi_air_r
    )

    R_air_i = (

        Psi_air_i_zz
        - k**2 * Psi_air_i
    )

    # ==================================================
    # RETURN AIR RESIDUALS
    # ==================================================

    return (

        R_air_r,
        R_air_i
    )