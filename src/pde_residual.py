import torch


# ==========================================================
# Gradient utility
# ==========================================================

def gradients(u, x):
    """
    Compute du/dx with proper gradient tracking.
    """

    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]


# ==========================================================
# Layer Residual
# ==========================================================

def residual_layer_FGM(model, x3, k, c, params):
    """
    Kelvin-Voigt viscoelastic FGM layer.

    Network output:
        U[:,0] = real part
        U[:,1] = imaginary part

    Returns:
        residual_real
        residual_imag
    """

    x3 = x3.clone().detach().requires_grad_(True)

    # ------------------------------------------------------
    # Network output
    # ------------------------------------------------------

    U = model(x3)

    U_real = U[:, 0:1]
    U_imag = U[:, 1:2]

    # ------------------------------------------------------
    # First derivatives
    # ------------------------------------------------------

    U_real_x3 = gradients(U_real, x3)
    U_imag_x3 = gradients(U_imag, x3)

    # ------------------------------------------------------
    # Second derivatives
    # ------------------------------------------------------

    U_real_x3x3 = gradients(U_real_x3, x3)
    U_imag_x3x3 = gradients(U_imag_x3, x3)

    # ------------------------------------------------------
    # Material parameters
    # ------------------------------------------------------

    alpha1 = params["alpha"]

    Ge0 = params["Ge_0"]
    Gv0 = params["Gv_0"]

    rho0 = params["rho_0"]
    P0 = params["P_0"]

    # ------------------------------------------------------
    # Angular frequency
    # ------------------------------------------------------

    omega = k * c

    # ------------------------------------------------------
    # Complex Kelvin-Voigt shear modulus
    # ------------------------------------------------------

    G_star = Ge0 - 1j * omega * Gv0

    # ------------------------------------------------------
    # Complex SH-wave velocity
    # ------------------------------------------------------

    beta = torch.sqrt(G_star / rho0)

    # ------------------------------------------------------
    # Lambda
    # ------------------------------------------------------

    Lambda = k**2 * (
        (c / beta)**2
        - (1.0 + P0 / G_star)
    )

    # ------------------------------------------------------
    # Construct complex displacement
    # ------------------------------------------------------

    U_complex = torch.complex(
        U_real,
        U_imag
    )

    U_x3 = torch.complex(
        U_real_x3,
        U_imag_x3
    )

    U_x3x3 = torch.complex(
        U_real_x3x3,
        U_imag_x3x3
    )

    # ------------------------------------------------------
    # Complex PDE residual
    # ------------------------------------------------------

    res = (
        (1.0 + alpha1 * x3) * U_x3x3
        + alpha1 * U_x3
        + Lambda * (1.0 + alpha1 * x3) * U_complex
    )

    # ------------------------------------------------------
    # Separate real and imaginary residuals
    # ------------------------------------------------------

    res_real = torch.real(res)
    res_imag = torch.imag(res)

    return res_real, res_imag


# ==========================================================
# Half-space Residual
# ==========================================================

def residual_halfspace_FGM(model, x3, k, c, params):
    """
    Kelvin-Voigt viscoelastic FGM half-space.

    Network output:
        U[:,0] = real part
        U[:,1] = imaginary part

    Returns:
        residual_real
        residual_imag
    """

    x3 = x3.clone().detach().requires_grad_(True)

    # ------------------------------------------------------
    # Network output
    # ------------------------------------------------------

    U = model(x3)

    U_real = U[:, 0:1]
    U_imag = U[:, 1:2]

    # ------------------------------------------------------
    # First derivatives
    # ------------------------------------------------------

    U_real_x3 = gradients(U_real, x3)
    U_imag_x3 = gradients(U_imag, x3)

    # ------------------------------------------------------
    # Second derivatives
    # ------------------------------------------------------

    U_real_x3x3 = gradients(U_real_x3, x3)
    U_imag_x3x3 = gradients(U_imag_x3, x3)

    # ------------------------------------------------------
    # Material parameters
    # ------------------------------------------------------

    alpha2 = params["alpha"]

    Ge0 = params["Ge_0"]
    Gv0 = params["Gv_0"]

    rho0 = params["rho_0"]
    P0 = params["P_0"]

    # ------------------------------------------------------
    # Angular frequency
    # ------------------------------------------------------

    omega = k * c

    # ------------------------------------------------------
    # Complex Kelvin-Voigt shear modulus
    # ------------------------------------------------------

    G_star = Ge0 - 1j * omega * Gv0

    # ------------------------------------------------------
    # Complex SH-wave velocity
    # ------------------------------------------------------

    beta = torch.sqrt(G_star / rho0)

    # ------------------------------------------------------
    # Lambda
    # ------------------------------------------------------

    Lambda = torch.sqrt(
        1.0
        + P0 / G_star
        - (c / beta)**2
    )

    # ------------------------------------------------------
    # Construct complex displacement
    # ------------------------------------------------------

    U_complex = torch.complex(
        U_real,
        U_imag
    )

    U_x3 = torch.complex(
        U_real_x3,
        U_imag_x3
    )

    U_x3x3 = torch.complex(
        U_real_x3x3,
        U_imag_x3x3
    )

    # ------------------------------------------------------
    # Complex PDE residual
    # ------------------------------------------------------

    res = (
        U_x3x3
        + (2.0 * alpha2)
        / (1.0 + alpha2 * x3)
        * U_x3
        - k**2 * Lambda**2 * U_complex
    )

    # ------------------------------------------------------
    # Separate real and imaginary residuals
    # ------------------------------------------------------

    res_real = torch.real(res)
    res_imag = torch.imag(res)

    return res_real, res_imag