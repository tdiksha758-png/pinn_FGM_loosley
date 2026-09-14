from sympy import beta
import torch

def gradients(u, x):
    """Compute du/dx with proper gradient tracking."""
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]


def residual_layer(model, x3, k, c, params):
    """
    PINN residual for piezomagnetic layer
    """

    x3 = x3.clone().detach().requires_grad_(True)

    out = model(x3)

    # Outputs
    U = out[:, 0:1]
    Phi = out[:, 1:2]

    # First and second derivatives
    U_x3 = gradients(U, x3)
    Phi_x3 = gradients(Phi, x3)

    U_x3x3 = gradients(U_x3, x3)
    Phi_x3x3 = gradients(Phi_x3, x3)

        # Parameters
    c44_l = params["c44_l"]
    h15_l = params["h15_l"]
    mu11_l = params["mu11_l"]
    rho_l = params["rho_l"]
    P1 = params["P1"]

    # Mechanical governing equation
    res1 = (
        c44_l * U_x3x3
        + h15_l * Phi_x3x3
        - k**2 * (c44_l + P1 - rho_l * c**2) * U
        - h15_l * k**2 * Phi
    )/c44_l

    # Magnetic governing equation
    res2 = (
        h15_l * U_x3x3
        - mu11_l * Phi_x3x3
        - h15_l * k**2 * U
        + mu11_l * k**2 * Phi
    )/h15_l

    return res1, res2


def residual_halfspace(model, x3, k, c, params):
    """
    PINN residual for piezomagnetic half-space
    """

    x3 = x3.clone().detach().requires_grad_(True)

    out = model(x3)

    # Outputs
    U = out[:, 0:1]
    Phi = out[:, 1:2]

    # Derivatives
    U_x3 = gradients(U, x3)
    Phi_x3 = gradients(Phi, x3)

    U_x3x3 = gradients(U_x3, x3)
    Phi_x3x3 = gradients(Phi_x3, x3)

    # Parameters
    c44_h = params["c44_h"]
    h15_h = params["h15_h"]
    mu11_h = params["mu11_h"]
    rho_h = params["rho_h"]
    P2 = params["P2"]

    # Mechanical governing equation
    res1 = (
        c44_h * U_x3x3
        + h15_h * Phi_x3x3
        - k**2 * (c44_h + P2 - rho_h * c**2) * U
        - h15_h * k**2 * Phi
    )/c44_h

    # Magnetic governing equation
    res2 = (
        h15_h * U_x3x3
        - mu11_h * Phi_x3x3
        - h15_h * k**2 * U
        + mu11_h * k**2 * Phi
    )/h15_h

    return res1, res2