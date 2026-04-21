from sympy import beta
import torch

def gradients(u, x):
    """Compute du/dx with proper gradient tracking."""
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True        
    )[0]

def residual_layer_FGM(model, x3, k, c, params):
    """
    PINN residual for functionally graded SH-wave layer
    """
    x3 = x3.clone().detach().requires_grad_(True)
    U = model(x3)

    # First and second derivatives
    U_x3 = gradients(U, x3)
    U_x3x3 = gradients(U_x3, x3)

    # Parameters
    alpha1 = params["alpha"]
    mu0_l  = params["mu_0"]
    tau0_l = params["P_0"]
    beta_l= (params["mu_0"] / params["rho_0"])**0.5
    
    # Lambda
    Lambda_l = k**2 * ((c/beta_l)**2 - (1 + tau0_l/mu0_l))

    # Residual
    res1 = (1 + alpha1 * x3) * U_x3x3 + alpha1 * U_x3 + Lambda_l * (1 + alpha1 * x3) * U

    return res1


def residual_halfspace_FGM(model, x3, k, c, params):
    """
    PINN residual for functionally graded half-space
    """
    x3 = x3.clone().detach().requires_grad_(True)
    U = model(x3)

    # Derivatives
    U_x3 = gradients(U, x3)
    U_x3x3 = gradients(U_x3, x3)

    # Parameters
    alpha2 = params["alpha"]
    mu0_h  = params["mu_0"]
    tau0_h = params["P_0"]
    beta_h = (params["mu_0"] / params["rho_0"])**0.5
    # Lambda
    Lambda_h = torch.sqrt(1 + tau0_h/mu0_h - (c/beta_h)**2)

    # Residual
    res2 = U_x3x3 + (2*alpha2)/(1 + alpha2*x3) * U_x3 - k**2 * Lambda_h**2 * U

    return res2