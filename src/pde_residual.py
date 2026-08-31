import torch

# --------------------------------------------------
# Gradient helper
# --------------------------------------------------
def grad_res(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]


# ==================================================
# 🔹 Layer 1: Piezomagnetic (Upper layer)
#     Fields: w (mechanical displacement), psi (magnetic potential)
#     Real-valued PDE (no real/imag split)
# ==================================================
def residual_layer1_piezo(model_L1, x_L1, k, c, params_L1):

    x_L1 = x_L1.clone().detach().requires_grad_(True)

    # Create [x, k] input tensor
    k_L1 = torch.full_like(x_L1, k.item() if hasattr(k, 'item') else float(k))
    inp_L1 = torch.cat([x_L1, k_L1], dim=1)

    out_L1 = model_L1(inp_L1)

    w1, psi1 = out_L1[:, 0:1], out_L1[:, 1:2]

    # Derivatives (w.r.t. x_L1)
    w1_xx = grad_res(grad_res(w1, x_L1), x_L1)
    psi1_xx = grad_res(grad_res(psi1, x_L1), x_L1)

    C44_1 = params_L1["C44_1"]
    q15_1 = params_L1["q15_1"]
    mu11_1 = params_L1["mu11_1"]
    rho1 = params_L1["rho1"]
    sigma1 = params_L1["sigma_1"]

    # Equation (1): mechanical equilibrium
    R1 = (
        C44_1 * (w1_xx - k**2 * w1)
        + q15_1 * (psi1_xx - k**2 * psi1)
        + k**2 * (rho1 * c**2 - sigma1) * w1
    )/C44_1

    # Equation (2): magnetic (Gauss's law)
    R2 = (
        q15_1 * (w1_xx - k**2 * w1)
        - mu11_1 * (psi1_xx - k**2 * psi1)
    )/q15_1

    return R1, R2


# ==================================================
# 🔹 Layer 2: Piezomagnetic (Lower layer)
#     Fields: w (mechanical displacement), psi (magnetic potential)
#     Real-valued PDE (no real/imag split)
# ==================================================
def residual_layer2_piezo(model_L2, x_L2, k, c, params_L2):

    x_L2 = x_L2.clone().detach().requires_grad_(True)

    # Create [x, k] input tensor
    k_L2 = torch.full_like(x_L2, k.item() if hasattr(k, 'item') else float(k))
    inp_L2 = torch.cat([x_L2, k_L2], dim=1)

    out_L2 = model_L2(inp_L2)

    w2, psi2 = out_L2[:, 0:1], out_L2[:, 1:2]

    # Derivatives (w.r.t. x_L2)
    w2_xx = grad_res(grad_res(w2, x_L2), x_L2)
    psi2_xx = grad_res(grad_res(psi2, x_L2), x_L2)

    C44_2 = params_L2["C44_2"]
    q15_2 = params_L2["q15_2"]
    mu11_2 = params_L2["mu11_2"]
    rho2 = params_L2["rho2"]
    sigma2 = params_L2["sigma_2"]

    # Equation (3): mechanical equilibrium
    R1 = (
        C44_2 * (w2_xx - k**2 * w2)
        + q15_2 * (psi2_xx - k**2 * psi2)
        + k**2 * (rho2 * c**2 - sigma2) * w2
    )/C44_2

    # Equation (4): magnetic (Gauss's law)
    R2 = (
        q15_2 * (w2_xx - k**2 * w2)
        - mu11_2 * (psi2_xx - k**2 * psi2)
    )/q15_2

    return R1, R2