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
# 🔹 Layer 1: Piezo-viscoelastic (Upper layer)
# ==================================================
def residual_layer1_piezo(model_L1, x_L1, k, c, params_L1):

    x_L1 = x_L1.clone().detach().requires_grad_(True)
    
    # Create [x, k] input tensor
    k_L1 = torch.full_like(x_L1, k.item() if hasattr(k, 'item') else float(k))
    inp_L1 = torch.cat([x_L1, k_L1], dim=1)
    
    out_L1 = model_L1(inp_L1)

    U1_r, U1_i = out_L1[:, 0:1], out_L1[:, 1:2]
    Phi1_r, Phi1_i = out_L1[:, 2:3], out_L1[:, 3:4]

    # Derivatives (w.r.t. x_L1)
    U1_r_xx = grad_res(grad_res(U1_r, x_L1), x_L1)
    U1_i_xx = grad_res(grad_res(U1_i, x_L1), x_L1)

    Phi1_r_xx = grad_res(grad_res(Phi1_r, x_L1), x_L1)
    Phi1_i_xx = grad_res(grad_res(Phi1_i, x_L1), x_L1)

    C1_r = params_L1["C44R1"]
    C1_i = k*c * params_L1["C44I1"]

    e1_r = params_L1["e15R1"]
    e1_i = k*c * params_L1["e15I1"]

    tau1_r = params_L1["tauR1"]
    tau1_i = k*c * params_L1["tauI1"]

    rho1 = params_L1["rho1"]
    sigma1 = params_L1["sigma1"]

    # Equation 1
    R1_real = (
        C1_r * U1_r_xx - C1_i * U1_i_xx
        + e1_r * Phi1_r_xx - e1_i * Phi1_i_xx
        - k**2 * ((C1_r + sigma1 - rho1 * c**2) * U1_r - C1_i * U1_i)
        - k**2 * (e1_r * Phi1_r - e1_i * Phi1_i)
    )/C1_r

    R1_imag = (
        C1_r * U1_i_xx + C1_i * U1_r_xx
        + e1_r * Phi1_i_xx + e1_i * Phi1_r_xx
        - k**2 * ((C1_r + sigma1 - rho1 * c**2) * U1_i + C1_i * U1_r)
        - k**2 * (e1_r * Phi1_i + e1_i * Phi1_r)
    )/C1_r

    # Equation 2
    R2_real = (
        e1_r * U1_r_xx - e1_i * U1_i_xx
        - (tau1_r * Phi1_r_xx - tau1_i * Phi1_i_xx)
        + e1_r * U1_r - e1_i * U1_i
        - (tau1_r * Phi1_r - tau1_i * Phi1_i)
    )/e1_r

    R2_imag = (
        e1_r * U1_i_xx + e1_i * U1_r_xx
        - (tau1_r * Phi1_i_xx + tau1_i * Phi1_r_xx)
        + e1_r * U1_i + e1_i * U1_r
        - (tau1_r * Phi1_i + tau1_i * Phi1_r)
    )/e1_r

    return R1_real, R1_imag, R2_real, R2_imag


# ==================================================
# 🔹 Layer 2: Piezo-viscoelastic (Lower layer)
# ==================================================
def residual_layer2_piezo(model_L2, x_L2, k, c, params_L2):

    x_L2 = x_L2.clone().detach().requires_grad_(True)
    
    # Create [x, k] input tensor
    k_L2 = torch.full_like(x_L2, k.item() if hasattr(k, 'item') else float(k))
    inp_L2 = torch.cat([x_L2, k_L2], dim=1)
    
    out_L2 = model_L2(inp_L2)

    U2_r, U2_i = out_L2[:, 0:1], out_L2[:, 1:2]
    Phi2_r, Phi2_i = out_L2[:, 2:3], out_L2[:, 3:4]

    # Derivatives (w.r.t. x_L2)
    U2_r_xx = grad_res(grad_res(U2_r, x_L2), x_L2)
    U2_i_xx = grad_res(grad_res(U2_i, x_L2), x_L2)

    Phi2_r_xx = grad_res(grad_res(Phi2_r, x_L2), x_L2)
    Phi2_i_xx = grad_res(grad_res(Phi2_i, x_L2), x_L2)

    C2_r = params_L2["C44R2"]
    C2_i = k*c * params_L2["C44I2"]

    e2_r = params_L2["e15R2"]
    e2_i = k*c * params_L2["e15I2"]

    tau2_r = params_L2["tauR2"]
    tau2_i = k*c * params_L2["tauI2"]

    rho2 = params_L2["rho2"]
    sigma2 = params_L2["sigma2"]

    # Equation 1
    R1_real = (
        C2_r * U2_r_xx - C2_i * U2_i_xx
        + e2_r * Phi2_r_xx - e2_i * Phi2_i_xx
        - k**2 * ((C2_r + sigma2 - rho2 * c**2) * U2_r - C2_i * U2_i)
        - k**2 * (e2_r * Phi2_r - e2_i * Phi2_i)
    )/C2_r

    R1_imag = (
        C2_r * U2_i_xx + C2_i * U2_r_xx
        + e2_r * Phi2_i_xx + e2_i * Phi2_r_xx
        - k**2 * ((C2_r + sigma2 - rho2 * c**2) * U2_i + C2_i * U2_r)
        - k**2 * (e2_r * Phi2_i + e2_i * Phi2_r)
    )/C2_r

    # Equation 2
    R2_real = (
        e2_r * U2_r_xx - e2_i * U2_i_xx
        - (tau2_r * Phi2_r_xx - tau2_i * Phi2_i_xx)
        + e2_r * U2_r - e2_i * U2_i
        - (tau2_r * Phi2_r - tau2_i * Phi2_i)
    )/e2_r

    R2_imag = (
        e2_r * U2_i_xx + e2_i * U2_r_xx
        - (tau2_r * Phi2_i_xx + tau2_i * Phi2_r_xx)
        + e2_r * U2_i + e2_i * U2_r
        - (tau2_r * Phi2_i + tau2_i * Phi2_r)
    )/e2_r

    return R1_real, R1_imag, R2_real, R2_imag


# ==================================================
# 🔹 Layer 3: Air / Vacuum
# ==================================================
def residual_layer3_air(model_L3, x_L3, k, tau_0):

    x_L3 = x_L3.clone().detach().requires_grad_(True)

    # Create [x, k] input tensor
    k_L3 = torch.full_like(x_L3, k.item() if hasattr(k, 'item') else float(k))
    inp_L3 = torch.cat([x_L3, k_L3], dim=1)
    
    out_L3 = model_L3(inp_L3)

    Phi3_r = out_L3[:, 2:3]
    Phi3_i = out_L3[:, 3:4]

    Phi3_r_x  = grad_res(Phi3_r, x_L3)
    Phi3_i_x  = grad_res(Phi3_i, x_L3)

    Phi3_r_xx = grad_res(Phi3_r_x, x_L3)
    Phi3_i_xx = grad_res(Phi3_i_x, x_L3)

    # PDE residual
    R_phi_real = Phi3_r_xx - k**2 * Phi3_r
    R_phi_imag = Phi3_i_xx - k**2 * Phi3_i

    # Derived electric displacement
    Dx3_r = -tau_0 * Phi3_r_x
    Dx3_i = -tau_0 * Phi3_i_x

    return (
        R_phi_real,
        R_phi_imag,
        Dx3_r,
        Dx3_i
    )