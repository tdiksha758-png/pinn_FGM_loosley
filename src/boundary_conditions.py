# ==================================================
# 🔹 TOP SURFACE (x = -h1)
#
# SHORT CIRCUIT CASE
#
# BC:
# σ_xz = 0
# φ = 0
#
# ==================================================
# def top_surface_bc(model_L1, x_top, params_L1, k, c):
#
#     x_top = x_top.clone().detach().requires_grad_(True)
#
#     k_top = torch.full_like(
#         x_top,
#         k.item() if hasattr(k, 'item') else float(k)
#     )
#
#     inp_top = torch.cat([x_top, k_top], dim=1)
#
#     out = model_L1(inp_top)
#
#     U_r, U_i = out[:, 0:1], out[:, 1:2]
#     Phi_r, Phi_i = out[:, 2:3], out[:, 3:4]
#
#     U_r_x   = grad_bc(U_r,   x_top)
#     U_i_x   = grad_bc(U_i,   x_top)
#     Phi_r_x = grad_bc(Phi_r, x_top)
#     Phi_i_x = grad_bc(Phi_i, x_top)
#
#     C_r = params_L1["C44R1"]
#     C_i = k*c*params_L1["C44I1"]
#
#     e_r = params_L1["e15R1"]
#     e_i = k*c*params_L1["e15I1"]
#
#     sigma_r = (
#         C_r * U_r_x - C_i * U_i_x
#         + e_r * Phi_r_x - e_i * Phi_i_x
#     ) / C_r
#
#     sigma_i = (
#         C_r * U_i_x + C_i * U_r_x
#         + e_r * Phi_i_x + e_i * Phi_r_x
#     ) / C_r
#
#     return sigma_r, sigma_i, Phi_r, Phi_i


# ==================================================
# 🔹 TOP SURFACE (x = -h1)
#
# OPEN CIRCUIT CASE
#
# σ_xz^(1) = 0
# φ^(1) = φ^(a)
# D_x^(1) = D_x^(a)
#
# D_x^(a) = -tau_0 ∂φ^(a)/∂x
#
# ==================================================
import torch

# --------------------------------------------------
# Gradient helper
# --------------------------------------------------
def grad_bc(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]


# ==================================================
# 🔹 TOP SURFACE (x = -h1)  — SHORT CIRCUIT
#    BC: σ_xz = 0,  φ = 0
# ==================================================
def top_surface_bc(
    model_L1,
    model_L3,
    x_top,
    params_L1,
    tau_0,
    k,
    c
):

    x_top = x_top.clone().detach().requires_grad_(True)

    k_top = torch.full_like(
        x_top,
        k.item() if hasattr(k, 'item') else float(k)
    )

    inp_top = torch.cat([x_top, k_top], dim=1)

    # ==================================================
    # Layer 1 (Piezo-viscoelastic)
    # ==================================================
    out1 = model_L1(inp_top)

    U_r, U_i = out1[:, 0:1], out1[:, 1:2]
    Phi1_r, Phi1_i = out1[:, 2:3], out1[:, 3:4]

    U_r_x   = grad_bc(U_r, x_top)
    U_i_x   = grad_bc(U_i, x_top)

    Phi1_r_x = grad_bc(Phi1_r, x_top)
    Phi1_i_x = grad_bc(Phi1_i, x_top)

    # ==================================================
    # Air layer
    # ==================================================
    out3 = model_L3(inp_top)

    PhiA_r = out3[:, 2:3]
    PhiA_i = out3[:, 3:4]

    PhiA_r_x = grad_bc(PhiA_r, x_top)
    PhiA_i_x = grad_bc(PhiA_i, x_top)

    # ==================================================
    # Material parameters
    # ==================================================
    C_r = params_L1["C44R1"]
    C_i = k * c * params_L1["C44I1"]

    e_r = params_L1["e15R1"]
    e_i = k * c * params_L1["e15I1"]

    tau_r = params_L1["tauR1"]
    tau_i = k * c * params_L1["tauI1"]

    # ==================================================
    # σxz = 0
    # ==================================================
    sigma_r = (
        C_r * U_r_x
        - C_i * U_i_x
        + e_r * Phi1_r_x
        - e_i * Phi1_i_x
    ) / C_r

    sigma_i = (
        C_r * U_i_x
        + C_i * U_r_x
        + e_r * Phi1_i_x
        + e_i * Phi1_r_x
    ) / C_r

    # ==================================================
    # φ(1) = φ(a)
    # ==================================================
    phi_r = Phi1_r - PhiA_r
    phi_i = Phi1_i - PhiA_i

    # ==================================================
    # Dx(1)
    # ==================================================
    Dx1_r = (
        e_r * U_r_x
        - e_i * U_i_x
        - tau_r * Phi1_r_x
        + tau_i * Phi1_i_x
    ) / e_r

    Dx1_i = (
        e_r * U_i_x
        + e_i * U_r_x
        - tau_r * Phi1_i_x
        - tau_i * Phi1_r_x
    ) / e_r

    # ==================================================
    # Dx(a) = -tau0 dφ(a)/dx
    # ==================================================
    DxA_r = (-tau_0 * PhiA_r_x) / e_r
    DxA_i = (-tau_0 * PhiA_i_x) / e_r

    # ==================================================
    # Dx(1) = Dx(a)
    # ==================================================
    Dx_r = Dx1_r - DxA_r
    Dx_i = Dx1_i - DxA_i

    return (
        sigma_r,
        sigma_i,
        phi_r,
        phi_i,
        Dx_r,
        Dx_i
    )
# ==================================================
# 🔹 BOTTOM SURFACE (x = h2)
#    BC: σ_xz = 0,  D_x = 0
# ==================================================
def bottom_surface_bc(model_L2, x_bot, params_L2, k, c):

    x_bot = x_bot.clone().detach().requires_grad_(True)
    
    # Create [x, k] input tensor
    k_bot = torch.full_like(x_bot, k.item() if hasattr(k, 'item') else float(k))
    inp_bot = torch.cat([x_bot, k_bot], dim=1)
    
    out = model_L2(inp_bot)

    U_r, U_i = out[:, 0:1], out[:, 1:2]
    Phi_r, Phi_i = out[:, 2:3], out[:, 3:4]

    U_r_x   = grad_bc(U_r,   x_bot)
    U_i_x   = grad_bc(U_i,   x_bot)
    Phi_r_x = grad_bc(Phi_r, x_bot)
    Phi_i_x = grad_bc(Phi_i, x_bot)

    C_r   = params_L2["C44R2"];  C_i   = k*c * params_L2["C44I2"]
    e_r   = params_L2["e15R2"];  e_i   = k*c * params_L2["e15I2"]
    tau_r = params_L2["tauR2"];  tau_i = k*c * params_L2["tauI2"]

    # σ_xz = 0  — divide by C_r
    sigma_r = (C_r * U_r_x - C_i * U_i_x + e_r * Phi_r_x - e_i * Phi_i_x) / C_r
    sigma_i = (C_r * U_i_x + C_i * U_r_x + e_r * Phi_i_x + e_i * Phi_r_x) / C_r

    # D_x = e*∂u - tau*∂φ = 0  — divide by e_r so residual ~ O(∂u)
    Dx_r = (e_r * U_r_x - e_i * U_i_x - tau_r * Phi_r_x + tau_i * Phi_i_x) / e_r
    Dx_i = (e_r * U_i_x + e_i * U_r_x - tau_r * Phi_i_x - tau_i * Phi_r_x) / e_r

    return sigma_r, sigma_i, Dx_r, Dx_i


# ==================================================
# 🔹 INTERFACE (x = 0) — imperfect sliding contact
#
#  (1)  σ1 = (1-δ) σ2
#  (2)  δ σ1 + (1-δ) kF u2 = (1-δ) kF u1
#         ↔  δ σ1 + (1-δ) kF (u2 - u1) = 0
#  (3)  φ1 = (1-δ) φ2
#  (4)  D1 = (1-δ) D2
# ==================================================
def imperfect_interface_bc(
    model_L1, model_L2, x_int,
    params_L1, params_L2, params_int,
    k, c
):
    delta = params_int["delta"]
    F     = params_int["F"]

    x_int = x_int.clone().detach().requires_grad_(True)

    # Create [x, k] input tensors
    k_int = torch.full_like(x_int, k.item() if hasattr(k, 'item') else float(k))
    inp_int = torch.cat([x_int, k_int], dim=1)

    out1 = model_L1(inp_int)
    out2 = model_L2(inp_int)

    U1_r,   U1_i   = out1[:, 0:1], out1[:, 1:2]
    Phi1_r, Phi1_i = out1[:, 2:3], out1[:, 3:4]

    U2_r,   U2_i   = out2[:, 0:1], out2[:, 1:2]
    Phi2_r, Phi2_i = out2[:, 2:3], out2[:, 3:4]

    U1_r_x   = grad_bc(U1_r,   x_int)
    U1_i_x   = grad_bc(U1_i,   x_int)
    Phi1_r_x = grad_bc(Phi1_r, x_int)
    Phi1_i_x = grad_bc(Phi1_i, x_int)

    U2_r_x   = grad_bc(U2_r,   x_int)
    U2_i_x   = grad_bc(U2_i,   x_int)
    Phi2_r_x = grad_bc(Phi2_r, x_int)
    Phi2_i_x = grad_bc(Phi2_i, x_int)

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    C1_r   = params_L1["C44R1"];   C1_i   = k*c * params_L1["C44I1"]
    e1_r   = params_L1["e15R1"];   e1_i   = k*c * params_L1["e15I1"]
    tau1_r = params_L1["tauR1"];   tau1_i = k*c * params_L1["tauI1"]

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    C2_r   = params_L2["C44R2"];   C2_i   = k * c * params_L2["C44I2"]
    e2_r   = params_L2["e15R2"];   e2_i   = k * c * params_L2["e15I2"]
    tau2_r = params_L2["tauR2"];   tau2_i = k * c * params_L2["tauI2"]

    # ── Stresses (physical units, Pa/m) ───────────────────────────────────────
    sig1_r = C1_r*U1_r_x - C1_i*U1_i_x + e1_r*Phi1_r_x - e1_i*Phi1_i_x
    sig1_i = C1_r*U1_i_x + C1_i*U1_r_x + e1_r*Phi1_i_x + e1_i*Phi1_r_x

    sig2_r = C2_r*U2_r_x - C2_i*U2_i_x + e2_r*Phi2_r_x - e2_i*Phi2_i_x
    sig2_i = C2_r*U2_i_x + C2_i*U2_r_x + e2_r*Phi2_i_x + e2_i*Phi2_r_x

    # ── Electric displacements (physical units, C/m²) ─────────────────────────
    Dx1_r = e1_r*U1_r_x - e1_i*U1_i_x - tau1_r*Phi1_r_x + tau1_i*Phi1_i_x
    Dx1_i = e1_r*U1_i_x + e1_i*U1_r_x - tau1_r*Phi1_i_x - tau1_i*Phi1_r_x

    Dx2_r = e2_r*U2_r_x - e2_i*U2_i_x - tau2_r*Phi2_r_x + tau2_i*Phi2_i_x
    Dx2_i = e2_r*U2_i_x + e2_i*U2_r_x - tau2_r*Phi2_i_x - tau2_i*Phi2_r_x

    one_d = 1.0 - delta

    # ── Eq (1): σ1 - (1-δ) σ2 = 0
    #    Units: Pa/m  →  divide by C1_r to get O(∂u) ~ O(1)
    eq1_r = (sig1_r/one_d -  sig2_r) / C1_r
    eq1_i = (sig1_i/one_d -  sig2_i) / C1_r

    # ── Eq (2): δ σ1 + (1-δ) kF (u2 - u1) = 0
    #    [δ σ1] ~ Pa/m,  [(1-δ) kF Δu] ~ (1/m)(Pa/m)(m) = Pa/m  ✓ consistent
    #    divide by C1_r
    eq2_r = ((delta * sig1_r)/one_d + (k * F * (U2_r - U1_r))) / C1_r
    eq2_i = ((delta * sig1_i)/one_d + (k * F * (U2_i - U1_i))) / C1_r

    # ── Eq (3): φ1 - (1-δ) φ2 = 0
    #    Both are network outputs with same units; no extra scaling needed.
    eq3_r = (Phi1_r/one_d - Phi2_r)/C1_r
    eq3_i = (Phi1_i/one_d - Phi2_i)/C1_r

    # ── Eq (4): D1 - (1-δ) D2 = 0
    #    Units: C/m²  →  divide by e1_r to get O(∂u) ~ O(1)
    eq4_r = (Dx1_r/one_d -  Dx2_r) / e1_r
    eq4_i = (Dx1_i/one_d -  Dx2_i) / e1_r

    return eq1_r, eq1_i, eq2_r, eq2_i, eq3_r, eq3_i, eq4_r, eq4_i