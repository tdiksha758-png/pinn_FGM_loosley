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
# 🔹 TOP SURFACE (x = -h1)  — Layer 1
#
#  BC1:  C44_1 * dw1/dx + q15_1 * dpsi1/dx = 0
#  BC2:  psi1 = 0
# ==================================================
def top_surface_bc(model_L1, x_top, params_L1, k):

    x_top = x_top.clone().detach().requires_grad_(True)

    # Create [x, k] input tensor
    k_top = torch.full_like(x_top, k.item() if hasattr(k, 'item') else float(k))
    inp_top = torch.cat([x_top, k_top], dim=1)

    out = model_L1(inp_top)

    w1, psi1 = out[:, 0:1], out[:, 1:2]

    w1_x   = grad_bc(w1,   x_top)
    psi1_x = grad_bc(psi1, x_top)

    C44_1 = params_L1["C44_1"]
    q15_1 = params_L1["q15_1"]

    # BC1: generalized stress-free surface
    bc1 = (C44_1 * w1_x + q15_1 * psi1_x)/C44_1

    # BC2: magnetic potential vanishes
    bc2 = psi1

    return bc1, bc2


# ==================================================
# 🔹 BOTTOM SURFACE (x = h2)  — Layer 2
#
#  BC3:  C44_2 * dw2/dx + q15_2 * dpsi2/dx = 0
#  BC4:  q15_2 * dw2/dx - mu11_2 * dpsi2/dx = 0
# ==================================================
def bottom_surface_bc(model_L2, x_bot, params_L2, k):

    x_bot = x_bot.clone().detach().requires_grad_(True)

    # Create [x, k] input tensor
    k_bot = torch.full_like(x_bot, k.item() if hasattr(k, 'item') else float(k))
    inp_bot = torch.cat([x_bot, k_bot], dim=1)

    out = model_L2(inp_bot)

    w2, psi2 = out[:, 0:1], out[:, 1:2]

    w2_x   = grad_bc(w2,   x_bot)
    psi2_x = grad_bc(psi2, x_bot)

    C44_2  = params_L2["C44_2"]
    q15_2  = params_L2["q15_2"]
    mu11_2 = params_L2["mu11_2"]

    # BC3: generalized stress-free surface
    bc3 = (C44_2 * w2_x + q15_2 * psi2_x)/C44_2

    # BC4: magnetic induction condition
    bc4 = (q15_2 * w2_x - mu11_2 * psi2_x)/q15_2

    return bc3, bc4


# ==================================================
# 🔹 INTERFACE (x = 0) — imperfect sliding contact
#
#  BC5: [C44_1 dw1/dx + q15_1 dpsi1/dx] - (1-δ)[C44_2 dw2/dx + q15_2 dpsi2/dx] = 0
#  BC6: δ[C44_1 dw1/dx + q15_1 dpsi1/dx] + (1-δ)kF(w2 - w1) = 0
#  BC7: psi1 - (1-δ) psi2 = 0
#  BC8: [q15_1 dw1/dx - mu11_1 dpsi1/dx] - (1-δ)[q15_2 dw2/dx - mu11_2 dpsi2/dx] = 0
# ==================================================
def imperfect_interface_bc(
    model_L1, model_L2, x_int,
    params_L1, params_L2, params_int,
    k
):
    delta = params_int["delta"]
    F     = params_int["F"]

    x_int = x_int.clone().detach().requires_grad_(True)

    # Create [x, k] input tensor
    k_int = torch.full_like(x_int, k.item() if hasattr(k, 'item') else float(k))
    inp_int = torch.cat([x_int, k_int], dim=1)

    out1 = model_L1(inp_int)
    out2 = model_L2(inp_int)

    w1, psi1 = out1[:, 0:1], out1[:, 1:2]
    w2, psi2 = out2[:, 0:1], out2[:, 1:2]

    w1_x   = grad_bc(w1,   x_int)
    psi1_x = grad_bc(psi1, x_int)

    w2_x   = grad_bc(w2,   x_int)
    psi2_x = grad_bc(psi2, x_int)

    # ── Layer 1 material constants ──────────────────────────────
    C44_1  = params_L1["C44_1"]
    q15_1  = params_L1["q15_1"]
    mu11_1 = params_L1["mu11_1"]

    # ── Layer 2 material constants ──────────────────────────────
    C44_2  = params_L2["C44_2"]
    q15_2  = params_L2["q15_2"]
    mu11_2 = params_L2["mu11_2"]

    # ── Generalized stresses ────────────────────────────────────
    stress1 = C44_1 * w1_x + q15_1 * psi1_x
    stress2 = C44_2 * w2_x + q15_2 * psi2_x

    # ── Generalized magnetic inductions ─────────────────────────
    induction1 = q15_1 * w1_x - mu11_1 * psi1_x
    induction2 = q15_2 * w2_x - mu11_2 * psi2_x

    one_d = 1.0 - delta

    # BC5: stress continuity (sliding-weighted)
    bc5 = (stress1 - one_d * stress2)/C44_1

    # BC6: sliding/traction balance
    bc6 = (delta * stress1 + one_d * k * F * (w2 - w1))/C44_1

    # BC7: magnetic potential continuity (sliding-weighted)
    bc7 = psi1 - one_d * psi2

    # BC8: magnetic induction continuity (sliding-weighted)
    bc8 = (induction1 - one_d * induction2)/q15_1

    return bc5, bc6, bc7, bc8