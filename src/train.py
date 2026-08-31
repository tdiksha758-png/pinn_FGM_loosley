import torch
import torch.optim as optim

from .networks import get_all_networks
from .config import CONFIG
from .sampling import (
    sample_domain_points,
    sample_top_surface,
    sample_interface,
    sample_bottom_surface
)
from .losses import total_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# Train for single k
# ==================================================
def train_for_single_k(
    k,
    model_L1,
    model_L2,
    c,
    n_epochs=5000,
    n_domain=5000,
    n_bc=1000,
    n_int=1000,
    lr=1e-3
):

    print(f"\nTraining for k = {k:.3f} on {DEVICE}")

    # --------------------------------------------------
    # Load parameters
    # --------------------------------------------------
    params_L1 = CONFIG["LAYER1"]
    params_L2 = CONFIG["LAYER2"]
    geom      = CONFIG["GEOMETRY"]
    domain    = CONFIG["DOMAIN"]

    def to_tensor_dict(d):
        return {
            k: torch.tensor(v, device=DEVICE, dtype=torch.float32)
            if isinstance(v, (int, float)) else v
            for k, v in d.items()
        }

    params_L1 = to_tensor_dict(params_L1)
    params_L2 = to_tensor_dict(params_L2)
    params_int = to_tensor_dict(CONFIG["INTERFACE"])

    weights = CONFIG["TRAINING"]["loss_weights"]

    # --------------------------------------------------
    # 🔹 Shear wave speeds (ADDED)
    # --------------------------------------------------
    c_shear_L1 = torch.sqrt(params_L1["C44_1"] / params_L1["rho1"])
    c_shear_L2 = torch.sqrt(params_L2["C44_2"] / params_L2["rho2"])

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = optim.Adam(
        [
            {"params": model_L1.parameters(), "lr": lr},
            {"params": model_L2.parameters(), "lr": lr},
            {"params": [c], "lr": 1e-4},
        ]
    )

    best_loss = float("inf")
    best_c = c.item()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(1, n_epochs + 1):

        # -------- Sampling --------
        x_L1, x_L2 = sample_domain_points(n_domain, domain)

        x_top = sample_top_surface(n_bc, geom)
        x_int = sample_interface(n_int)
        x_bot = sample_bottom_surface(n_bc, geom)

        def ensure_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            if x.ndim == 1:
                x = x.unsqueeze(1)
            return x

        x_L1 = ensure_tensor(x_L1)
        x_L2 = ensure_tensor(x_L2)

        x_top = ensure_tensor(x_top)
        x_int = ensure_tensor(x_int)
        x_bot = ensure_tensor(x_bot)

        optimizer.zero_grad()

        # -------- Base PINN loss --------
        loss, logs = total_loss(
         model_L1,
         model_L2,
         x_L1,
         x_L2,
         x_top,
         x_int,
         x_bot,
         params_L1,
         params_L2,
         params_int,
         k,
         c,
         u_pde=weights["pde"],
         u_bc=weights["bc"],
         u_int=weights["interface"]
         )

        # --------------------------------------------------
        # 🔹 Physics penalty using c_shear (ADDED)
        # --------------------------------------------------
        physics_penalty = (
            100.0 * torch.relu(c_shear_L1 - c)**2 +
            100.0 * torch.relu(c - c_shear_L2)**2
        )

        total_loss_val = loss + physics_penalty

        total_loss_val.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model_L1.parameters()) +
            list(model_L2.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        # Track best
        if total_loss_val.item() < best_loss:
            best_loss = total_loss_val.item()
            best_c = c.item()

        # Logging
        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"Loss = {total_loss_val.item():.3e} | "
                f"c = {c.item():.6f} | "
                f"PDE = {logs.get('pde',0):.2e} | "
                f"BC_TOP = {logs.get('bc_top',0):.2e} | "
                f"BC_BOT = {logs.get('bc_bottom',0):.2e} | "
                f"INT = {logs.get('interface',0):.2e}"
            )

    return best_c


# ==================================================
# Dispersion sweep
# ==================================================
def train_dispersion():

    k_vals = torch.linspace(
        CONFIG["WAVENUMBER"]["k_min"],
        CONFIG["WAVENUMBER"]["k_max"],
        CONFIG["WAVENUMBER"]["num_k"]
    )

    model_L1, model_L2 = get_all_networks()

    model_L1.to(DEVICE)
    model_L2.to(DEVICE)

    # 🔹 Better initial guess using shear speeds (ADDED)
    params_L1 = CONFIG["LAYER1"]
    params_L2 = CONFIG["LAYER2"]

    c_init = 0.5 * (
        (params_L1["C44_1"] / params_L1["rho1"])**0.5 +
        (params_L2["C44_2"] / params_L2["rho2"])**0.5
    )

    c = torch.nn.Parameter(
        torch.tensor(c_init, device=DEVICE, dtype=torch.float32)
    )

    dispersion = []

    for idx, k in enumerate(k_vals):

        print(f"\n{'='*50}")
        print(f"Training for k = {k.item():.3f} ({idx+1}/{len(k_vals)})")
        print(f"{'='*50}")

        c_val = train_for_single_k(
            k.item(),
            model_L1,
            model_L2,
            c,
            n_epochs=5000 if idx == 0 else 2500
        )

        dispersion.append([k.item(), c_val])

    return dispersion


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    print("\nRunning PINN solver (2-layer piezomagnetic)...\n")

    results = train_dispersion()

    print("\n✓ Training completed")