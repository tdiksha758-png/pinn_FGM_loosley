import torch
import torch.optim as optim
from src.networks import get_all_networks
from src.config import CONFIG
from src.sampling import (
    sample_domain_points,
    sample_top_surface,
    sample_interface,
    sample_far_field
)
from src.losses import total_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# Train for a single wavenumber k
# ==================================================
def train_for_single_k(
    k,
    model_layer,
    model_half,
    c,
    n_epochs=5000,
    n_domain=5000,
    n_bc=1000,
    n_int=1000,
    n_far=1000,
    lr=1e-3
):
    """
    Train PINN for a fixed wave number k (Love-wave / SH mode)
    Returns learned phase velocity c
    """

    print(f"\nTraining for k = {k:.3f} on {DEVICE}")

    # --------------------------------------------------
    # Load parameters
    # --------------------------------------------------
    params_layer = CONFIG["LAYER"]
    params_half  = CONFIG["SUBSTRATE"]
    geom         = CONFIG["GEOMETRY"]

    # Convert parameters to tensors
    params_layer = {
        key: torch.tensor(val, device=DEVICE, dtype=torch.float32)
        if isinstance(val, (int, float)) else val
        for key, val in params_layer.items()
    }

    params_half = {
        key: torch.tensor(val, device=DEVICE, dtype=torch.float32)
        if isinstance(val, (int, float)) else val
        for key, val in params_half.items()
    }

    # Compute layer and half-space shear velocities
    c_shear_layer = torch.sqrt(params_layer["mu_0"] / params_layer["rho_0"])
    c_shear_half  = torch.sqrt(params_half["mu_0"] / params_half["rho_0"])

    # --------------------------------------------------
    # Optimizer (model + eigenvalue)
    # --------------------------------------------------
    optimizer = optim.Adam(
        [
            {"params": model_layer.parameters(), "lr": lr},
            {"params": model_half.parameters(), "lr": lr},
            {"params": [c], "lr": 1e-4},  # eigenvalue learns slowly
        ]
    )

    best_loss = float("inf")
    best_c = c.item()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(1, n_epochs + 1):

        # ---- Sample collocation points ----
        z_layer, z_half = sample_domain_points(n_domain, geom)
        z_top  = sample_top_surface(n_bc, geom)
        z_int  = sample_interface(n_int)
        z_far  = sample_far_field(n_far, geom)

        # Ensure tensors
        for name in ["z_layer", "z_half", "z_top", "z_int", "z_far"]:
            arr = locals()[name]
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
            if arr.ndim == 1:
                arr = arr.unsqueeze(1)
            locals()[name] = arr

        optimizer.zero_grad()

        # ---- Total PINN loss (includes PDE, BCs, interface, far-field, amplitude) ----
        loss, logs = total_loss(
            model_layer,
            model_half,
            z_layer,
            z_half,
            z_top,
            z_int,
            z_far,
            params_layer,
            params_half,
            k,
            c,
            w_pde=1.0,
            w_bc=10.0,
            w_int=50.0,  # interface weight
            w_far=5.0,
            w_amp=100.0  # amplitude fixing at top
        )

        # --------------------------------------------------
        # Love-wave physics penalty
        # --------------------------------------------------
        physics_penalty = (
            1000.0 * torch.relu(c_shear_layer - c)**2 +
            1000.0 * torch.relu(c - c_shear_half)**2
        )

        total_loss_val = loss + physics_penalty
        total_loss_val.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model_layer.parameters()) + list(model_half.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        # Hard clamp for eigenvalue c
        with torch.no_grad():
            c.data.clamp_(c_shear_layer * 1.001, c_shear_half * 0.999)

        # Track best
        if total_loss_val.item() < best_loss:
            best_loss = total_loss_val.item()
            best_c = c.item()

        # Logging
        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"Loss = {loss.item():.3e} | "
                f"Phys = {physics_penalty.item():.2e} | "
                f"c = {c.item():.6f} | "
                f"PDE = {logs.get('pde', 0):.2e} | "
                f"BC = {logs.get('bc_top', 0):.2e} | "
                f"INT = {logs.get('interface', 0):.2e} | "
                f"FAR = {logs.get('far', 0):.2e} | "
                f"AMP = {logs.get('amp',0):.2e}"
            )

    return best_c


# ==================================================
# Dispersion sweep over k
# ==================================================
def train_dispersion():
    geom = CONFIG["GEOMETRY"]

    k_vals = torch.linspace(
        geom["k_min"],
        geom["k_max"],
        geom["num_k"]
    )

    # Initialize networks ONCE (use activation from CONFIG)
    model_layer, model_half = get_all_networks(activation=CONFIG.get("ACTIVATION", "tanh"))
    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # Initialize eigenvalue c
    params_layer = CONFIG["LAYER"]
    params_half  = CONFIG["HALFSPACE"]

    c_shear_layer = (params_layer["mu44_0"] / params_layer["rho_0"]) ** 0.5
    c_shear_half  = (params_half["mu44_0"] / params_half["rho_0"]) ** 0.5

    c = torch.nn.Parameter(
        torch.tensor(0.5*(c_shear_layer + c_shear_half),
                     device=DEVICE,
                     dtype=torch.float32)
    )

    dispersion = []

    for idx, k in enumerate(k_vals):
        print(f"\n{'='*50}\nTraining for k = {k.item():.3f} ({idx+1}/{len(k_vals)})\n{'='*50}")
        c_val = train_for_single_k(
            k.item(),
            model_layer,
            model_half,
            c,
            n_epochs=5000 if idx == 0 else 2500
        )
        dispersion.append([k.item(), c_val])

    return dispersion


# ==================================================
# Main execution
# ==================================================
if __name__ == "__main__":
    print("\nRunning Love-wave PINN solver...\n")

    model_layer, model_half = get_all_networks(activation=CONFIG.get("ACTIVATION", "tanh"))
    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    params_layer = CONFIG["LAYER"]
    params_half  = CONFIG["HALFSPACE"]

    c_init = 0.5 * ((params_layer["mu44_0"]/params_layer["rho_0"])**0.5 +
                    (params_half["mu44_0"]/params_half["rho_0"])**0.5)

    c = torch.nn.Parameter(torch.tensor(c_init, device=DEVICE, dtype=torch.float32))

    test_k = 0.6
    test_c = train_for_single_k(test_k, model_layer, model_half, c,
                                n_epochs=1500, n_domain=200, n_bc=50, n_int=50, n_far=50, lr=1e-3)
    print(f"\n✓ Test result: c({test_k}) = {test_c:.6f}")

    # Full dispersion sweep (uncomment when ready)
    # dispersion = train_dispersion()
    # torch.save(dispersion, "dispersion_curve.pt")