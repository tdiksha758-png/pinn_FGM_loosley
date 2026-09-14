import torch
import torch.optim as optim

from networks import get_all_networks
from config import CONFIG
from sampling import (
    sample_domain_points,
    sample_top_surface,
    sample_interface,
    sample_far_field
)
from losses import total_loss


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
    Train PINN for a fixed wavenumber k.

    Returns learned phase velocity c.
    """

    print(f"\nTraining for k = {k:.3f} on {DEVICE}")

    # --------------------------------------------------
    # Load parameters
    # --------------------------------------------------
    params_layer = CONFIG["LAYER"]
    params_half = CONFIG["SUBSTRATE"]
    geom = CONFIG["GEOMETRY"]

    # --------------------------------------------------
    # Convert numerical parameters to tensors
    # --------------------------------------------------
    params_layer = {
        key: torch.tensor(
            val,
            device=DEVICE,
            dtype=torch.float32
        )
        if isinstance(val, (int, float))
        else val
        for key, val in params_layer.items()
    }

    params_half = {
        key: torch.tensor(
            val,
            device=DEVICE,
            dtype=torch.float32
        )
        if isinstance(val, (int, float))
        else val
        for key, val in params_half.items()
    }

    # --------------------------------------------------
    # Reference / shear velocities
    #
    # beta_l = sqrt(c44_l / rho_l)
    # --------------------------------------------------
    beta_l = torch.sqrt(
        params_layer["c44_l"] /
        params_layer["rho_l"]
    )

    beta_h = torch.sqrt(
        params_half["c44_h"] /
        params_half["rho_h"]
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = optim.Adam(
        [
            {
                "params": model_layer.parameters(),
                "lr": lr
            },
            {
                "params": model_half.parameters(),
                "lr": lr
            },
            {
                "params": [c],
                "lr": 1e-4
            }
        ]
    )

    best_loss = float("inf")
    best_c = c.item()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(1, n_epochs + 1):

        # ----------------------------------------------
        # Sample collocation points
        # ----------------------------------------------
        z_layer, z_half = sample_domain_points(
            n_domain,
            geom
        )

        z_top = sample_top_surface(
            n_bc,
            geom
        )

        z_int = sample_interface(
            n_int
        )

        z_far = sample_far_field(
            n_far,
            geom
        )

        optimizer.zero_grad()

        # ----------------------------------------------
        # Total PINN loss
        # ----------------------------------------------
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
            w_pde=10.0,
            w_bc=1.0,
            w_int=0.01,
            w_far=0.01,
            w_amp=100.0
        )

        # ----------------------------------------------
        # Phase velocity constraint
        # ----------------------------------------------
        physics_penalty = (
            1000.0 *
            torch.relu(beta_l - c)**2
        )

        total_loss_val = (
            loss + physics_penalty
        )

        # ----------------------------------------------
        # Backpropagation
        # ----------------------------------------------
        total_loss_val.backward()

        # ----------------------------------------------
        # Gradient clipping
        # ----------------------------------------------
        torch.nn.utils.clip_grad_norm_(
            list(model_layer.parameters())
            + list(model_half.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        # ----------------------------------------------
        # Keep c in a physically reasonable range
        # ----------------------------------------------
        with torch.no_grad():

            c.data.clamp_(
                beta_l * 0.001,
                beta_l * 0.999
            )

        # ----------------------------------------------
        # Track best solution
        # ----------------------------------------------
        if total_loss_val.item() < best_loss:

            best_loss = total_loss_val.item()
            best_c = c.item()

        # ----------------------------------------------
        # Logging
        # ----------------------------------------------
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
                f"AMP = {logs.get('amp', 0):.2e}"
            )

    return best_c


# ==================================================
# Dispersion sweep over k
# ==================================================
def train_dispersion():

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------
    geom = CONFIG["GEOMETRY"]

    # --------------------------------------------------
    # Wavenumber values
    # --------------------------------------------------
    wave = CONFIG["WAVENUMBER"]

    k_vals = torch.linspace(
        wave["k_min"],
        wave["k_max"],
        wave["num_k"],
        device=DEVICE
    )

    # --------------------------------------------------
    # Initialize networks
    # --------------------------------------------------
    model_layer, model_half = get_all_networks()

    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # --------------------------------------------------
    # Material parameters
    # --------------------------------------------------
    params_layer = CONFIG["LAYER"]
    params_half = CONFIG["SUBSTRATE"]

    # --------------------------------------------------
    # Reference velocity
    #
    # beta_l = sqrt(c44_l / rho_l)
    # --------------------------------------------------
    beta_l = (
        params_layer["c44_l"]
        / params_layer["rho_l"]
    ) ** 0.5

    # --------------------------------------------------
    # Initial phase velocity
    # --------------------------------------------------
    c_init = 0.5 * beta_l

    c = torch.nn.Parameter(
        torch.tensor(
            c_init,
            device=DEVICE,
            dtype=torch.float32
        )
    )

    dispersion = []

    # --------------------------------------------------
    # Sweep over wavenumber
    # --------------------------------------------------
    for idx, k in enumerate(k_vals):

        print(
            f"\n{'='*50}\n"
            f"Training for k = {k.item():.3f} "
            f"({idx + 1}/{len(k_vals)})\n"
            f"{'='*50}"
        )

        c_val = train_for_single_k(
            k.item(),
            model_layer,
            model_half,
            c,
            n_epochs=5000 if idx == 0 else 2500
        )

        dispersion.append(
            [k.item(), c_val]
        )

    return dispersion


# ==================================================
# Main execution
# ==================================================
if __name__ == "__main__":

    print(
        "\nRunning piezomagnetic SH-wave "
        "PINN solver...\n"
    )

    # --------------------------------------------------
    # Initialize networks
    # --------------------------------------------------
    model_layer, model_half = get_all_networks()

    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # --------------------------------------------------
    # Material parameters
    # --------------------------------------------------
    params_layer = CONFIG["LAYER"]
    params_half = CONFIG["SUBSTRATE"]

    # --------------------------------------------------
    # Reference velocity
    # --------------------------------------------------
    beta_l = (
        params_layer["c44_l"]
        / params_layer["rho_l"]
    ) ** 0.5

    # --------------------------------------------------
    # Initial phase velocity
    # --------------------------------------------------
    c_init = 0.5 * beta_l

    c = torch.nn.Parameter(
        torch.tensor(
            c_init,
            device=DEVICE,
            dtype=torch.float32
        )
    )

    # --------------------------------------------------
    # Test case
    # --------------------------------------------------
    test_k = 0.05

    test_c = train_for_single_k(
        test_k,
        model_layer,
        model_half,
        c,
        n_epochs=1500,
        n_domain=200,
        n_bc=50,
        n_int=50,
        n_far=50,
        lr=1e-3
    )

    print(
        f"\n✓ Test result: "
        f"c({test_k}) = {test_c:.6f}"
    )

    # --------------------------------------------------
    # Full dispersion sweep
    # --------------------------------------------------
    # dispersion = train_dispersion()
    # torch.save(dispersion, "dispersion_curve.pt")