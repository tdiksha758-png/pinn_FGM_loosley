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


# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ==========================================================
# Train for a Single Wavenumber
# ==========================================================

def train_for_single_k(
    k,
    model_layer,
    model_half,
    c,
    n_epochs=5000,
    n_domain=1000,
    n_bc=200,
    n_int=200,
    n_far=200,
    lr=1e-3
):

    print(
        f"\nTraining for k = {k:.4f}"
    )

    # ------------------------------------------------------
    # Parameters
    # ------------------------------------------------------

    params_layer = CONFIG["LAYER"]
    params_half = CONFIG["SUBSTRATE"]

    geom = CONFIG["GEOMETRY"]

    # ------------------------------------------------------
    # Convert numerical parameters to tensors
    # ------------------------------------------------------

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

    # ======================================================
    # Optimizer
    # ======================================================

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

    # ======================================================
    # Best solution
    # ======================================================

    best_loss = float("inf")

    best_c = c.detach().item()


    # ======================================================
    # Training Loop
    # ======================================================

    for epoch in range(
        1,
        n_epochs + 1
    ):

        # --------------------------------------------------
        # Sample collocation points
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Make sure tensors have correct dimensions
        # --------------------------------------------------

        points = [
            z_layer,
            z_half,
            z_top,
            z_int,
            z_far
        ]

        points = [

            p.to(
                DEVICE,
                dtype=torch.float32
            )

            for p in points

        ]

        z_layer, z_half, z_top, z_int, z_far = points


        # --------------------------------------------------
        # Clear gradients
        # --------------------------------------------------

        optimizer.zero_grad()


        # ==================================================
        # PINN Loss
        # ==================================================

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

            # ----------------------------------------------
            # Loss weights
            # ----------------------------------------------

            w_pde=1.0,

            w_bc=5.0,

            w_int=10.0,

            w_far=5.0,

            w_amp=1.0

        )


        # ==================================================
        # Backpropagation
        # ==================================================

        loss.backward()


        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            list(model_layer.parameters())
            +
            list(model_half.parameters())
            +
            [c],

            max_norm=1.0

        )


        # --------------------------------------------------
        # Optimizer step
        # --------------------------------------------------

        optimizer.step()


        # ==================================================
        # Track best solution
        # ==================================================

        if loss.item() < best_loss:

            best_loss = loss.item()

            best_c = c.detach().item()


        # ==================================================
        # Logging
        # ==================================================

        if epoch % 100 == 0:

            print(

                f"Epoch {epoch:5d} | "

                f"Loss = {loss.item():.4e} | "

                f"PDE = {logs['pde']:.4e} | "

                f"BC = {logs['bc_top']:.4e} | "

                f"INT = {logs['interface']:.4e} | "

                f"FAR = {logs['far']:.4e} | "

                f"AMP = {logs['amp']:.4e} | "

                f"c = {c.item():.6f}"

            )


    # ======================================================
    # Final result
    # ======================================================

    print(
        f"\nBest result for k = {k:.4f}: "
        f"c = {best_c:.6f}"
    )

    print(
        f"Best loss = {best_loss:.4e}"
    )

    return best_c


# ==========================================================
# Dispersion Sweep
# ==========================================================

def train_dispersion():

    wave = CONFIG["WAVENUMBER"]


    # ------------------------------------------------------
    # Wavenumber values
    # ------------------------------------------------------

    k_vals = torch.linspace(

        wave["k_min"],

        wave["k_max"],

        wave["num_k"]

    )


    # ------------------------------------------------------
    # Build networks once
    # ------------------------------------------------------

    model_layer, model_half = get_all_networks()

    model_layer.to(DEVICE)
    model_half.to(DEVICE)


    # ------------------------------------------------------
    # Initial phase velocity
    # ------------------------------------------------------

    params_layer = CONFIG["LAYER"]
    params_half = CONFIG["SUBSTRATE"]


    cs1 = (

        params_layer["Ge_0"]
        /
        params_layer["rho_0"]

    ) ** 0.5


    # ------------------------------------------------------
    # IMPORTANT
    #
    # Do NOT constrain c between the two shear velocities.
    #
    # Your analytical solution has c/cs1 > 1.
    # ------------------------------------------------------

    c_initial = 1.30 * cs1


    c = torch.nn.Parameter(

        torch.tensor(

            c_initial,

            device=DEVICE,

            dtype=torch.float32

        )

    )


    dispersion = []


    # ======================================================
    # Sweep over k
    # ======================================================

    for idx, k in enumerate(k_vals):

        k_value = k.item()


        print(
            "\n"
            + "=" * 60
        )

        print(

            f"Training "
            f"k = {k_value:.4f} "
            f"({idx + 1}/{len(k_vals)})"

        )

        print(
            "=" * 60
        )


        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        c_value = train_for_single_k(

            k_value,

            model_layer,
            model_half,

            c,

            n_epochs=5000
            if idx == 0
            else 3000,

            n_domain=1000,

            n_bc=200,

            n_int=200,

            n_far=200,

            lr=1e-3

        )


        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        dispersion.append(

            [
                k_value,
                c_value
            ]

        )


    return dispersion


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print(
        "\nRunning "
        "viscoelastic Love-wave PINN solver..."
    )


    # ======================================================
    # Build models
    # ======================================================

    model_layer, model_half = get_all_networks()

    model_layer.to(DEVICE)
    model_half.to(DEVICE)


    # ======================================================
    # Initial c
    # ======================================================

    params_layer = CONFIG["LAYER"]
    params_half = CONFIG["SUBSTRATE"]


    cs1 = (

        params_layer["Ge_0"]
        /
        params_layer["rho_0"]

    ) ** 0.5


    c_initial = 1.30 * cs1


    c = torch.nn.Parameter(

        torch.tensor(

            c_initial,

            device=DEVICE,

            dtype=torch.float32

        )

    )


    print(
        f"\nReference shear velocity "
        f"cs1 = {cs1:.3f} m/s"
    )

    print(
        f"Initial phase velocity "
        f"c = {c_initial:.3f} m/s"
    )

    print(
        f"Initial c/cs1 = "
        f"{c_initial / cs1:.4f}"
    )


    # ======================================================
    # TEST ONE WAVENUMBER FIRST
    # ======================================================

    test_k = 0.4


    test_c = train_for_single_k(

        test_k,

        model_layer,

        model_half,

        c,

        n_epochs=3000,

        n_domain=1000,

        n_bc=200,

        n_int=200,

        n_far=200,

        lr=1e-3

    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        f"TEST RESULT:"
    )

    print(
        f"k = {test_k:.4f}"
    )

    print(
        f"c = {test_c:.6f} m/s"
    )

    print(
        f"c/cs1 = {test_c / cs1:.6f}"
    )

    print(
        "=" * 60
    )


    # ======================================================
    # FULL DISPERSION
    # ======================================================

    # Uncomment only after the single-k test works.
    #
    # dispersion = train_dispersion()
    #
    # torch.save(
    #     dispersion,
    #     "dispersion_curve.pt"
    # )
    #
    # print(
    #     "\nSaved: dispersion_curve.pt"
    # )