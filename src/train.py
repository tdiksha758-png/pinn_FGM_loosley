import torch
import torch.optim as optim

from .networks import get_all_networks
from .config import CONFIG

from .sampling import (
    sample_domain_points,
    sample_top_surface,
    sample_interface,
    sample_far_field
)

from .losses import total_loss


# ==================================================
# DEVICE
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# TRAIN FOR SINGLE WAVENUMBER
# ==================================================
def train_for_single_k(
    k,
    model_layer,
    model_half,
    c,
    params_L,
    params_H,
    geom,
    loss_history_all,
    n_epochs=5000,
    n_domain=5000,
    n_bc=1000,
    n_int=1000,
    n_far=1000,
    lr=1e-3
):

    print(f"\nTraining for k = {k:.6f} on {DEVICE}")

    # --------------------------------------------------
    # Interface stiffness
    # --------------------------------------------------
    h1 = geom["h1"]

    kappa = params_L["c44_star"] / h1


    # --------------------------------------------------
    # OPTIMIZER
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


    # --------------------------------------------------
    # TRACK BEST SOLUTION
    # --------------------------------------------------
    best_loss = float("inf")

    best_c = c.item()

    k_val = float(k)


    # --------------------------------------------------
    # INITIALIZE LOSS STORAGE
    # --------------------------------------------------
    if k_val not in loss_history_all:

        loss_history_all[k_val] = {

            "total": [],

            "pde": [],

            "air": [],

            "bc": [],

            "interface": [],

            "far": [],

            "amp": []
        }


    # ==================================================
    # TRAINING LOOP
    # ==================================================
    for epoch in range(1, n_epochs + 1):

        # --------------------------------------------------
        # SAMPLE POINTS
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
        # ZERO GRAD
        # --------------------------------------------------
        optimizer.zero_grad()


        # --------------------------------------------------
        # TOTAL LOSS
        # --------------------------------------------------
        loss, logs = total_loss(

            model_layer,
            model_half,

            z_layer,
            z_half,

            z_top,
            z_int,
            z_far,

            params_L,
            params_H,

            k,
            c,

            kappa
        )


        # --------------------------------------------------
        # STORE LOSSES
        # --------------------------------------------------
        loss_history_all[k_val]["total"].append(
            loss.item()
        )

        loss_history_all[k_val]["pde"].append(
            logs["pde"]
        )

        loss_history_all[k_val]["air"].append(
            logs["air"]
        )

        loss_history_all[k_val]["bc"].append(
            logs["bc_top"]
        )

        loss_history_all[k_val]["interface"].append(
            logs["interface"]
        )

        loss_history_all[k_val]["far"].append(
            logs["far"]
        )

        loss_history_all[k_val]["amp"].append(
            logs["amp"]
        )


        # --------------------------------------------------
        # BACKPROPAGATION
        # --------------------------------------------------
        loss.backward()


        # --------------------------------------------------
        # GRADIENT CLIPPING
        # --------------------------------------------------
        torch.nn.utils.clip_grad_norm_(
            list(model_layer.parameters())
            + list(model_half.parameters()),
            max_norm=1.0
        )


        # --------------------------------------------------
        # OPTIMIZER STEP
        # --------------------------------------------------
        optimizer.step()


        # --------------------------------------------------
        # SAVE BEST c
        # --------------------------------------------------
        if loss.item() < best_loss:

            best_loss = loss.item()

            best_c = c.item()


        # --------------------------------------------------
        # PRINT
        # --------------------------------------------------
        if epoch % 500 == 0:

            print(
                f"Epoch {epoch:6d} | "
                f"Loss = {loss.item():.3e} | "
                f"c = {c.item():.6f}"
            )


    return best_c


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    print("\nRunning PINN solver...\n")


    # ==================================================
    # LOAD PARAMETERS
    # ==================================================
    params_L = {

        k_: torch.tensor(
            v,
            device=DEVICE,
            dtype=torch.float32
        )

        for k_, v in CONFIG["LAYER"].items()
    }


    params_H = {

        k_: torch.tensor(
            v,
            device=DEVICE,
            dtype=torch.float32
        )

        for k_, v in CONFIG["HALFSPACE"].items()
    }


    geom = CONFIG["GEOMETRY"]


    # ==================================================
    # BUILD NETWORKS
    # ==================================================
    model_layer, model_half = get_all_networks()

    model_layer.to(DEVICE)

    model_half.to(DEVICE)


    # ==================================================
    # INITIAL PHASE VELOCITY
    # ==================================================
    c = torch.nn.Parameter(

        torch.sqrt(
            params_L["c44_star"]
            /
            params_L["rho"]
        )
    )


    # ==================================================
    # LOSS STORAGE
    # ==================================================
    loss_history_all = {}


    # ==================================================
    # TEST WAVENUMBER
    # ==================================================
    test_k = 0.1


    # ==================================================
    # TRAIN
    # ==================================================
    c_val = train_for_single_k(

        test_k,

        model_layer,
        model_half,

        c,

        params_L,
        params_H,

        geom,

        loss_history_all,

        n_epochs=1500,

        n_domain=500,

        n_bc=100,

        n_int=100,

        n_far=100
    )


    # ==================================================
    # RESULTS
    # ==================================================
    print(f"\n✓ Test result:")

    print(f"c({test_k}) = {c_val:.6f}")


    # ==================================================
    # VERIFY LOSS STORAGE
    # ==================================================
    k0 = list(loss_history_all.keys())[0]

    print(
        "\nStored epochs:",
        len(loss_history_all[k0]["total"])
    )