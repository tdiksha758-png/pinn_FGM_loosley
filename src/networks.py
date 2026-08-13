import torch
import torch.nn as nn


# ==========================================================
# Generic PINN Network
# ==========================================================

class PINN(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        width=128,
        depth=8,
        activation="tanh"
    ):

        super().__init__()

        layers = []

        # --------------------------------------------------
        # Select activation
        # --------------------------------------------------

        def get_activation():

            if activation == "tanh":
                return nn.Tanh()

            elif activation == "sigmoid":
                return nn.Sigmoid()

            elif activation == "softplus":
                return nn.Softplus()

            elif activation == "relu":
                return nn.ReLU()

            elif activation == "swish":
                return nn.SiLU()

            else:
                raise ValueError(
                    f"Unknown activation: {activation}"
                )

        # --------------------------------------------------
        # Input layer
        # --------------------------------------------------

        layers.append(
            nn.Linear(in_dim, width)
        )

        layers.append(
            get_activation()
        )

        # --------------------------------------------------
        # Hidden layers
        # --------------------------------------------------

        for _ in range(depth - 1):

            layers.append(
                nn.Linear(width, width)
            )

            layers.append(
                get_activation()
            )

        # --------------------------------------------------
        # Output layer
        # --------------------------------------------------

        layers.append(
            nn.Linear(width, out_dim)
        )

        self.model = nn.Sequential(*layers)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    def forward(self, x):

        return self.model(x)


# ==========================================================
# Network Factory
# ==========================================================

def get_all_networks(activation="tanh"):

    # ------------------------------------------------------
    # Layer
    # Output:
    # column 0 -> Real(V)
    # column 1 -> Imag(V)
    # ------------------------------------------------------

    net_layer = PINN(
        in_dim=1,
        out_dim=2,
        width=30,
        depth=5,
        activation=activation
    )

    # ------------------------------------------------------
    # Half-space
    # Output:
    # column 0 -> Real(V)
    # column 1 -> Imag(V)
    # ------------------------------------------------------

    net_halfspace = PINN(
        in_dim=1,
        out_dim=2,
        width=30,
        depth=5,
        activation=activation
    )

    return net_layer, net_halfspace