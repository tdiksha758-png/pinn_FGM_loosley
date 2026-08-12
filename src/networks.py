import torch
import torch.nn as nn


# ==========================================================
# Generic PINN Network
# ==========================================================

class PINN(nn.Module):

    def __init__(self, in_dim, out_dim, width=128, depth=8):

        super().__init__()

        layers = []

        layers.append(
            nn.Linear(in_dim, width)
        )

        layers.append(
            nn.Tanh()
        )

        for _ in range(depth - 1):

            layers.append(
                nn.Linear(width, width)
            )

            layers.append(
                nn.Tanh()
            )

        layers.append(
            nn.Linear(width, out_dim)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)


# ==========================================================
# Network Factory
# ==========================================================

def get_all_networks():

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
        depth=5
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
        depth=5
    )

    return net_layer, net_halfspace