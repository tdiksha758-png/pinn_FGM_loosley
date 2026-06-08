import torch
import torch.nn as nn


# ==================================================
# GENERIC PINN NETWORK
# ==================================================

class PINN(nn.Module):
    """
    Fully-connected PINN with Tanh activation
    """

    def __init__(

        self,

        in_dim,
        out_dim,

        width=64,
        depth=8

    ):

        super().__init__()

        layers = []

        # Input layer

        layers.append(
            nn.Linear(in_dim, width)
        )

        layers.append(
            nn.Tanh()
        )

        # Hidden layers

        for _ in range(depth - 1):

            layers.append(
                nn.Linear(width, width)
            )

            layers.append(
                nn.Tanh()
            )

        # Output layer

        layers.append(
            nn.Linear(width, out_dim)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)


# ==================================================
# NETWORK FACTORY
# ==================================================

def get_all_networks(

    width=64,

    depth_layer=8,

    depth_half=8,

    depth_air=4

):

    """
    Returns:
        net_air
        net_layer
        net_halfspace
    """

    # Air/Vacuum network

    net_air = PINN(

        in_dim=1,

        out_dim=2,

        width=width,

        depth=depth_air
    )

    # Upper layer network

    net_layer = PINN(

        in_dim=1,

        out_dim=4,

        width=width,

        depth=depth_layer
    )

    # Half-space network

    net_halfspace = PINN(

        in_dim=1,

        out_dim=4,

        width=width,

        depth=depth_half
    )

    return (

        net_air,

        net_layer,

        net_halfspace
    )