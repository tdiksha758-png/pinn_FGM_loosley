import torch
import torch.nn as nn


# ==================================================
# GENERIC PINN NETWORK
# ==================================================
class PINN(nn.Module):
    """
    Fully-connected PINN

    Input:
        z

    Outputs depend on medium
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

        # ==================================================
        # Input layer
        # ==================================================
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        # ==================================================
        # Hidden layers
        # ==================================================
        for _ in range(depth - 1):

            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        # ==================================================
        # Output layer
        # ==================================================
        layers.append(nn.Linear(width, out_dim))

        self.model = nn.Sequential(*layers)

    # ==================================================
    # Forward pass
    # ==================================================
    def forward(self, x):

        return self.model(x)


# ==================================================
# NETWORK FACTORY
# ==================================================
def get_all_networks():

    """
    Returns:
        net_air
        net_layer
        net_halfspace
    """

    # ==================================================
    # AIR / VACUUM NETWORK
    # ==================================================
    #
    # Outputs:
    #
    # [Phi_air_r,
    #  Phi_air_i]
    #
    # D3_air is computed using autograd:
    #
    # D3_air = -eps0 * dPhi_air/dz
    #
    # ==================================================

    net_air = PINN(

        in_dim=1,

        out_dim=2,

        width=64,

        depth=4
    )

    # ==================================================
    # UPPER PIEZO-VISCOELASTIC LAYER
    # ==================================================
    #
    # Outputs:
    #
    # [U_r,
    #  U_i,
    #  Phi_r,
    #  Phi_i]
    #
    # ==================================================

    net_layer = PINN(

        in_dim=1,

        out_dim=4,

        width=64,

        depth=8
    )

    # ==================================================
    # LOWER HALF-SPACE
    # ==================================================
    #
    # Outputs:
    #
    # [U_r,
    #  U_i,
    #  Phi_r,
    #  Phi_i]
    #
    # ==================================================

    net_halfspace = PINN(

        in_dim=1,

        out_dim=4,

        width=64,

        depth=8
    )

    return (
        net_air,
        net_layer,
        net_halfspace
    )