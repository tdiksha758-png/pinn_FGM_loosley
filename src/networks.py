import torch
import torch.nn as nn
from itertools import cycle


# ==================================================
# CUSTOM ACTIVATIONS
# ==================================================

class Sin(nn.Module):

    def forward(self, x):

        return torch.sin(x)


class Arctan(nn.Module):

    def forward(self, x):

        return torch.arctan(x)


# ==================================================
# ACTIVATION FUNCTION SELECTOR
# ==================================================

def get_activation(name):

    activations = {

        # ------------------------------------------
        # Standard PINN activations
        # ------------------------------------------

        "tanh": nn.Tanh(),

        "sigmoid": nn.Sigmoid(),

        "relu": nn.ReLU(),

        "gelu": nn.GELU(),

        "silu": nn.SiLU(),

        "softplus": nn.Softplus(),

        # ------------------------------------------
        # Custom activations
        # ------------------------------------------

        "sin": Sin(),

        "arctan": Arctan()

    }

    if name not in activations:

        raise ValueError(
            f"Unknown activation function: {name}"
        )

    return activations[name]


# ==================================================
# RESOLVE ACTIVATION LIST
# ==================================================

def resolve_activation_list(
    activation,
    depth
):

    # ------------------------------------------
    # Single activation
    # ------------------------------------------

    if isinstance(activation, str):

        return [

            get_activation(activation)

            for _ in range(depth)
        ]

    # ------------------------------------------
    # Multiple activations
    # ------------------------------------------

    if isinstance(activation, (list, tuple)):

        names = list(activation)

        if len(names) == depth:

            return [
                get_activation(n)
                for n in names
            ]

        if len(names) < depth:

            c = cycle(names)

            return [
                get_activation(next(c))
                for _ in range(depth)
            ]

        return [
            get_activation(n)
            for n in names[:depth]
        ]

    # ------------------------------------------
    # Fallback
    # ------------------------------------------

    return [

        get_activation(str(activation))

        for _ in range(depth)
    ]


# ==================================================
# GENERIC PINN NETWORK
# ==================================================

class PINN(nn.Module):
    """
    Fully-connected PINN
    """

    def __init__(

        self,

        in_dim,
        out_dim,

        width=64,
        depth=8,

        activation="tanh"

    ):

        super().__init__()

        layers = []

        # ==================================================
        # Resolve activations
        # ==================================================

        act_modules = resolve_activation_list(
            activation,
            depth
        )

        # ==================================================
        # Input layer
        # ==================================================

        layers.append(
            nn.Linear(in_dim, width)
        )

        layers.append(
            act_modules[0]
        )

        # ==================================================
        # Hidden layers
        # ==================================================

        for i in range(depth - 1):

            layers.append(
                nn.Linear(width, width)
            )

            layers.append(
                act_modules[i + 1]
            )

        # ==================================================
        # Output layer
        # ==================================================

        layers.append(
            nn.Linear(width, out_dim)
        )

        # ==================================================
        # Sequential model
        # ==================================================

        self.model = nn.Sequential(*layers)

    # ==================================================
    # Forward pass
    # ==================================================

    def forward(self, x):

        return self.model(x)


# ==================================================
# NETWORK FACTORY
# ==================================================

def get_all_networks(

    width=64,

    depth_layer=8,

    depth_half=8,

    depth_air=4,

    activation="tanh"

):

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
    # [Psi_air_r,
    #  Psi_air_i]
    #
    # ==================================================

    net_air = PINN(

        in_dim=1,

        out_dim=2,

        width=width,

        depth=depth_air,

        activation=activation
    )

    # ==================================================
    # UPPER PIEZO-VISCOELASTIC LAYER
    # ==================================================
    #
    # Outputs:
    #
    # [U_r,
    #  U_i,
    #  Psi_r,
    #  Psi_i]
    #
    # ==================================================

    net_layer = PINN(

        in_dim=1,

        out_dim=4,

        width=width,

        depth=depth_layer,

        activation=activation
    )

    # ==================================================
    # LOWER HALF-SPACE
    # ==================================================
    #
    # Outputs:
    #
    # [U_r,
    #  U_i,
    #  Psi_r,
    #  Psi_i]
    #
    # ==================================================

    net_halfspace = PINN(

        in_dim=1,

        out_dim=4,

        width=width,

        depth=depth_half,

        activation=activation
    )

    return (

        net_air,

        net_layer,

        net_halfspace
    )