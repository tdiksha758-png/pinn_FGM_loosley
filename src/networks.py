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

        "silu": nn.SiLU(),       # Swish

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


def resolve_activation_list(activation, depth):
    """
    Accepts either a single activation name or a list/tuple of names and
    returns a list of activation modules of length `depth`.
    """
    # Single name -> repeat for all layers
    if isinstance(activation, (str,)):
        return [get_activation(activation) for _ in range(depth)]

    # If it's an iterable of names, convert to list and broadcast/crop
    if isinstance(activation, (list, tuple)):
        names = list(activation)
        if len(names) == depth:
            return [get_activation(n) for n in names]
        if len(names) < depth:
            # repeat the sequence to match depth
            c = cycle(names)
            return [get_activation(next(c)) for _ in range(depth)]
        # more provided than needed -> truncate
        return [get_activation(n) for n in names[:depth]]

    # Fallback: treat as single activation
    return [get_activation(str(activation)) for _ in range(depth)]


# ==================================================
# GENERIC PINN NETWORK
# ==================================================

class PINN(nn.Module):
    """
    Fully-connected neural network for PINN
    """

    def __init__(

        self,
        in_dim,
        out_dim,
        width=30,
        depth=4,
        activation="tanh"

    ):

        super().__init__()

        layers = []

        # resolve activation modules for each layer (length == depth)
        act_modules = resolve_activation_list(activation, depth)

        # ------------------------------------------
        # Input layer
        # ------------------------------------------
        layers.append(nn.Linear(in_dim, width))
        layers.append(act_modules[0])

        # ------------------------------------------
        # Hidden layers
        # ------------------------------------------
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act_modules[i + 1])

        # ------------------------------------------
        # Output layer
        # ------------------------------------------

        layers.append(
            nn.Linear(width, out_dim)
        )

        # ------------------------------------------
        # Build sequential model
        # ------------------------------------------

        self.model = nn.Sequential(*layers)

    # ------------------------------------------------

    def forward(self, x):

        return self.model(x)


# ==================================================
# NETWORK FACTORY
# ==================================================

def get_all_networks(

    width=50,
    depth=8,
    activation="tanh"

):
    """
    Returns PINN models for:
    - Functionally graded layer
    - Functionally graded half-space
    """

    # ----------------------------------------------
    # Layer network
    # ----------------------------------------------

    net_layer = PINN(

        in_dim=1,
        out_dim=1,

        width=width,
        depth=depth,

        activation=activation

    )

    # ----------------------------------------------
    # Half-space network
    # ----------------------------------------------

    net_halfspace = PINN(

        in_dim=1,
        out_dim=1,

        width=width,
        depth=depth,

        activation=activation

    )

    return net_layer, net_halfspace