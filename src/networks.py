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
        return torch.atan(x)


# ==================================================
# ACTIVATION SELECTOR
# ==================================================

def get_activation(name):
    name = name.lower()

    if name == "tanh":
        return nn.Tanh()

    elif name == "sigmoid":
        return nn.Sigmoid()

    elif name == "relu":
        return nn.ReLU()

    elif name == "gelu":
        return nn.GELU()

    elif name == "silu":
        return nn.SiLU()

    elif name == "softplus":
        return nn.Softplus()

    elif name == "sin":
        return Sin()

    elif name == "arctan":
        return Arctan()

    else:
        raise ValueError(f"Unknown activation function: {name}")


# ==================================================
# GENERIC PINN NETWORK
# ==================================================

class PINN(nn.Module):
    """
    Fully-connected PINN for each layer.

    Input  : [x, k]
    Output : [U_r, U_i, Phi_r, Phi_i]
    """

    def __init__(
        self,
        in_dim=2,
        out_dim=4,
        width=64,
        depth=3,
        activation="tanh"
    ):
        super().__init__()

        layers = []

        # --------------------------------------------------
        # If activation is a list, cycle through activations
        # Example: ["sin", "tanh", "gelu"]
        # --------------------------------------------------
        if isinstance(activation, list):
            act_cycle = cycle(activation)
        else:
            act_cycle = cycle([activation])

        # --------------------------------------------------
        # Input layer
        # --------------------------------------------------
        layers.append(nn.Linear(in_dim, width))
        layers.append(get_activation(next(act_cycle)))

        # --------------------------------------------------
        # Hidden layers
        # --------------------------------------------------
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(get_activation(next(act_cycle)))

        # --------------------------------------------------
        # Output layer
        # --------------------------------------------------
        layers.append(nn.Linear(width, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==================================================
# NETWORK FACTORY FOR YOUR THREE-DOMAIN PROBLEM
# ==================================================

def get_all_networks(
    width=128,
    depth=5,
    activation="tanh"
):
    """
    Returns PINN models for:

    Layer 1 : Piezo-viscoelastic upper layer
    Layer 2 : Piezo-viscoelastic lower layer
    Layer 3 : Air / vacuum layer

    Input  : [x, k]
    Output : [U_r, U_i, Phi_r, Phi_i]
    """

    model_L1 = PINN(
        in_dim=2,
        out_dim=4,
        width=width,
        depth=depth,
        activation=activation
    )

    model_L2 = PINN(
        in_dim=2,
        out_dim=4,
        width=width,
        depth=depth,
        activation=activation
    )

    model_L3 = PINN(
        in_dim=2,
        out_dim=4,
        width=width,
        depth=depth,
        activation=activation
    )

    return model_L1, model_L2, model_L3