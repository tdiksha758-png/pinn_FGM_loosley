import torch
import torch.nn as nn


# --------------------------------------------------
# Generic PINN network
# --------------------------------------------------
class PINN(nn.Module):
    """
    Fully-connected neural network for multi-field PINN
    Outputs: [U_r, U_i, Phi_r, Phi_i]
    """

    def __init__(self, in_dim, out_dim, width=64, depth=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Network factory
# --------------------------------------------------
def get_all_networks(width=128, depth=5):
    """
    Returns PINN models for:
    - Layer 1 (piezo-viscoelastic)
    - Layer 2 (piezo-viscoelastic)
    - Air layer (electrostatic)
    
    ✅ INCREASED CAPACITY for better convergence
    """

    net_L1 = PINN(
        in_dim=2,
        out_dim=4,
        width=width,
        depth=depth
    )
    net_L2 = PINN(
        in_dim=2,
        out_dim=4,
        width=width,
        depth=depth
    )

    # 🔹 Air layer
    net_L3 = PINN(
        in_dim=2,
        out_dim=4,
        width=width,
        depth=depth
    )

    return net_L1, net_L2, net_L3