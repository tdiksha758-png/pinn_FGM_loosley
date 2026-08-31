import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Utility: uniform sampling
# --------------------------------------------------
def sample_uniform(n, low, high):
    return low + (high - low) * torch.rand(
        n, 1, device=DEVICE, dtype=torch.float32
    )


# --------------------------------------------------
# Domain sampling (2 regions — no air layer)
# --------------------------------------------------
def sample_domain_points(n_domain, domain):
    """
    Returns:
        x_L1 : points in layer1   [domain["layer1"][0], domain["layer1"][1]]
        x_L2 : points in layer2   [domain["layer2"][0], domain["layer2"][1]]
    """

    # Unpack ranges
    xL1_min, xL1_max = domain["layer1"]
    xL2_min, xL2_max = domain["layer2"]

    # Sample
    x_L1 = sample_uniform(n_domain, xL1_min, xL1_max)
    x_L2 = sample_uniform(n_domain, xL2_min, xL2_max)

    return x_L1, x_L2


# --------------------------------------------------
# Top surface (x = -h1)
# --------------------------------------------------
def sample_top_surface(n_boundary, geom):
    """
    Top surface of layer1 at x = -h1
    """

    h1 = geom["h1"]

    x_top = torch.full(
        (n_boundary, 1),
        -float(h1),
        dtype=torch.float32,
        device=DEVICE
    )

    return x_top


# --------------------------------------------------
# Interface between Layer1 and Layer2 (x = 0)
# --------------------------------------------------
def sample_interface(n_interface):
    """
    Interface between layer1 and layer2 at x = 0
    """

    x_int = torch.zeros((n_interface, 1), device=DEVICE)

    return x_int


# --------------------------------------------------
# Bottom surface (x = h2)
# --------------------------------------------------
def sample_bottom_surface(n_boundary, geom):
    """
    Bottom of layer2 at x = h2
    """

    h2 = geom["h2"]

    x_bot = torch.full(
        (n_boundary, 1),
        float(h2),
        dtype=torch.float32,
        device=DEVICE
    )

    return x_bot