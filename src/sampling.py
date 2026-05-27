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
# Domain sampling (3 regions)
# --------------------------------------------------
def sample_domain_points(n_domain, domain):
    """
    Returns:
        x_L1 : points in layer1   [domain["LAYER1"][0], domain["LAYER1"][1]]
        x_L2 : points in layer2   [domain["LAYER2"][0], domain["LAYER2"][1]]
        x_L3 : points in air      [domain["AIR"][0], domain["AIR"][1]]
    """

    # Unpack ranges
    xL1_min, xL1_max = domain["LAYER1"]
    xL2_min, xL2_max = domain["LAYER2"]
    xL3_min, xL3_max = domain["AIR"]

    # Sample
    x_L1 = sample_uniform(n_domain, xL1_min, xL1_max)
    x_L2 = sample_uniform(n_domain, xL2_min, xL2_max)
    x_L3 = sample_uniform(n_domain, xL3_min, xL3_max)

    return x_L1, x_L2, x_L3


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


# --------------------------------------------------
# Air far-field (x = -h1 - h3)
# --------------------------------------------------
# Generic far-field sampler (wrapper)
def sample_far_field(n_far, geom):
    """
    Far-field in air at x = -h1 - h3
    """

    h1 = geom["h1"]
    h3 = geom["h3"]

    x_far = torch.full(
        (n_far, 1),
        -(float(h1) + float(h3)),
        
        dtype=torch.float32,
        device=DEVICE
    )

    return x_far