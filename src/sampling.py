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
# Domain sampling
# --------------------------------------------------
def sample_domain_points(n_domain, geom):
    """
    Returns:
        z_layer : points in layer [-h1, 0]
        z_half  : points in half-space [0, H_trunc]
    """

    h1 = geom.get("h1", 2.0)
    H  = geom.get("H_trunc", 30.0)

    # Layer: z ∈ [-h1, 0]
    z_layer = sample_uniform(n_domain, -h1, 0.0)

    # Half-space: z ∈ [0, H]
    z_half = sample_uniform(n_domain, 0.0, H)

    return z_layer.to(DEVICE), z_half.to(DEVICE)


# --------------------------------------------------
# Top surface (z = -h1)
# --------------------------------------------------
def sample_top_surface(n_boundary, geom):
    """
    Top free surface at z = -h1
    """

    h1 = geom.get("h1", 20.0)

    z_top = torch.full(
        (n_boundary, 1),
        -float(h1),
        dtype=torch.float32,
        device=DEVICE
    )

    return z_top


# --------------------------------------------------
# Boundary (z = 0)
# --------------------------------------------------
def sample_boundary(n_boundary):
    """
    Boundary between layer and half-space
    """

    z_int = torch.zeros(
        (n_boundary, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    return z_int


# --------------------------------------------------
# Far-field (z = H_trunc)
# --------------------------------------------------
def sample_far_field(n_far, geom):
    """
    Far-field boundary for half-space
    """

    H = geom.get("H_trunc", 30.0)

    z_far = torch.full(
        (n_far, 1),
        float(H),
        dtype=torch.float32,
        device=DEVICE
    )

    return z_far