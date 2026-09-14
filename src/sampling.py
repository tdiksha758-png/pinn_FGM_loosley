import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Utility: uniform sampling
# --------------------------------------------------
def sample_uniform(n, low, high):
    return low + (high - low) * torch.rand(
        n, 1,
        device=DEVICE,
        dtype=torch.float32
    )


# --------------------------------------------------
# Domain sampling
# --------------------------------------------------
def sample_domain_points(n_domain, geom):
    """
    Returns:

        z_layer : points in upper layer [-h1, 0]

        z_half  : points in lower half-space
                  [0, H_trunc]
    """

    h1 = geom["h1"]
    H_trunc = geom["H_trunc"]

    # Upper piezomagnetic layer:
    # z ∈ [-h1, 0]
    z_layer = sample_uniform(
        n_domain,
        -h1,
        0.0
    )

    # Lower piezomagnetic half-space:
    # z ∈ [0, H_trunc]
    z_half = sample_uniform(
        n_domain,
        0.0,
        H_trunc
    )

    return (
        z_layer.to(DEVICE),
        z_half.to(DEVICE)
    )


# --------------------------------------------------
# Top surface boundary: z = -h1
# --------------------------------------------------
def sample_top_surface(n_boundary, geom):
    """
    Top free surface of the upper piezomagnetic layer:

        z = -h1
    """

    h1 = geom["h1"]

    z_top = torch.full(
        (n_boundary, 1),
        -float(h1),
        dtype=torch.float32,
        device=DEVICE
    )

    return z_top


# --------------------------------------------------
# Interface boundary: z = 0
# --------------------------------------------------
def sample_interface(n_interface):
    """
    Interface between the upper layer and
    lower half-space:

        z = 0
    """

    z_int = torch.zeros(
        (n_interface, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    return z_int


# --------------------------------------------------
# Far-field boundary: z = H_trunc
# --------------------------------------------------
def sample_far_field(n_far, geom):
    """
    Truncated far-field boundary of the lower
    piezomagnetic half-space:

        z = H_trunc
    """

    H_trunc = geom["H_trunc"]

    z_far = torch.full(
        (n_far, 1),
        float(H_trunc),
        dtype=torch.float32,
        device=DEVICE
    )

    return z_far