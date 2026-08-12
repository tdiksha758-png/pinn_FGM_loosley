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
        z_layer   : points in layer domain [-H, 0] (non-dimensional)
        z_half    : points in half-space [0, L] (non-dimensional)
    """

    H = geom["L"]              # Layer thickness
    H_trunc = geom["H_trunc"]  # Truncated half-space
    z_layer = sample_uniform(
        n_domain,
        -H,
        0.0
    )

    z_half = sample_uniform(
        n_domain,
        0.0,
        H_trunc
    )
    return (
        z_layer.to(DEVICE),
        z_half.to(DEVICE),
    )


# --------------------------------------------------
# Top surface boundary (z = -H)
# --------------------------------------------------
def sample_top_surface(n_boundary, geom):
    """
    Top free surface at z = -H (non-dimensional)
    """

    H = geom["L"]

    z_top = torch.full(
        (n_boundary, 1),
        -float(H),                # force float value
        dtype=torch.float32,      # force float dtype
        device=DEVICE
    )

    return z_top



# --------------------------------------------------
# Interface boundary (z = 0)
# --------------------------------------------------
def sample_interface(n_interface):
    """
    Interface between layer and half-space at z = 0 (non-dimensional)
    """

    z_int = torch.zeros((n_interface, 1))

    return z_int.to(DEVICE)


# --------------------------------------------------
# Far-field boundary (z = L)
# --------------------------------------------------
def sample_far_field(n_far, geom):
    """
    Far-field boundary for half-space at z = L (non-dimensional)
    """

    L = geom.get("L", 29.0*6)

    H_trunc = geom["H_trunc"]

    z_far = torch.full(

        (n_far,1),

        H_trunc,

        dtype=torch.float32,

        device=DEVICE

    )

    return z_far.to(DEVICE)
