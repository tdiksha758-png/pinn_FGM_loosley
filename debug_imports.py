import sys, os, traceback

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
print("ROOT:", ROOT)

try:
    from src.networks import get_all_networks
    from src.config import CONFIG
    from src.sampling import sample_domain_points, sample_top_surface, sample_interface, sample_far_field
    from src.losses import total_loss
    from src.pde_residual import residual_layer_FGM, residual_halfspace_FGM
    from src.boundary_conditions import top_surface_bc, interface_layer_halfspace, halfspace_far_field_bc
    print("All imports succeeded")
except Exception as e:
    print("Import failed:", e)
    traceback.print_exc()
