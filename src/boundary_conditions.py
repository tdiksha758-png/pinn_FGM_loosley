import torch
from .utils import gradients

# --------------------------------------------------
# Top surface boundary condition (z = -H_layer)
# Stress-free: tau_23 = 0
# --------------------------------------------------
def top_surface_bc(model_layer, z_top, material_params):
    """
    Stress-free top surface:
    tau_23 = mu44 * dV/dz = 0
    Single real field
    """

    z_top = z_top.clone().detach().requires_grad_(True)

    V = model_layer(z_top)
    V_z = gradients(V, z_top)

    # Linear FGM: mu44 = mu44_0 * (1 + alpha1 * z)
    mu44_0 = material_params["mu_0"]
    alpha1 = material_params["alpha"]
    P1 = material_params.get("P_0", 0.0)

    mu44 = mu44_0 * (1 + alpha1 * z_top)
    tau =  V_z 
    return tau


# # --------------------------------------------------
# # Interface between layer and half-space (z = 0)
# # --------------------------------------------------
# def interface_layer_halfspace(model_layer, model_half, z_int,
#                               params_layer, params_half):
#     """
#     Interface conditions:
#     - Displacement continuity: V_layer = V_half
#     - Stress continuity: mu44_layer * dV_layer/dz = mu44_half * dV_half/dz
#     """

#     z_int = z_int.clone().detach().requires_grad_(True)

#     V_layer = model_layer(z_int)
#     V_half  = model_half(z_int)

#     # Derivatives
#     V_layer_z = gradients(V_layer, z_int)
#     V_half_z  = gradients(V_half, z_int)

#     # Graded shear moduli
#     mu44_l0 = params_layer["mu_0"]
#     alpha1 = params_layer["alpha"]
#     P1 = params_layer.get("P_0", 0.0)
#     mu44_l = mu44_l0 * (1 + alpha1 * z_int)

#     mu44_h0 = params_half["mu_0"]
#     alpha2 = params_half["alpha"]
#     P2 = params_half.get("P_0", 0.0)
#     mu44_h = mu44_h0 * (1 + alpha2 * z_int)**2

#     # Residuals: displacement and stress continuity
#     res_disp = V_layer - V_half
#     res_stress = (mu44_l * V_layer_z  - (mu44_h * V_half_z)) / mu44_l0

#     return  res_stress

def imperfect_interface_bc(model_layer, model_half, z_int, params_layer, params_half):
    """
    Imperfect interface condition:
    tau_23 = K * (V_half - V_layer)
    """

     
    z_int = z_int.clone().detach().requires_grad_(True)

    V_layer = model_layer(z_int)
    V_half  = model_half(z_int)

    # Derivatives for shear stress
    V_layer_z = gradients(V_layer, z_int)
    V_half_z  = gradients(V_half, z_int)
    
    # Shear moduli
    mu44_l = params_layer["mu_0"] * (1 + params_layer["alpha"] * z_int)
    mu44_h = params_half["mu_0"] * (1 + params_half["alpha"] * z_int)**2
    K= params_layer["mu_0"]/(params_layer["s"]*params_layer["L"]);
    mu44_l0=params_layer["mu_0"]
    # Interfacial stiffness
    # K = params_layer.get("K", 1e3)

    # Residual: tau_23 - K*(V_half - V_layer)
    res_interface =((mu44_l * V_layer_z) - K * (V_half - V_layer))/mu44_l
    res_stress = (mu44_l * V_layer_z  - (mu44_h * V_half_z)) / mu44_l0

    return res_interface, res_stress
# --------------------------------------------------
# Far-field boundary condition (z -> infinity)
# --------------------------------------------------
def halfspace_far_field_bc(model_half, z_far):
    """
    Half-space decay condition: V -> 0 as z -> infinity
    """
    z_far = z_far.clone().detach().requires_grad_(True)
    V_far = model_half(z_far)

    return V_far