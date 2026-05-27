import sys, os, traceback

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
print("ROOT:", ROOT)

# Test imports one by one
imports_to_test = [
    ("src.networks", "get_all_networks"),
    ("src.config", "CONFIG"),
    ("src.sampling", "sample_domain_points"),
    ("src.losses", "total_loss"),
]

for module_name, item_name in imports_to_test:
    try:
        module = __import__(module_name, fromlist=[item_name])
        item = getattr(module, item_name)
        print(f"✓ {module_name}.{item_name}")
    except Exception as e:
        print(f"✗ {module_name}.{item_name}: {e}")

# Special check for boundary_conditions
print("\nDetailed check for src.boundary_conditions:")
try:
    import src.boundary_conditions as bc_module
    print(f"  Module imported successfully")
    print(f"  Available attributes: {dir(bc_module)}")
except Exception as e:
    print(f"  ✗ Failed to import: {e}")
    traceback.print_exc()
