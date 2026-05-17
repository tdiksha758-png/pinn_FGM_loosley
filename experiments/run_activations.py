import os
import sys
import time
import argparse
import csv

# Ensure project root is importable so `src` can be imported as a package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
# import src as package so relative imports inside src/ work
import importlib
src = importlib.import_module("src")
networks = importlib.import_module("src.networks")
train = importlib.import_module("src.train")
CONFIG = importlib.import_module("src.config").CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ACTIVATIONS = [
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    "silu",
    "softplus",
    "sin",
    "arctan",
]


def run_for_activation(act, test_k=0.6, epochs=200, save_model=True, out_dir="."):
    print(f"\n=== Running activation: {act} ===\n")
    CONFIG["ACTIVATION"] = act

    # build networks using explicit activation
    model_layer, model_half = networks.get_all_networks(activation=act)
    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # compute initial c robustly (support SUBSTRATE or HALFSPACE keys)
    params_layer = CONFIG.get("LAYER")
    params_half = CONFIG.get("HALFSPACE", CONFIG.get("SUBSTRATE"))

    c_init = 0.5 * ((params_layer["mu_0"]/params_layer["rho_0"])**0.5 +
                    (params_half["mu_0"]/params_half["rho_0"])**0.5)

    c = torch.nn.Parameter(torch.tensor(c_init, device=DEVICE, dtype=torch.float32))

    start = time.time()
    best_c = train.train_for_single_k(test_k, model_layer, model_half, c,
                                      n_epochs=epochs, n_domain=200, n_bc=50, n_int=50, n_far=50, lr=1e-3)
    duration = time.time() - start

    result = {
        "activation": act,
        "test_k": test_k,
        "c": float(best_c),
        "duration_s": float(duration),
    }

    # save result row
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "activation_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["activation", "test_k", "c", "duration_s"])
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    # save model states
    if save_model:
        torch.save(model_layer.state_dict(), os.path.join(out_dir, f"model_layer_{act}.pt"))
        torch.save(model_half.state_dict(), os.path.join(out_dir, f"model_half_{act}.pt"))

    print(f"Done {act}: c={best_c:.6f} (took {duration:.1f}s)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run short PINN runs across activation functions and save outputs.")
    parser.add_argument("--activations", nargs="*", default=DEFAULT_ACTIVATIONS,
                        help="List of activations to test")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per activation (quick default)")
    parser.add_argument("--out", default="experiments_output", help="Output folder under project root")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false", help="Do not save model weights")

    args = parser.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    for act in args.activations:
        try:
            run_for_activation(act, test_k=0.6, epochs=args.epochs, save_model=args.save_model, out_dir=out_dir)
        except Exception as e:
            print(f"Error running activation {act}: {e}")

    print("\nAll done. Results saved to:", out_dir)
