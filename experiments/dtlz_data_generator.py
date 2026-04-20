from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.configurations.model_config_dtlz import ModelConfigurationDTLZ
from models.model_wrapper_dtlz import ModelWrapperDTLZ


# =========================
# Configuration
# =========================

DTLZ_PROBLEMS = {
    "dtlz1": 5000,
    "dtlz2": 5000,
    "dtlz3": 8000,
    "dtlz4": 5000,
    "dtlz5": 6000,
    "dtlz6": 6000,
    "dtlz7": 7000,
}

N_VARS = 12
N_OBJS = 3
SEED   = 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "dtlz_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Generator
# =========================

def generate_dtlz_csv(problem_name: str, n_rows: int):
    print(f"[DTLZ] Generating {problem_name} ({n_rows} rows)")

    # Model config + wrapper (single source of truth)
    model_config = ModelConfigurationDTLZ(
        problem=problem_name,
        n_vars=N_VARS,
        n_objs=N_OBJS,
        seed=SEED,
    )
    wrapper = ModelWrapperDTLZ(model_config)

    rng = np.random.default_rng(SEED)

    # Sample decision space
    X = rng.random((n_rows, N_VARS))

    # Evaluate objectives directly via pymoo problem
    F = wrapper.problem.evaluate(X, return_values_of=["F"])
    F = np.asarray(F)

    # Build BL-compatible table
    data = {}

    # X columns (numeric → capitalized)
    for j in range(N_VARS):
        data[f"X{j}"] = X[:, j]

    # Y columns (multi-objective, minimized → '-')
    for k in range(N_OBJS):
        data[f"F{k+1}-"] = F[:, k]

    df = pd.DataFrame(data)

    out_path = OUT_DIR / f"{problem_name}_v{N_VARS}_m{N_OBJS}.csv"
    df.to_csv(out_path, index=False)

    print(f"  ✔ wrote {out_path}  shape={df.shape}")


# =========================
# Main
# =========================

if __name__ == "__main__":
    for problem, rows in DTLZ_PROBLEMS.items():
        generate_dtlz_csv(problem, rows)

    print("\nAll DTLZ CSV tables generated successfully.")
