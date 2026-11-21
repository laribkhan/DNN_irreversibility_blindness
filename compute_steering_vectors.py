import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(".")
MASTER_JSONL = PROJECT_ROOT / "data" / "processed" / "reversibility_dataset_master.jsonl"
HS_DIR = PROJECT_ROOT / "results" / "hidden_states"
OUT_JSON = PROJECT_ROOT / "results" / "probes" / "steering_vectors.json"

# Best layers identified from your Step 3 linear probe runs
BEST_LAYERS = {
    "Qwen2.5-7B-Instruct": 20,
    "Llama-3.1-8B-Instruct": 31,
}


def main():
    with MASTER_JSONL.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    # Use training split (split_1 == 'train') and binary irreversible vs reversible pairs
    is_train = np.array([r["split_1"] == "train" for r in rows])
    irrev_mask = is_train & np.array([r["label"] == "irreversible" for r in rows])
    rev_mask = is_train & np.array([r["label"] == "reversible" for r in rows])

    rng = np.random.default_rng(42)
    payload = {}

    for model_short, layer_idx in BEST_LAYERS.items():
        npz_path = HS_DIR / f"{model_short}_hidden_states.npz"
        if not npz_path.exists():
            print(f"[SKIP] Missing {npz_path}")
            continue

        data = np.load(npz_path)
        X_last = data["last_token"].astype(np.float32)

        mu_irrev = X_last[irrev_mask, layer_idx, :].mean(axis=0)
        mu_rev = X_last[rev_mask, layer_idx, :].mean(axis=0)

        steer_vec = mu_irrev - mu_rev
        vec_norm = float(np.linalg.norm(steer_vec))

        # Compulsory Section 7.2 control: random direction of the EXACT same L2 norm
        rand_raw = rng.normal(size=steer_vec.shape).astype(np.float32)
        rand_norm = float(np.linalg.norm(rand_raw))
        rand_vec = rand_raw * (vec_norm / (rand_norm + 1e-12))

        payload[model_short] = {
            "layer": int(layer_idx),
            "l2_norm": vec_norm,
            "steering_vector": steer_vec.tolist(),
            "random_control_vector": rand_vec.tolist(),
        }
        print(
            f"{model_short} | Layer {layer_idx} | "
            f"n_irrev_train={irrev_mask.sum()}, n_rev_train={rev_mask.sum()} | "
            f"L2 norm={vec_norm:.4f}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved steering vectors to {OUT_JSON}!")


if __name__ == "__main__":
    main()