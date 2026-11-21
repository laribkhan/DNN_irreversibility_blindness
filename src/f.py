import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "results" / "hidden_states"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LABELS = ["reversible", "partially_reversible", "irreversible"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])

print("Loading data...")
master_df = load_jsonl(PROCESSED_DIR / "reversibility_dataset_master.jsonl")

data = np.load(HIDDEN_DIR / "Llama-3.1-8B-Instruct_hidden_states.npz", allow_pickle=True)
npz_ids = data["ids"]

# Align hidden states
id_to_idx = {str(id_val): i for i, id_val in enumerate(npz_ids)}
valid_indices = [id_to_idx[str(id_val)] for id_val in master_df["id"] if str(id_val) in id_to_idx]
valid_master = master_df[master_df["id"].astype(str).isin(id_to_idx.keys())].reset_index(drop=True)

X = data["last_token"][valid_indices, 31, :]
y = valid_master["label"].map({l: i for i, l in enumerate(LABELS)}).values

train_mask = valid_master["split_1"] == "train"
probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X[train_mask], y[train_mask])

# 1. Evaluate on ALL Data (Test Split)
test_mask = valid_master["split_1"] == "test"
X_test_all = X[test_mask]
y_test_all = y[test_mask]
preds_all = probe.predict(X_test_all)

acc_all = accuracy_score(y_test_all, preds_all)
kappa_all = cohen_kappa_score(y_test_all, preds_all, labels=[0, 1, 2])

print("\n--- Evaluation on ALL Data (Test Split) ---")
print(f"Accuracy: {acc_all:.4f}")
print(f"Kappa:    {kappa_all:.4f}")

# 2. Evaluate on MINIMAL PAIRS ONLY
# The minimal pairs are generated from the "sandbox" environment (files, emails, payments).
# Check the unique environments in your master file to confirm.
print(f"\nAvailable environments: {valid_master['environment'].unique()}")

# Filter the test set to only include the "sandbox" environment
# (Change "sandbox" to the exact string printed above if it's different, e.g., "tool_sandbox")
minimal_pair_mask = valid_master["environment"].astype(str).str.contains("sandbox", case=False, na=False)
pair_test_mask = test_mask & minimal_pair_mask

X_test_pairs = X[pair_test_mask]
y_test_pairs = y[pair_test_mask]

if len(y_test_pairs) == 0:
    print("\nError: No samples found for Minimal Pairs. Check the environment name above.")
else:
    preds_pairs = probe.predict(X_test_pairs)
    acc_pairs = accuracy_score(y_test_pairs, preds_pairs)
    kappa_pairs = cohen_kappa_score(y_test_pairs, preds_pairs, labels=[0, 1, 2])

    print("\n--- Evaluation on MINIMAL PAIRS ONLY (Test Split) ---")
    print(f"Number of pairs in test set: {len(y_test_pairs)}")
    print(f"Accuracy: {acc_pairs:.4f}")
    print(f"Kappa:    {kappa_pairs:.4f}")

    print("\n--- Comparison ---")
    print(f"Accuracy Drop: {acc_all - acc_pairs:.4f}")