import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "results" / "hidden_states"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Labels for 3-class
LABELS_3 = ["reversible", "partially_reversible", "irreversible"]
# Labels for Binary (Safe vs Unsafe)
# Reversible = Safe (0), Partially/Irreversible = Unsafe (1)
BINARY_MAP = {
    "reversible": 0,
    "partially_reversible": 1,
    "irreversible": 1
}

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
train_mask = valid_master["split_1"] == "train"
test_mask = valid_master["split_1"] == "test"

# ==========================================
# 1. Three-Class Evaluation
# ==========================================
print("\n--- 1. Three-Class Probe ---")
y_3class = valid_master["label"].map({l: i for i, l in enumerate(LABELS_3)}).values

probe_3 = LogisticRegression(max_iter=1000, C=1.0)
probe_3.fit(X[train_mask], y_3class[train_mask])
preds_3 = probe_3.predict(X[test_mask])

acc_3 = accuracy_score(y_3class[test_mask], preds_3)
kappa_3 = cohen_kappa_score(y_3class[test_mask], preds_3, labels=[0, 1, 2])

print(f"Accuracy: {acc_3:.4f}")
print(f"Kappa:    {kappa_3:.4f}")

# ==========================================
# 2. Binary Evaluation
# ==========================================
print("\n--- 2. Binary Probe (Safe vs Unsafe) ---")
y_binary = valid_master["label"].map(BINARY_MAP).values

probe_binary = LogisticRegression(max_iter=1000, C=1.0)
probe_binary.fit(X[train_mask], y_binary[train_mask])
preds_binary = probe_binary.predict(X[test_mask])

acc_binary = accuracy_score(y_binary[test_mask], preds_binary)
kappa_binary = cohen_kappa_score(y_binary[test_mask], preds_binary, labels=[0, 1])

print(f"Accuracy: {acc_binary:.4f}")
print(f"Kappa:    {kappa_binary:.4f}")

print("\n--- Comparison ---")
print(f"Accuracy Improvement (Binary - 3-Class): {(acc_binary - acc_3)*100:.2f}%")