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

# Align hidden states with labels
id_to_idx = {str(id_val): i for i, id_val in enumerate(npz_ids)}
valid_indices = [id_to_idx[str(id_val)] for id_val in master_df["id"] if str(id_val) in id_to_idx]
valid_master = master_df[master_df["id"].astype(str).isin(id_to_idx.keys())].reset_index(drop=True)

y_true = valid_master["label"].map({l: i for i, l in enumerate(LABELS)}).values
train_mask = valid_master["split_1"] == "train"
test_mask = valid_master["split_1"] == "test"

# ==========================================
# 1. Embedding-Layer Probe (Layer 0)
# ==========================================
print("\n--- 1. Embedding-Layer Probe (Layer 0) ---")
hidden_states_layer0 = data["last_token"][:, 0, :]
X_l0 = hidden_states_layer0[valid_indices]

probe_l0 = LogisticRegression(max_iter=1000, C=1.0)
probe_l0.fit(X_l0[train_mask], y_true[train_mask])
preds_l0 = probe_l0.predict(X_l0[test_mask])

print(f"Layer 0 Accuracy: {accuracy_score(y_true[test_mask], preds_l0):.4f}")
print(f"Layer 0 Kappa:    {cohen_kappa_score(y_true[test_mask], preds_l0):.4f}")

# ==========================================
# 2. Control Task (Hewitt & Liang)
# ==========================================
print("\n--- 2. Control Task (Hewitt & Liang) ---")
if "goal" in valid_master.columns:
    control_labels = valid_master["goal"].str.len().values
    control_labels = pd.qcut(control_labels, q=3, labels=False)
    
    probe_control = LogisticRegression(max_iter=1000, C=1.0)
    probe_control.fit(X_l0[train_mask], control_labels[train_mask])
    preds_control = probe_control.predict(X_l0[test_mask])
    
    print(f"Control Task (Predicting length) Accuracy: {accuracy_score(control_labels[test_mask], preds_control):.4f}")
else:
    print("Skipping: 'goal' text column not found to calculate length.")