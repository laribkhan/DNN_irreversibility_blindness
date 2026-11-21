import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "results" / "hidden_states"

LABELS = ["reversible", "partially_reversible", "irreversible"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])

# 1. Load the master dataset
print("Loading master dataset...")
master_df = load_jsonl(PROJECT_ROOT / "data/processed/reversibility_dataset_master.jsonl")

# 2. Load hidden states and align by ID
print("Loading hidden states and aligning...")
data = np.load(HIDDEN_DIR / "Qwen2.5-7B-Instruct_hidden_states.npz", allow_pickle=True)
best_layer = 20
hidden_states = data["last_token"][:, best_layer, :]
npz_ids = data["ids"]

id_to_idx = {str(id_val): i for i, id_val in enumerate(npz_ids)}
valid_indices = [id_to_idx[str(id_val)] for id_val in master_df["id"] if str(id_val) in id_to_idx]
valid_master = master_df[master_df["id"].astype(str).isin(id_to_idx.keys())].reset_index(drop=True)

X = hidden_states[valid_indices]
y = valid_master["label"].map({l: i for i, l in enumerate(LABELS)})

# 3. Train Probe on split_1 train, test on split_1 test
train_mask = valid_master["split_1"] == "train"
test_mask = valid_master["split_1"] == "test"

print("Training probe...")
probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X[train_mask], y[train_mask])

# 4. Generate Predictions
preds = probe.predict(X[test_mask])
test_df = valid_master[test_mask].copy()
test_df["probe_label"] = [LABELS[p] for p in preds]

# 5. Calculate A and K
rev_mask = test_df["label"] == "reversible"
irrev_mask = test_df["label"] == "irreversible"

accuracy_A = accuracy_score(test_df[rev_mask]["label"], test_df[rev_mask]["probe_label"])
accuracy_K = accuracy_score(test_df[irrev_mask]["label"], test_df[irrev_mask]["probe_label"])

print(f"\nAccuracy A (Reversible subset):   {accuracy_A:.4f}")
print(f"Accuracy K (Irreversible subset): {accuracy_K:.4f}")

# 6. Calculate Headline Gap
gap = accuracy_K - accuracy_A
print(f"\n==================================================")
print(f"Headline Number (Irreversibility blindness gap K - A): {gap:.4f}")
print(f"==================================================")

# 7. Calculate RQ5 AUROC
print("\n--- RQ5: AUROC of Prediction Uncertainty ---")
probs = probe.predict_proba(X[test_mask])
max_probs = np.max(probs, axis=1) 
is_correct = (preds == y[test_mask]).astype(int)

auroc = roc_auc_score(is_correct, max_probs)
print(f"AUROC: {auroc:.4f}")
print("(An AUROC > 0.85 means the model is highly confident when it is right, and highly uncertain when it is wrong.)")