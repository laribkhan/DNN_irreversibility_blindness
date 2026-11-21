import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.utils import shuffle

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

# Use the best layer (31 for Llama)
hidden_states = data["last_token"][:, 31, :]
npz_ids = data["ids"]

# Align hidden states with labels
id_to_idx = {str(id_val): i for i, id_val in enumerate(npz_ids)}
valid_indices = [id_to_idx[str(id_val)] for id_val in master_df["id"] if str(id_val) in id_to_idx]
valid_master = master_df[master_df["id"].astype(str).isin(id_to_idx.keys())].reset_index(drop=True)

X = hidden_states[valid_indices]
y_true = valid_master["label"].map({l: i for i, l in enumerate(LABELS)}).values

train_mask = valid_master["split_1"] == "train"
test_mask = valid_master["split_1"] == "test"

# --- THE SHUFFLE MAGIC ---
# We shuffle the labels ONLY within the training set, 
# so the test set remains intact to evaluate against.
X_train = X[train_mask]
y_train = y_true[train_mask]
X_test = X[test_mask]
y_test = y_true[test_mask]

# Shuffle the training labels randomly (using a fixed seed for reproducibility)
y_train_shuffled = shuffle(y_train, random_state=42)

print("Training probe on SHUFFLED labels...")
probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X_train, y_train_shuffled)

# Predict on the UNTOUCHED test set
preds = probe.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, preds)
kappa = cohen_kappa_score(y_test, preds, labels=[0, 1, 2])

print("\n--- Shuffled-Label Probe Results (Test Set) ---")
print(f"Raw Accuracy: {acc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print("\n(Note: Accuracy should be ~33.3% and Kappa should be ~0.0. If it is high, you have a data leak!)")