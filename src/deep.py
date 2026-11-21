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
# Load the ScienceWorld pairs file which contains the horizon labels
pairs_df = load_jsonl(PROCESSED_DIR / "scienceworld_pairs.jsonl")
data = np.load(HIDDEN_DIR / "Llama-3.1-8B-Instruct_hidden_states.npz", allow_pickle=True)
npz_ids = data["ids"]

# Align hidden states with the scienceworld pairs
id_to_idx = {str(id_val): i for i, id_val in enumerate(npz_ids)}
valid_indices = [id_to_idx[str(id_val)] for id_val in pairs_df["id"] if str(id_val) in id_to_idx]
valid_df = pairs_df[pairs_df["id"].astype(str).isin(id_to_idx.keys())].reset_index(drop=True)

X = data["last_token"][valid_indices, 31, :]

# We use the split column if it exists, otherwise assume standard train/test split
if "split_1" in valid_df.columns:
    train_mask = valid_df["split_1"] == "train"
    test_mask = valid_df["split_1"] == "test"
else:
    # Fallback: 80/20 random split if no pre-defined splits
    np.random.seed(42)
    indices = np.random.permutation(len(valid_df))
    split_idx = int(0.8 * len(valid_df))
    train_mask = np.zeros(len(valid_df), dtype=bool)
    test_mask = np.zeros(len(valid_df), dtype=bool)
    train_mask[indices[:split_idx]] = True
    test_mask[indices[split_idx:]] = True
    print("No 'split_1' column found. Using random 80/20 split.")

# Loop through the horizons
for horizon in ["label_h1", "label_h2", "label_h3"]:
    if horizon not in valid_df.columns:
        print(f"Skipping {horizon}: Column not found.")
        continue

    print(f"\n--- Reversibility Horizon: {horizon} ---")
    
    # Map the labels to integers
    y = valid_df[horizon].map({l: i for i, l in enumerate(LABELS)})
    
    # Check for missing labels in the column
    if y.isna().any():
        print(f"Warning: Found unknown labels in {horizon}. Dropping them.")
        valid_rows = y.notna()
        y = y[valid_rows]
        X_clean = X[valid_rows.values]
        train_clean = train_mask[valid_rows.values]
        test_clean = test_mask[valid_rows.values]
    else:
        X_clean = X
        train_clean = train_mask
        test_clean = test_mask
        y_clean = y
        
    # Train probe
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_clean[train_clean], y_clean[train_clean])
    
    # Test probe
    preds = probe.predict(X_clean[test_clean])
    
    acc = accuracy_score(y_clean[test_clean], preds)
    kappa = cohen_kappa_score(y_clean[test_clean], preds, labels=[0, 1, 2])
    
    print(f"Train size: {train_clean.sum()} | Test size: {test_clean.sum()}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Kappa:    {kappa:.4f}")