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

# Identify environments
envs = valid_master["environment"].astype(str).str.lower().unique()
print(f"Available environments: {envs}")

# We will try to identify the Sandbox and ScienceWorld columns
sandbox_mask = valid_master["environment"].astype(str).str.contains("sandbox", case=False, na=False)
sci_mask = valid_master["environment"].astype(str).str.contains("sci", case=False, na=False)

# If your environments are named differently (e.g., "files" and "science"), adjust the .str.contains() above.

def train_and_test(train_mask, test_mask, direction_name):
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"Skipping {direction_name}: Not enough data.")
        return
        
    print(f"\n--- Cross-Environment Transfer: {direction_name} ---")
    
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X[train_mask], y[train_mask])
    preds = probe.predict(X[test_mask])
    
    acc = accuracy_score(y[test_mask], preds)
    kappa = cohen_kappa_score(y[test_mask], preds, labels=[0, 1, 2])
    
    print(f"Train size: {train_mask.sum()} | Test size: {test_mask.sum()}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Kappa:    {kappa:.4f}")

# 1. Train on Sandbox, Test on ScienceWorld
train_and_test(sandbox_mask, sci_mask, "Sandbox → ScienceWorld")

# 2. Train on ScienceWorld, Test on Sandbox
train_and_test(sci_mask, sandbox_mask, "ScienceWorld → Sandbox")