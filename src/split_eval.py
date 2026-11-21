import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIR = PROJECT_ROOT / "results" / "hidden_states"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LABELS = ["reversible", "partially_reversible", "irreversible"]
SPLITS = ["split_1", "split_2", "split_3"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])

def evaluate_splits(model_name, npz_filename, best_layer):
    print(f"\n==================== {model_name} ====================")
    
    # 1. Load Data
    master_df = load_jsonl(PROCESSED_DIR / "reversibility_dataset_master.jsonl")
    data = np.load(HIDDEN_DIR / npz_filename, allow_pickle=True)
    
    hidden_states = data["last_token"][:, best_layer, :]
    npz_ids = data["ids"]

    # Align hidden states with master IDs
    id_to_idx = {str(id_val): i for i, id_val in enumerate(npz_ids)}
    valid_indices = [id_to_idx[str(id_val)] for id_val in master_df["id"] if str(id_val) in id_to_idx]
    valid_master = master_df[master_df["id"].astype(str).isin(id_to_idx.keys())].reset_index(drop=True)
    
    X = hidden_states[valid_indices]
    y = valid_master["label"].map({l: i for i, l in enumerate(LABELS)}).values

    kappas = []
    accuracies = []

    # 2. Loop through the 3 splits
    for split in SPLITS:
        train_mask = valid_master[split] == "train"
        test_mask = valid_master[split] == "test"
        
        # Train probe
        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X[train_mask], y[train_mask])
        
        # Test probe
        preds = probe.predict(X[test_mask])
        
        # Calculate metrics
        kappa = cohen_kappa_score(y[test_mask], preds, labels=[0, 1, 2])
        acc = accuracy_score(y[test_mask], preds)
        
        kappas.append(kappa)
        accuracies.append(acc)
        
        print(f"{split}: Kappa = {kappa:.4f} | Accuracy = {acc:.4f}")

    # 3. Calculate Mean and Standard Deviation
    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas, ddof=1)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)

    print("-" * 40)
    print(f"Kappa across splits:    {mean_kappa:.4f} ± {std_kappa:.4f}")
    print(f"Accuracy across splits: {mean_acc:.4f} ± {std_acc:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_splits(
        model_name="Llama-3.1-8B-Instruct", 
        npz_filename="Llama-3.1-8B-Instruct_hidden_states.npz", 
        best_layer=31
    )
    evaluate_splits(
        model_name="Qwen2.5-7B-Instruct", 
        npz_filename="Qwen2.5-7B-Instruct_hidden_states.npz", 
        best_layer=20
    )