import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = PROJECT_ROOT / "results" / "probes"
HIDDEN_DIR = PROJECT_ROOT / "results" / "hidden_states"

LABELS = ["reversible", "partially_reversible", "irreversible"]
MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]

def generate_predictions(model_name):
    print(f"\nGenerating probe predictions for {model_name}...")
    
    # 1. Load the hidden states
    npz_path = HIDDEN_DIR / f"{model_name}_hidden_states.npz"
    data = np.load(npz_path, allow_pickle=True)
    
    # 2. Get best layer from summary
    summary_path = PROBES_DIR / f"{model_name}_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    # Determine best layer and whether to use last_token or mean_action
    # Fallbacks just in case the JSON uses different keys
    best_layer = int(summary.get("best_layer", summary.get("best_layer_idx", 0)))
    best_feat = summary.get("best_feature", "last_token") 
    
    print(f"  -> Best layer: {best_layer} | Feature type: {best_feat}")
    
    # 3. Extract the hidden states for that specific layer
    # The array might be 3D: (num_examples, num_layers, hidden_dim)
    features = data[best_feat]
    
    # Check dimensions to slice the correct layer
    if len(features.shape) == 3:
        hidden_states = features[:, best_layer, :]
    else:
        # If it's already 2D, assume it's the best layer
        hidden_states = features
        
    print(f"  -> Extracted hidden states shape: {hidden_states.shape}")
    
    # 4. Load stated data and align by IDs
    stated_path = PROBES_DIR / f"{model_name}_rq2_stated.jsonl"
    with open(stated_path, "r", encoding="utf-8") as f:
        stated_data = [json.loads(line) for line in f if line.strip()]
    df_stated = pd.DataFrame(stated_data)
    
    # Create a mapping from ID to row index in the hidden_states array
    # because we must ensure the hidden states align with the IDs
    npz_ids = data["ids"]
    id_to_idx = {str(id_val): i for i, id_val in enumerate(npz_ids)}
    
    # Filter df_stated to only those present in npz_ids, and get their indices
    valid_rows = []
    valid_indices = []
    for i, row in df_stated.iterrows():
        row_id = str(row["id"])
        if row_id in id_to_idx:
            valid_rows.append(row)
            valid_indices.append(id_to_idx[row_id])
            
    df_valid = pd.DataFrame(valid_rows)
    aligned_hidden_states = hidden_states[valid_indices]
    
    # 5. Train a quick probe
    y = df_valid["true_label"].map({l: i for i, l in enumerate(LABELS)})
    
    print("  -> Training LogisticRegression probe...")
    probe = LogisticRegression(max_iter=1000, C=1.0) # Removed multi_class='multinomial' for compatibility
    probe.fit(aligned_hidden_states, y)
    
    # 6. Predict
    print("  -> Generating predictions...")
    preds = probe.predict(aligned_hidden_states)
    
    # 7. Save to JSONL
    out_path = PROBES_DIR / f"{model_name}_probe_preds.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in df_valid.iterrows():
            record = {
                "id": row["id"],
                "probe_label": LABELS[preds[i]]
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"  -> Successfully saved {len(preds)} predictions to {out_path.name}")

if __name__ == "__main__":
    for model in MODELS:
        generate_predictions(model)