import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

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
test_mask = valid_master["split_1"] == "test"

probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X[train_mask], y[train_mask])
preds = probe.predict(X[test_mask])
y_test = y[test_mask]

# ==========================================
# 1. Bootstrap Confidence Intervals (1,000 resamples)
# ==========================================
print("\n--- 1. Bootstrap Confidence Intervals (1,000 resamples) ---")
n_boot = 1000
accuracies = []
rng = np.random.default_rng(42)

for _ in range(n_boot):
    idx = rng.integers(0, len(y_test), size=len(y_test))
    acc = accuracy_score(y_test[idx], preds[idx])
    accuracies.append(acc)

lower = np.percentile(accuracies, 2.5)
upper = np.percentile(accuracies, 97.5)
print(f"Probe Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

# ==========================================
# 2. McNemar's Test (Reversible vs Irreversible)
# ==========================================
print("\n--- 2. McNemar's Test ---")
# We test if the probe makes significantly more errors on irreversible vs reversible
rev_mask = valid_master[test_mask]["label"] == "reversible"
irrev_mask = valid_master[test_mask]["label"] == "irreversible"

# Get correct/incorrect booleans
correct = (preds == y_test)
rev_correct = correct[rev_mask.values]
irrev_correct = correct[irrev_mask.values]

# To use McNemar's, we need a paired setup. 
# Let's match the number of reversible and irreversible examples
min_len = min(len(rev_correct), len(irrev_correct))
rev_correct = rev_correct[:min_len]
irrev_correct = irrev_correct[:min_len]

# Create contingency table
# [Both correct, Rev correct Irrev wrong]
# [Irrev correct Rev wrong, Both wrong]
both_correct = np.sum(rev_correct & irrev_correct)
rev_only_correct = np.sum(rev_correct & ~irrev_correct)
irrev_only_correct = np.sum(~rev_correct & irrev_correct)
both_wrong = np.sum(~rev_correct & ~irrev_correct)

table = [[both_correct, rev_only_correct], 
         [irrev_only_correct, both_wrong]]

print(f"Contingency Table:\n{pd.DataFrame(table, index=['Rev Correct', 'Rev Wrong'], columns=['Irrev Correct', 'Irrev Wrong'])}")
result = mcnemar(table, exact=True)
print(f"McNemar's Test p-value: {result.pvalue:.4f}")
if result.pvalue < 0.05:
    print("Conclusion: Significant difference in error rates between reversible and irreversible.")
else:
    print("Conclusion: No significant difference in error rates (Probe is equally good at both).")