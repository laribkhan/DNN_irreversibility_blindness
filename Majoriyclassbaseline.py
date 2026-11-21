import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])

print("Loading master dataset...")
master_df = load_jsonl(PROCESSED_DIR / "reversibility_dataset_master.jsonl")

# 1. Find the majority class in the TRAIN split
# (It's important to find the majority class on the training data, not the test data)
train_df = master_df[master_df["split_1"] == "train"]
majority_class = train_df["label"].mode()[0]

print(f"\nMajority Class found in training data: '{majority_class}'")

# 2. Evaluate the baseline on the TEST splits
print("\nEvaluating Dummy Classifier (always predicting majority class):")
print(f"{'Split':<10} | {'Accuracy':<10} | {'Kappa':<10}")
print("-" * 35)

for split in ["split_1", "split_2", "split_3"]:
    test_df = master_df[master_df[split] == "test"]
    
    # The baseline predicts the majority class for EVERY row
    y_true = test_df["label"]
    y_pred = [majority_class] * len(test_df)
    
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"{split:<10} | {acc:.4f}     | {kappa:.4f}")

print("\n--- End of Baseline Results ---")