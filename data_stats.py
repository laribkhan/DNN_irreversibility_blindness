import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_FILE = PROJECT_ROOT / "data" / "processed" / "reversibility_dataset_master.jsonl"

with open(MASTER_FILE, "r", encoding="utf-8") as f:
    df = pd.DataFrame([json.loads(line) for line in f if line.strip()])

print("--- RQ1 Dataset Statistics ---")
print(f"Total Scenarios: {len(df)}")

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nEnvironment Distribution:")
if "environment" in df.columns:
    print(df["environment"].value_counts())

print("\nSplit Sizes (Test Counts):")
for split in ["split_1", "split_2", "split_3"]:
    if split in df.columns:
        test_count = (df[split] == "test").sum()
        print(f"{split}: {test_count} test examples")