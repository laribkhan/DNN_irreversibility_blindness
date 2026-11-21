import json
from pathlib import Path
import pandas as pd
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = PROJECT_ROOT / "results" / "probes"

def analyze_rq2(model_name):
    file_path = PROBES_DIR / f"{model_name}_rq2_stated.jsonl"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    df = pd.DataFrame(data)
    
    print(f"\n--- RQ2 Stated Output Summary for {model_name} ---")
    print(f"Total examples: {len(df)}")
    
    # 1. Distribution of what the model actually said
    print("\n1. Distribution of Stated Labels:")
    print(df["stated_label"].value_counts())
    
    # 2. Accuracy against ground truth
    if "true_label" in df.columns:
        accuracy = (df["stated_label"] == df["true_label"]).mean()
        print(f"\n2. Stated Label Accuracy (vs True Label): {accuracy*100:.2f}%")
        
    # 3. Parse failures or refusals (if any)
    if "parse_error" in df.columns:
        errors = df["parse_error"].notna().sum()
        print(f"\n3. Parse Errors/Refusals: {errors}")

if __name__ == "__main__":
    analyze_rq2("Llama-3.1-8B-Instruct")
    analyze_rq2("Qwen2.5-7B-Instruct")