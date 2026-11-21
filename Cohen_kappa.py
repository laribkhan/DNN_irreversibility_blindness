import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROBES_DIR = PROJECT_ROOT / "results" / "probes"

LABELS = ["reversible", "partially_reversible", "irreversible"]
SPLITS = ["split_1", "split_2", "split_3"]

def load_jsonl(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)

def find_file(filename: str) -> Path:
    search_dirs = [PROCESSED_DIR, PROJECT_ROOT, PROJECT_ROOT / "results", PROBES_DIR]
    for d in search_dirs:
        p = d / filename
        if p.exists(): return p
    matches = list(PROJECT_ROOT.rglob(filename))
    if matches: return matches[0]
    raise FileNotFoundError(f"Could not find {filename}")

def load_probe_predictions(probe_path: Path) -> pd.DataFrame:
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")
    df = load_jsonl(probe_path)
    if "probe_label" not in df.columns:
        for alt in ["pred_label", "prediction", "predicted_label", "label"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "probe_label"})
                break
    return df[["id", "probe_label"]]

def attach_probe_predictions(master_df, stated_df, probe_df):
    merged = (
        master_df[["id", "scenario_id", "environment", "label", *SPLITS]]
        .merge(stated_df[["id", "stated_label", "true_label"]], on="id", how="inner")
        .merge(probe_df[["id", "probe_label"]], on="id", how="inner")
    )
    return merged

def bootstrap_kappa_ci(stated, probe, weights="linear", n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    s_arr, p_arr = stated.to_numpy(), probe.to_numpy()
    n = len(s_arr)
    scores = [
        cohen_kappa_score(s_arr[idx], p_arr[idx], labels=LABELS, weights=weights)
        for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))
    ]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)

def evaluate_metric(df: pd.DataFrame, target_col: str, weights="linear"):
    records = []
    subsets = [("overall", df)] + [(col, df[df[col] == "test"]) for col in SPLITS]

    for name, sub in subsets:
        target = sub[target_col]
        probe = sub["probe_label"]
        
        kappa = cohen_kappa_score(target, probe, labels=LABELS, weights=weights)
        agree = accuracy_score(target, probe)
        _, lo, hi = bootstrap_kappa_ci(target, probe, weights=weights)

        records.append({
            "subset": name,
            "n": len(sub),
            "raw_agreement": round(float(agree), 4),
            f"kappa_{weights}": round(float(kappa), 4),
            "ci_95_low": round(lo, 4),
            "ci_95_high": round(hi, 4),
        })
    return pd.DataFrame(records)

def run_for_model(model_name, stated_filename, probe_filename, master_df):
    print(f"\n==================== {model_name} ====================")
    stated_df = load_jsonl(find_file(stated_filename))
    probe_df = load_probe_predictions(find_file(probe_filename))

    eval_df = attach_probe_predictions(master_df, stated_df, probe_df)

    print("\n--- 1. Probe vs TRUE Label (Does the probe know the truth?) ---")
    true_summary = evaluate_metric(eval_df, "true_label")
    print(true_summary.to_string(index=False))
    print("\nConfusion matrix (rows = true_label, cols = probe_label):")
    print(pd.DataFrame(
        confusion_matrix(eval_df["true_label"], eval_df["probe_label"], labels=LABELS),
        index=[f"true_{l}" for l in LABELS],
        columns=[f"probe_{l}" for l in LABELS],
    ))

    print("\n--- 2. Probe vs STATED Label (Does the text match the probe?) ---")
    stated_summary = evaluate_metric(eval_df, "stated_label")
    print(stated_summary.to_string(index=False))
    print("\nConfusion matrix (rows = stated_label, cols = probe_label):")
    print(pd.DataFrame(
        confusion_matrix(eval_df["stated_label"], eval_df["probe_label"], labels=LABELS),
        index=[f"stated_{l}" for l in LABELS],
        columns=[f"probe_{l}" for l in LABELS],
    ))

if __name__ == "__main__":
    master_df = load_jsonl(find_file("reversibility_dataset_master.jsonl"))

    run_for_model(
        model_name="Llama-3.1-8B-Instruct",
        stated_filename="Llama-3.1-8B-Instruct_rq2_stated.jsonl",
        probe_filename="Llama-3.1-8B-Instruct_probe_preds.jsonl",
        master_df=master_df,
    )
    run_for_model(
        model_name="Qwen2.5-7B-Instruct",
        stated_filename="Qwen2.5-7B-Instruct_rq2_stated.jsonl",
        probe_filename="Qwen2.5-7B-Instruct_probe_preds.jsonl",
        master_df=master_df,
    )