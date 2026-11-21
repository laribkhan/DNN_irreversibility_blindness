import json
import csv
import random
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

def finalize_all():
    sandbox_path = Path("data/processed/sandbox_pairs.jsonl")
    sw_path = Path("data/processed/scienceworld_pairs.jsonl")

    all_rows = []
    for p in [sandbox_path, sw_path]:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                all_rows.append(json.loads(line))

    # Assign scenario-grouped 80/20 splits for 3 random seeds (Section 7.4)
    scenarios = [r["scenario_id"] for r in all_rows]
    for split_idx, seed in enumerate([42, 100, 2026], start=1):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, test_idx = next(gss.split(all_rows, groups=scenarios))
        test_set = set(test_idx)
        for i, row in enumerate(all_rows):
            row[f"split_{split_idx}"] = "test" if i in test_set else "train"

    # Save master deliverable dataset (Section 4.3 & Section 8)
    master_path = Path("data/processed/reversibility_dataset_master.jsonl")
    with master_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    # Export 100 random blind samples for Human Quality Control & Cohen's Kappa (Section 4.2)
    random.seed(42)
    qc_sample = random.sample(all_rows, 100)
    qc_blind_csv = Path("data/processed/human_qc_100_blind.csv")
    qc_key_json = Path("data/processed/human_qc_100_ground_truth_key.json")

    with qc_blind_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "environment", "state_text", "action_text", "annotator_1_label", "annotator_2_label"])
        for r in qc_sample:
            writer.writerow([r["id"], r["environment"], r["state_text"], r["action_text"], "", ""])

    with qc_key_json.open("w", encoding="utf-8") as f:
        json.dump({r["id"]: r["label"] for r in qc_sample}, f, indent=2)

    counts = {}
    env_counts = {}
    for r in all_rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
        env_counts[r["environment"]] = env_counts.get(r["environment"], 0) + 1

    print(f"\nSUCCESS! Master dataset saved to: {master_path}")
    print(f"Total pairs: {len(all_rows)} (Target >= 2,000)")
    print(f"By Environment: {env_counts}")
    print(f"By Class: {counts}")
    print(f"Split 1 Train/Test counts: {sum(1 for r in all_rows if r['split_1']=='train')} train / {sum(1 for r in all_rows if r['split_1']=='test')} test")
    print(f"Exported 100 blind QC pairs to: {qc_blind_csv}")

if __name__ == "__main__":
    finalize_all()