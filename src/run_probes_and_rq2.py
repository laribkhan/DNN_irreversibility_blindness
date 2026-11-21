import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, cohen_kappa_score

PROJECT_ROOT = Path(".")
MASTER_JSONL = PROJECT_ROOT / "data" / "processed" / "reversibility_dataset_master.jsonl"
HS_DIR = PROJECT_ROOT / "results" / "hidden_states"
PROBES_DIR = PROJECT_ROOT / "results" / "probes"
PROBES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    with MASTER_JSONL.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows


def bootstrap_ci_balanced_acc(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    if not scores:
        val = balanced_accuracy_score(y_true, y_pred)
        return val, val, val
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def lexical_baseline_score(rows, split_col="split_1", minimal_only=True):
    sub = [r for r in rows if r["label"] in ("reversible", "irreversible")]
    if minimal_only:
        sub = [r for r in sub if r.get("is_minimal_pair", False)]
    train_rows = [r for r in sub if r[split_col] == "train"]
    test_rows = [r for r in sub if r[split_col] == "test"]

    vec = CountVectorizer(ngram_range=(1, 2), lowercase=True)
    X_train = vec.fit_transform([r["action_text"] for r in train_rows])
    X_test = vec.transform([r["action_text"] for r in test_rows])
    y_train = np.array([1 if r["label"] == "irreversible" else 0 for r in train_rows])
    y_test = np.array([1 if r["label"] == "irreversible" else 0 for r in test_rows])

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return float(balanced_accuracy_score(y_test, preds))


def hewitt_liang_verb_control(X, rows, layer_idx, train_mask, test_mask, seed=42):
    """Hewitt & Liang (2019) style control task: assign each action verb a fixed random label."""
    rng = np.random.default_rng(seed)
    verbs = sorted({r["action_text"].split()[0].strip('{"tool": ') for r in rows})
    verb_to_ctrl = {v: int(rng.integers(0, 2)) for v in verbs}
    y_ctrl = np.array([verb_to_ctrl.get(r["action_text"].split()[0].strip('{"tool": '), 0) for r in rows])

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X[train_mask, layer_idx, :], y_ctrl[train_mask])
    preds = clf.predict(X[test_mask, layer_idx, :])
    return float(balanced_accuracy_score(y_ctrl[test_mask], preds))


def evaluate_model(model_short: str, rows: list):
    npz_path = HS_DIR / f"{model_short}_hidden_states.npz"
    if not npz_path.exists():
        print(f"[SKIP] Missing hidden states for {model_short}: {npz_path}")
        return None

    data = np.load(npz_path)
    X_last = data["last_token"].astype(np.float32)
    X_mean = data["mean_action"].astype(np.float32)
    num_layers = X_last.shape[1]

    # Binary task mask: reversible (0) vs irreversible (1)
    bin_mask = np.array([r["label"] in ("reversible", "irreversible") for r in rows])
    y_bin_all = np.array([1 if r["label"] == "irreversible" else 0 for r in rows])

    # Majority class baseline on binary test set (split_1)
    s1_test = np.array([r["split_1"] == "test" for r in rows]) & bin_mask
    s1_train = np.array([r["split_1"] == "train" for r in rows]) & bin_mask
    maj_class = int(np.bincount(y_bin_all[s1_train]).argmax())
    maj_acc = float(balanced_accuracy_score(y_bin_all[s1_test], np.full(s1_test.sum(), maj_class)))

    lex_min = lexical_baseline_score(rows, split_col="split_1", minimal_only=True)
    lex_all = lexical_baseline_score(rows, split_col="split_1", minimal_only=False)

    layer_records = []
    best_layer = 0
    best_bal_acc = -1.0

    for l in range(num_layers):
        split_bal_accs = []
        split_f1s = []
        split_aurocs = []
        split_selectivities = []
        split_shuffled = []
        mean_tok_accs = []

        for split_col, seed in [("split_1", 42), ("split_2", 100), ("split_3", 2026)]:
            tr = np.array([r[split_col] == "train" for r in rows]) & bin_mask
            te = np.array([r[split_col] == "test" for r in rows]) & bin_mask

            # Main linear probe (last action token)
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            clf.fit(X_last[tr, l, :], y_bin_all[tr])
            preds = clf.predict(X_last[te, l, :])
            probs = clf.predict_proba(X_last[te, l, :])[:, 1]

            bal = balanced_accuracy_score(y_bin_all[te], preds)
            f1 = f1_score(y_bin_all[te], preds, zero_division=0)
            auroc = roc_auc_score(y_bin_all[te], probs) if len(np.unique(y_bin_all[te])) > 1 else 0.5

            # Shuffled label baseline
            rng = np.random.default_rng(seed)
            y_shuf_tr = rng.permutation(y_bin_all[tr])
            clf_shuf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            clf_shuf.fit(X_last[tr, l, :], y_shuf_tr)
            shuf_acc = balanced_accuracy_score(y_bin_all[te], clf_shuf.predict(X_last[te, l, :]))

            # Hewitt & Liang control task & selectivity
            ctrl_acc = hewitt_liang_verb_control(X_last, rows, l, tr, te, seed=seed)
            selectivity = bal - ctrl_acc

            # Ablation 1: mean over action tokens
            clf_mean = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            clf_mean.fit(X_mean[tr, l, :], y_bin_all[tr])
            mean_acc = balanced_accuracy_score(y_bin_all[te], clf_mean.predict(X_mean[te, l, :]))

            split_bal_accs.append(bal)
            split_f1s.append(f1)
            split_aurocs.append(auroc)
            split_selectivities.append(selectivity)
            split_shuffled.append(shuf_acc)
            mean_tok_accs.append(mean_acc)

        avg_bal = float(np.mean(split_bal_accs))
        if avg_bal > best_bal_acc:
            best_bal_acc = avg_bal
            best_layer = l

        # 95% bootstrap CI on split_1 for this layer
        clf_s1 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_s1.fit(X_last[s1_train, l, :], y_bin_all[s1_train])
        preds_s1 = clf_s1.predict(X_last[s1_test, l, :])
        _, ci_low, ci_high = bootstrap_ci_balanced_acc(y_bin_all[s1_test], preds_s1, n_boot=1000)

        layer_records.append({
            "model": model_short,
            "layer": l,
            "bal_acc_mean": avg_bal,
            "bal_acc_std": float(np.std(split_bal_accs)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "macro_f1_mean": float(np.mean(split_f1s)),
            "auroc_mean": float(np.mean(split_aurocs)),
            "selectivity_mean": float(np.mean(split_selectivities)),
            "shuffled_acc_mean": float(np.mean(split_shuffled)),
            "mean_action_tok_acc": float(np.mean(mean_tok_accs)),
            "majority_baseline_s1": maj_acc,
            "lexical_min_pairs_s1": lex_min,
            "lexical_all_s1": lex_all,
        })

    # Compulsory Ablation #3: Cross-environment transfer at best_layer (split_1)
    sb_tr = np.array([r["environment"] == "tool_sandbox" and r["split_1"] == "train" for r in rows]) & bin_mask
    sw_te = np.array([r["environment"] == "scienceworld" and r["split_1"] == "test" for r in rows]) & bin_mask
    sw_tr = np.array([r["environment"] == "scienceworld" and r["split_1"] == "train" for r in rows]) & bin_mask
    sb_te = np.array([r["environment"] == "tool_sandbox" and r["split_1"] == "test" for r in rows]) & bin_mask

    clf_sb2sw = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_last[sb_tr, best_layer, :], y_bin_all[sb_tr])
    acc_sb2sw = float(balanced_accuracy_score(y_bin_all[sw_te], clf_sb2sw.predict(X_last[sw_te, best_layer, :])))

    clf_sw2sb = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_last[sw_tr, best_layer, :], y_bin_all[sw_tr])
    acc_sw2sb = float(balanced_accuracy_score(y_bin_all[sb_te], clf_sw2sb.predict(X_last[sb_te, best_layer, :])))

    # RQ2 Stated Answers evaluation
    rq2_path = PROBES_DIR / f"{model_short}_rq2_stated.jsonl"
    rq2_summary = {}
    if rq2_path.exists():
        with rq2_path.open("r", encoding="utf-8") as f:
            rq2_rows = [json.loads(line) for line in f]
        y_true_3 = [r["true_label"] for r in rq2_rows]
        y_stat_3 = [r["stated_label"] for r in rq2_rows]
        stated_bal = float(balanced_accuracy_score(y_true_3, y_stat_3))

        # Probe binary predictions on split_1 test set at best_layer
        clf_best = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_last[s1_train, best_layer, :], y_bin_all[s1_train])
        probe_bin_test = clf_best.predict(X_last[s1_test, best_layer, :])
        stated_bin_test = np.array([1 if rq2_rows[i]["stated_label"] == "irreversible" else 0 for i in range(len(rows)) if s1_test[i]])
        kappa_probe_stated = float(cohen_kappa_score(stated_bin_test, probe_bin_test))

        # Fraction of paired minimal-pair scenarios where model correctly identifies irreversible version (K)
        irrev_stated_correct = [
            1 if r["stated_label"] == "irreversible" else 0
            for r in rq2_rows if r["true_label"] == "irreversible"
        ]
        k_score = float(np.mean(irrev_stated_correct)) if irrev_stated_correct else 0.0
        rq2_summary = {
            "stated_3class_balanced_acc": stated_bal,
            "kappa_stated_vs_probe_s1_test": kappa_probe_stated,
            "K_fraction_irreversible_correctly_stated": k_score,
        }

    df_layers = pd.DataFrame(layer_records)
    df_layers.to_csv(PROBES_DIR / f"{model_short}_layer_probes.csv", index=False)

    summary = {
        "model": model_short,
        "best_layer": int(best_layer),
        "best_layer_bal_acc": best_bal_acc,
        "layer_0_bal_acc": float(df_layers.loc[df_layers["layer"] == 0, "bal_acc_mean"].iloc[0]),
        "majority_baseline_s1": maj_acc,
        "lexical_minimal_pairs_s1": lex_min,
        "lexical_all_data_s1": lex_all,
        "cross_env_sandbox_to_scienceworld": acc_sb2sw,
        "cross_env_scienceworld_to_sandbox": acc_sw2sb,
        **rq2_summary,
    }
    (PROBES_DIR / f"{model_short}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df_layers, summary


def main():
    rows = load_dataset()
    models = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct"]
    dfs = []
    summaries = []

    for m in models:
        res = evaluate_model(m, rows)
        if res is not None:
            df_m, sum_m = res
            dfs.append(df_m)
            summaries.append(sum_m)
            print(f"\n=== {m} Summary ===")
            print(json.dumps(sum_m, indent=2))

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        plt.figure(figsize=(9, 5))
        for m in all_df["model"].unique():
            sub = all_df[all_df["model"] == m]
            plt.plot(sub["layer"], sub["bal_acc_mean"], marker="o", markersize=3, label=f"{m} (Probe - Last Token)")
            plt.plot(sub["layer"], sub["shuffled_acc_mean"], linestyle="--", alpha=0.6, label=f"{m} (Shuffled Control)")

        ref = all_df.iloc[0]
        plt.axhline(ref["lexical_min_pairs_s1"], color="red", linestyle=":", label=f"Lexical Baseline (Minimal Pairs: {ref['lexical_min_pairs_s1']:.2f})")
        plt.axhline(0.50, color="gray", linestyle="-.", alpha=0.5, label="Chance (0.50)")
        plt.title("Linear Probe Balanced Accuracy Across Layers (Binary Reversibility)")
        plt.xlabel("Layer (0 = Embedding)")
        plt.ylabel("Balanced Accuracy (3-Split Mean)")
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plot_path = PROBES_DIR / "probe_layers_comparison.png"
        plt.savefig(plot_path, dpi=300)
        print(f"\nSaved probe comparison plot to {plot_path}")


if __name__ == "__main__":
    main()