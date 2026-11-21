import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1. Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = PROJECT_ROOT / "results" / "probes"
LOG_DIR = PROJECT_ROOT / "results" / "agent_logs"
FIG_DIR = PROBES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set global style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])

# =========================================================
# FIGURE 1: RQ1 Probe Accuracy vs Baselines (Using your CSV)
# =========================================================
print("Generating Figure 1: Probe Accuracy...")
llama_df = pd.read_csv(PROBES_DIR / "Llama-3.1-8B-Instruct_layer_probes.csv")
qwen_df = pd.read_csv(PROBES_DIR / "Qwen2.5-7B-Instruct_layer_probes.csv")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot real probe lines
ax.plot(llama_df['layer'], llama_df['bal_acc_mean'], color='#1f77b4', linewidth=2.5, label='Llama-3.1-8B (Probe)')
ax.plot(qwen_df['layer'], qwen_df['bal_acc_mean'], color='#2ca02c', linewidth=2.5, label='Qwen2.5-7B (Probe)')

# Plot shuffled control lines
ax.plot(llama_df['layer'], llama_df['shuffled_acc_mean'], color='#1f77b4', linestyle='--', alpha=0.6, label='Llama (Shuffled Control)')
ax.plot(qwen_df['layer'], qwen_df['shuffled_acc_mean'], color='#2ca02c', linestyle='--', alpha=0.6, label='Qwen (Shuffled Control)')

# Add Lexical Baseline and Chance
lexical_baseline = llama_df['lexical_min_pairs_s1'].iloc[0]
ax.axhline(lexical_baseline, color='red', linestyle=':', linewidth=2, label=f'Lexical Baseline ({lexical_baseline:.2f})')
ax.axhline(0.50, color='gray', linestyle='-.', alpha=0.5, label='Chance (0.50)')

ax.set_title("Linear Probe Balanced Accuracy Across Layers (Binary Reversibility)")
ax.set_xlabel("Layer (0 = Embedding)")
ax.set_ylabel("Balanced Accuracy (3-Split Mean)")
ax.set_ylim(0.4, 1.05)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "figure1_probe_accuracy.png", dpi=300)
plt.close()

# =========================================================
# FIGURE 2: RQ3 Agentic Behaviour (Using your Agent Logs)
# =========================================================
print("Generating Figure 2: RQ3 Behavior...")
# Load actual logs
llama_logs = load_jsonl(LOG_DIR / "Llama-3.1-8B-Instruct_agent_logs.jsonl")
qwen_logs = load_jsonl(LOG_DIR / "Qwen2.5-7B-Instruct_agent_logs.jsonl")

# Calculate rates for Irreversible variants
llama_irrev = llama_logs[llama_logs['variant'] == 'irreversible']
qwen_irrev = qwen_logs[qwen_logs['variant'] == 'irreversible']

# Metrics
metrics = {
    'Model': ['Llama-3.1-8B', 'Qwen2.5-7B'],
    'Caution Rate': [llama_irrev['cautious_before_irrev'].mean(), qwen_irrev['cautious_before_irrev'].mean()],
    'Irreversible Action Executed': [llama_irrev['took_irreversible_action'].mean(), qwen_irrev['took_irreversible_action'].mean()]
}
df_rq3 = pd.DataFrame(metrics)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(df_rq3['Model']))
width = 0.35

rects1 = ax.bar(x - width/2, df_rq3['Caution Rate'], width, label='Caution Rate (Before Irreversible)', color='#2ca02c', edgecolor='black')
rects2 = ax.bar(x + width/2, df_rq3['Irreversible Action Executed'], width, label='Irreversible Action Executed', color='#d62728', edgecolor='black')

ax.set_ylabel('Rate')
ax.set_title('RQ3: Irreversibility Blindness in Agentic Behaviour')
ax.set_xticks(x)
ax.set_xticklabels(df_rq3['Model'])
ax.set_ylim(0, 1.2)
ax.legend(loc='upper right')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height*100:.0f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig(FIG_DIR / "figure2_rq3_behavior.png", dpi=300)
plt.close()

# =========================================================
# FIGURE 3: RQ5 Uncertainty Bias (Using Summary JSON)
# =========================================================
print("Generating Figure 3: RQ5 Uncertainty...")
qwen_summary = load_json(PROBES_DIR / "Qwen2.5-7B-Instruct_rq5_summary.json")
llama_summary = load_json(PROBES_DIR / "Llama-3.1-8B-Instruct_rq5_summary (2).json")

# Extract Qwen probabilities for Reversible vs Irreversible states
qwen_rev = qwen_summary['by_variant']['reversible']
qwen_irrev = qwen_summary['by_variant']['irreversible']

classes = ['Reversible', 'Partially Rev.', 'Irreversible']
# Note: The summary JSON only gave us mean_p_irreversible and mean_p_reversible. 
# We calculate the remainder for partially.
qwen_rev_probs = [
    qwen_rev['mean_p_reversible'], 
    1.0 - (qwen_rev['mean_p_reversible'] + qwen_rev['mean_p_irreversible']), 
    qwen_rev['mean_p_irreversible']
]
qwen_irrev_probs = [
    qwen_irrev['mean_p_reversible'], 
    1.0 - (qwen_irrev['mean_p_reversible'] + qwen_irrev['mean_p_irreversible']), 
    qwen_irrev['mean_p_irreversible']
]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects3 = ax.bar(x - width/2, qwen_rev_probs, width, label='True State: Reversible', color='#1f77b4', edgecolor='black')
rects4 = ax.bar(x + width/2, qwen_irrev_probs, width, label='True State: Irreversible', color='#ff7f0e', edgecolor='black')

ax.set_ylabel('Predicted Probability')
ax.set_title('RQ5: Qwen Prediction Bias (Irreversibility Blindness)')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.1)
ax.legend()

def autolabel_float(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height*100:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel_float(rects3)
autolabel_float(rects4)

plt.tight_layout()
plt.savefig(FIG_DIR / "figure3_rq5_uncertainty.png", dpi=300)
plt.close()

print(f"\nAll 3 figures saved to: {FIG_DIR}")