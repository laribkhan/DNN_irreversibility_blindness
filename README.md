
 Irreversibility Blindness in LLM Agents

This repository contains the full codebase, dataset generation scripts, probing architectures, and statistical analyses for the paper: *"Irreversibility Blindness in LLM Agents: Measuring Stated Knowledge, Behavioural Sensitivity, Activation Steering, and Uncertainty."*

## Repository Structure
*   `data/processed/`: Contains the generated dataset (`reversibility_dataset_master.jsonl`), task definitions, and the `DATA_CARD.md`.
*   `results/`: Contains the generated hidden states (`.npz`), probe scores (`.csv`, `.json`), agent logs, and figures.
*   `src/`: Contains all Python scripts (detailed below).

## Python Scripts Reference (`src/`)

Here is a breakdown of every Python file in the `src/` directory and how to reproduce the results:

### 1. Environment & Dataset Generation
*   **`sandbox.py`**: The core deterministic Python simulation of the filesystem, email outbox, and wallet. Defines the state transitions and ground-truth reversibility labels.
*   **`generate_sandbox_data.py`**: Generates the 1,332 minimal-pair examples for the tool sandbox.
*   **`generate_scienceworld_data.py`**: Connects to the ScienceWorld API, uses Breadth-First Search (BFS), and generates the 900 physical reasoning examples.
*   **`finalize_dataset_and_splits.py`**: Combines the sandbox and ScienceWorld data into `reversibility_dataset_master.jsonl`, assigns the 3 scenario-grouped train/test splits, and exports the 100-item blind human QC sample.
*   **`run_agent_rq3.py`**: Builds the 60 paired tasks (120 rollouts) and contains the logic for the agent loop and regex parsing.

### 2. Probing & Hidden States (RQ1 & RQ2)
*   **`run_probes_and_rq2.py`**: Extracts hidden states from the models, trains linear probes across all layers, runs the baselines, and generates the layer-by-layer accuracy plots.
*   **`generate_probe_preds.py`**: Loads the `.npz` hidden states, trains the optimal probe, and saves the per-example predictions to `.jsonl`.
*   **`Cohen_kappa.py`**: Cross-references the internal probe predictions, the true labels, and the model's stated text outputs to calculate the confusion matrices and Cohen's Kappa.

### 3. Ablations & Baselines
*   **`Majoriyclassbaseline.py`**: Computes the dummy classifier baseline (always predicting the majority class).
*   **`lexical.py`**: Trains a Bag-of-Words TF-IDF Logistic Regression model on the text to test for surface-level lexical cues.
*   **`shuffled_probe.py`**: Trains a probe on randomly permuted labels to check for data leakage.
*   **`control.py` / `Hewitt.py`**: Runs the Layer 0 (embedding) probe and the Hewitt & Liang control task.
*   **`deep.py`**: Evaluates the probe on the ScienceWorld horizons (h=1, h=2, h=3).
*   **`n.py`**: Tests cross-environment transfer (Sandbox $\to$ ScienceWorld and ScienceWorld $\to$ Sandbox).
*   **`f.py`**: Evaluates the probe on strictly minimal pairs versus all data.
*   **`Binary_vs_three_class_labels.py`**: Tests if collapsing the labels to a binary Safe/Unsafe task improves probe accuracy.

### 4. Causal Steering & Uncertainty (RQ4 & RQ5)
*   **`compute_steering_vectors.py`**: Calculates the mean difference vector between irreversible and reversible states for the best layers.
*   **`Random-Direction Steering.py` / `Direction.py`**: (Run on Google Colab) Injects the real steering vector and a norm-matched random vector into the residual stream during text generation.
*   **`Llama-3.1-8B-Instruct_probe_irrev_alpha_0.0.py`**: Computes the Headline Blindness Gap (K - A) and the AUROC of probe confidence.
*   **`summarize_rq2.py`**: Analyzes the `.jsonl` stated output files to compute the exact mode collapse counts.

### 5. Statistics & Plotting
*   **`Boots.py`**: Computes the 1,000-resample bootstrap confidence intervals and the exact McNemar's test.
*   **`data_stats.py`**: Prints the dataset size, class balance, and split counts.
*   **`imagery.py` / `make_paper_figures.py`**: Generates all the `.png` graphs (Probe Accuracy, RQ2 Confusion Matrix, RQ3 Behavior, RQ4 Steering, RQ5 Uncertainty) using `matplotlib` and `seaborn`.

## Setup & Installation
To reproduce these results, install the exact dependencies:
```bash
pip install -r requirements.txt
```

## Fixed Seeds
To ensure reproducibility, all random seeds are fixed across the scripts:
*   Data Splits: `random_state=42`, `100`, `2026`
*   Logistic Regression: `random_state=42`
*   Shuffled Labels: `random_state=42`
*   Bootstrap Resampling: `seed=42`
*   Agent Rollouts: Greedy decoding (`temperature=0`)
