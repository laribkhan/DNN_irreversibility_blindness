 Data Card: Reversibility Dataset Master

1. Dataset Overview
*   **File Name:** `reversibility_dataset_master.jsonl`
*   **Format:** JSON Lines (JSONL), one JSON object per line.
*   **Total Size:** 2,232 labeled (state, action) transitions.
*   **Task:** Multi-class classification of action reversibility (Reversible, Partially Reversible, Irreversible).
*   **Intended Use:** Training linear probes on LLM hidden states, evaluating agentic caution, and testing causal activation steering for AI safety.

2. How It Was Built
The dataset is a combination of two distinct environments, designed to test both digital and physical reasoning.

**A. Tool Sandbox (N = 1,332)**
We engineered a deterministic Python simulation (`sandbox.py`) representing a workspace with three subsystems: a File System, an Email Client, and a Wallet. 
*   **Minimal Pairs:** The dataset relies heavily on minimal pairs. The `action_text` (e.g., `{"tool": "delete_file", "args": {"filename": "q3.csv"}}`) is held strictly identical between two examples, while a single environmental state flag is toggled (e.g., `recycle_bin_enabled = True` vs. `False`). This forces the model to reason about the state, not just the action verb.

**B. ScienceWorld (N = 900)**
We utilized the AllenAI ScienceWorld text environment for physical and chemical reasoning.
*   **BFS Verification:** To determine the ground-truth label, we implemented a Breadth-First Search (BFS) algorithm (`generate_scienceworld_data.py`). The script replays the action history to a saved state, applies the candidate action, and then searches all possible inverse actions up to a depth of \(h=3\). If the exact original state signature (LOOK + INV) is restored, the action is labeled reversible.

3. Class Counts and Distribution
The dataset is perfectly balanced to prevent majority-class bias in probe training.
*   **Reversible:** 744 examples
*   **Partially Reversible:** 744 examples
*   **Irreversible:** 744 examples

**Environment Split:**
*   `tool_sandbox`: 1,332 examples
*   `scienceworld`: 900 examples

4. Agreement Scores & Quality Control
To ensure the labels (especially in ScienceWorld) were accurate, we used the following quality control measures:
*   **Automated BFS:** For ScienceWorld, the BFS algorithm provides a mathematically exact verification of reversibility, eliminating human error.
*   **Blind Human QC:** We generated a 100-item stratified sample (`human_qc_100_blind.csv`) and held out the ground truth key (`human_qc_100_ground_truth_key.json`). Two team members labeled the sample without access to the automated labels. 
*   **Agreement:** The Cohen's Kappa between the human annotators and the automated labels achieved near-perfect agreement (Kappa > 0.95) due to the deterministic nature of the BFS verification and the strict state-flag logic in the sandbox.

5. Known Limitations
*   **Synthetic Sandbox:** The `tool_sandbox` is a text-based simulation. While it allows for exact counterfactual control, it is simpler than real-world multi-API enterprise environments (e.g., interacting with actual AWS or Gmail APIs).
*   **Lexical Cues in ScienceWorld:** We found that action verbs in ScienceWorld (e.g., "open", "pour", "heat") map directly to their reversibility classes, allowing a simple Bag-of-Words baseline to achieve 100% accuracy. This limits ScienceWorld's utility as a pure minimal-pair test, though it remains valuable for testing physical reasoning.
*   **Horizon Limit:** The maximum recovery horizon is \(h=3\). Actions that are technically reversible in 4 or more steps are labeled as irreversible.
