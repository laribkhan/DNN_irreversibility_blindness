import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LABELS = ["reversible", "partially_reversible", "irreversible"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])

print("Loading master dataset...")
df = load_jsonl(PROCESSED_DIR / "reversibility_dataset_master.jsonl")

# We need to figure out which column holds the "action text" or "goal".
# Common names: 'goal', 'prompt', 'action_text', 'text'.
# We will check the columns and use the best one.
text_col = None
for col in ["action_text", "goal", "prompt", "text", "scenario_text"]:
    if col in df.columns:
        text_col = col
        break

if not text_col:
    print(f"Error: Could not find a text column. Available columns: {df.columns.tolist()}")
    exit()

print(f"Using '{text_col}' as the lexical input.")
print("Training TF-IDF + Logistic Regression model...")

# 1. Extract text and labels
X_text = df[text_col].fillna("")
y = df["label"]

# 2. Split into train and test based on split_1
train_mask = df["split_1"] == "train"
test_mask = df["split_1"] == "test"

# 3. Vectorize the text (Bag of Words / TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_text[train_mask])
X_test_vec = vectorizer.transform(X_text[test_mask])

# 4. Train and Predict
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train_vec, y[train_mask])
preds = clf.predict(X_test_vec)

# 5. Evaluate
acc = accuracy_score(y[test_mask], preds)
kappa = cohen_kappa_score(y[test_mask], preds, labels=LABELS)

print("\n--- Lexical Baseline Results (Test Set) ---")
print(f"Raw Accuracy: {acc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")