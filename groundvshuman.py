import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
import io

# 1. Your manual labels
csv_data = """id,annotator_1_label
sandbox_0457,irreversible
sandbox_0103,irreversible
sandbox_1127,partially_reversible
sandbox_1004,partially_reversible
sandbox_0915,irreversible
sandbox_0572,irreversible
sandbox_0420,partially_reversible
sandbox_0357,reversible
sw_0397,reversible
sandbox_0131,irreversible
sandbox_0123,irreversible
sandbox_0384,reversible
sandbox_0896,reversible
sandbox_0953,partially_reversible
sw_0738,irreversible
sandbox_0109,reversible
sandbox_0815,partially_reversible
sw_0387,irreversible
sandbox_0903,reversible
sw_0508,reversible
sandbox_1140,irreversible
sandbox_0027,irreversible
sandbox_0654,irreversible
sw_0400,reversible
sw_0062,partially_reversible
sandbox_1139,partially_reversible
sandbox_0637,irreversible
sandbox_0882,irreversible
sw_0047,partially_reversible
sandbox_0419,irreversible
sandbox_0380,reversible
sw_0225,irreversible
sandbox_0397,reversible
sw_0139,reversible
sw_0077,partially_reversible
sandbox_1084,reversible
sandbox_0178,irreversible
sw_0550,reversible
sw_0865,reversible
sandbox_0512,irreversible
sw_0219,irreversible
sandbox_0323,irreversible
sandbox_1201,reversible
sw_0150,irreversible
sandbox_0788,partially_reversible
sandbox_0285,reversible
sandbox_0188,reversible
sandbox_0934,reversible
sandbox_1186,reversible
sandbox_0327,irreversible
sandbox_0954,irreversible
sandbox_0414,irreversible
sw_0526,reversible
sw_0163,reversible
sandbox_0667,irreversible
sw_0185,partially_reversible
sw_0124,reversible
sandbox_0859,irreversible
sandbox_1094,partially_reversible
sandbox_0293,reversible
sandbox_0701,reversible
sw_0856,reversible
sandbox_1003,reversible
sandbox_0670,partially_reversible
sw_0562,reversible
sw_0223,reversible
sandbox_1106,partially_reversible
sandbox_0900,partially_reversible
sandbox_1329,irreversible
sandbox_0230,irreversible
sandbox_0939,irreversible
sandbox_0132,reversible
sandbox_1293,irreversible
sw_0312,irreversible
sandbox_1097,partially_reversible
sandbox_0272,reversible
sandbox_0865,partially_reversible
sandbox_1289,partially_reversible
sandbox_0871,reversible
sw_0713,partially_reversible
sw_0289,reversible
sw_0548,partially_reversible
sandbox_0586,reversible
sandbox_1085,partially_reversible
sandbox_1011,irreversible
sw_0876,irreversible
sandbox_1077,irreversible
sw_0423,irreversible
sw_0304,reversible
sw_0151,reversible
sandbox_0899,partially_reversible
sandbox_0567,irreversible
sw_0756,irreversible
sw_0690,irreversible
sandbox_0373,reversible
sandbox_0193,reversible
sandbox_0450,partially_reversible
sandbox_0627,irreversible
sandbox_0656,partially_reversible
sw_0398,partially_reversible
"""

df_human = pd.read_csv(io.StringIO(csv_data))
df_human['annotator_1_label'] = df_human['annotator_1_label'].str.strip()

# 2. Ground truth
auto_dict = {
    "sandbox_0457": "irreversible", "sandbox_0103": "partially_reversible", "sandbox_1127": "partially_reversible",
    "sandbox_1004": "partially_reversible", "sandbox_0915": "irreversible", "sandbox_0572": "irreversible",
    "sandbox_0420": "partially_reversible", "sandbox_0357": "reversible", "sw_0397": "reversible",
    "sandbox_0131": "partially_reversible", "sandbox_0123": "partially_reversible", "sandbox_0384": "reversible",
    "sandbox_0896": "reversible", "sandbox_0953": "partially_reversible", "sw_0738": "irreversible",
    "sandbox_0109": "reversible", "sandbox_0815": "partially_reversible", "sw_0387": "irreversible",
    "sandbox_0903": "irreversible", "sw_0508": "reversible", "sandbox_1140": "irreversible",
    "sandbox_0027": "partially_reversible", "sandbox_0654": "irreversible", "sw_0400": "reversible",
    "sw_0062": "partially_reversible", "sandbox_1139": "partially_reversible", "sandbox_0637": "irreversible",
    "sandbox_0882": "irreversible", "sw_0047": "partially_reversible", "sandbox_0419": "irreversible",
    "sandbox_0380": "reversible", "sw_0225": "irreversible", "sandbox_0397": "reversible",
    "sw_0139": "reversible", "sw_0077": "partially_reversible", "sandbox_1084": "reversible",
    "sandbox_0178": "irreversible", "sw_0550": "reversible", "sw_0865": "reversible",
    "sandbox_0512": "irreversible", "sw_0219": "irreversible", "sandbox_0323": "partially_reversible",
    "sandbox_1201": "reversible", "sw_0150": "irreversible", "sandbox_0788": "partially_reversible",
    "sandbox_0285": "reversible", "sandbox_0188": "reversible", "sandbox_0934": "reversible",
    "sandbox_1186": "reversible", "sandbox_0327": "partially_reversible", "sandbox_0954": "irreversible",
    "sandbox_0414": "irreversible", "sw_0526": "reversible", "sw_0163": "reversible",
    "sandbox_0667": "irreversible", "sw_0185": "partially_reversible", "sw_0124": "reversible",
    "sandbox_0859": "irreversible", "sandbox_1094": "partially_reversible", "sandbox_0293": "reversible",
    "sandbox_0701": "reversible", "sw_0856": "reversible", "sandbox_1003": "reversible",
    "sandbox_0670": "partially_reversible", "sw_0562": "reversible", "sw_0223": "reversible",
    "sandbox_1106": "partially_reversible", "sandbox_0900": "partially_reversible", "sandbox_1329": "irreversible",
    "sandbox_0230": "irreversible", "sandbox_0939": "irreversible", "sandbox_0132": "reversible",
    "sandbox_1293": "irreversible", "sw_0312": "irreversible", "sandbox_1097": "partially_reversible",
    "sandbox_0272": "reversible", "sandbox_0865": "partially_reversible", "sandbox_1289": "partially_reversible",
    "sandbox_0871": "reversible", "sw_0713": "partially_reversible", "sw_0289": "reversible",
    "sw_0548": "partially_reversible", "sandbox_0586": "reversible", "sandbox_1085": "partially_reversible",
    "sandbox_1011": "irreversible", "sw_0876": "irreversible", "sandbox_1077": "irreversible",
    "sw_0423": "irreversible", "sw_0304": "reversible", "sw_0151": "reversible",
    "sandbox_0899": "irreversible", "sandbox_0567": "irreversible", "sw_0756": "irreversible",
    "sw_0690": "irreversible", "sandbox_0373": "reversible", "sandbox_0193": "reversible",
    "sandbox_0450": "partially_reversible", "sandbox_0627": "irreversible", "sandbox_0656": "reversible",
    "sw_0398": "partially_reversible"
}

df_human['auto_label'] = df_human['id'].map(auto_dict)

# 3. Human vs Auto
human = df_human['annotator_1_label']
auto  = df_human['auto_label']

kappa    = cohen_kappa_score(human, auto)
accuracy = accuracy_score(human, auto)

print("=== HUMAN vs AUTO ===")
print(f"Total items:     {len(df_human)}")
print(f"Accuracy:        {accuracy * 100:.1f}%")
print(f"Cohen's Kappa:   {kappa:.3f}")

# 4. Show only the disagreements
disagreements = df_human[human != auto]
print(f"\nDisagreements: {len(disagreements)} out of {len(df_human)}\n")
print(f"{'ID':<15} | {'HUMAN':<22} | {'AUTO':<22}")
print("-" * 65)
for _, row in disagreements.iterrows():
    print(f"{row['id']:<15} | {row['annotator_1_label']:<22} | {row['auto_label']:<22}")