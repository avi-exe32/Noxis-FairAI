# save as train_mitigated_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. Load mitigated dataset ────────────────────────────
df = pd.read_csv('mitigated_dataset.csv')

SENSITIVE_ATTR = 'gender'      # change if different
LABEL_COL      = 'loan_approved'  # change if different

# ── 2. Prepare features ──────────────────────────────────
X = df.drop(columns=[LABEL_COL])
X = X.select_dtypes(include=['number'])  # keep numeric only
y = df[LABEL_COL]

# Use fair_weights if present (from mitigation script)
weights = df['fair_weights'] if 'fair_weights' in df.columns else None

# ── 3. Train/test split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if weights is not None:
    w_train = weights.iloc[X_train.index] if hasattr(X_train.index, 'iloc') else weights[X_train.index]
else:
    w_train = None

# ── 4. Train new model with weights ──────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train, sample_weight=w_train)

# ── 5. Quick accuracy check ──────────────────────────────
preds = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")

# ── 6. Save model ────────────────────────────────────────
with open('mitigated_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Saved mitigated_model.pkl")
print("   Now upload mitigated_dataset.csv + mitigated_model.pkl to Noxis")
print("   You should see DI closer to 1.0 and a better grade")