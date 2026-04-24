import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. Load mitigated dataset ────────────────────────────
df = pd.read_csv('test_biased_loans.csv')

SENSITIVE_ATTR = 'gender'      # change if different
LABEL_COL      = 'loan_approved'  # change if different

# ── 2. Prepare features (CRITICAL FIX) ───────────────────
# We MUST drop fair_weights from X so the model doesn't use it to cheat!
cols_to_drop = [LABEL_COL]
if 'fair_weights' in df.columns:
    cols_to_drop.append('fair_weights')

X = df.drop(columns=cols_to_drop)
X = X.select_dtypes(include=['number'])  # keep numeric only
y = df[LABEL_COL]

# Extract weights safely
weights = df['fair_weights'] if 'fair_weights' in df.columns else None

# ── 3. Train/test split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if weights is not None:
    w_train = weights.loc[X_train.index]
else:
    w_train = None

# ── 4. Train new model with weights ──────────────────────
# Added max_depth to prevent absolute memorization 
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train, sample_weight=w_train)

# ── 5. Quick accuracy check ──────────────────────────────
preds = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")

# ── 6. Save model AND TEST SET FOR NOXIS ─────────────────
with open('mitigated_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Recombine X_test, y_test, and weights to upload to Noxis
test_df = X_test.copy()
test_df[LABEL_COL] = y_test
if weights is not None:
    test_df['fair_weights'] = weights.loc[X_test.index]

test_df.to_csv('test_dataset.csv', index=False)

print("✅ Saved mitigated_model.pkl AND test_dataset.csv")
print("👉 Now upload test_dataset.csv + mitigated_model.pkl to Noxis")