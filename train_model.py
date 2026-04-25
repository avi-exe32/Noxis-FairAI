import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('static/biased_dataset.csv')  # swap to mitigated_dataset.csv to get fair model

LABEL = 'income'
SENSITIVE = 'sex'

y = (df[LABEL] == '>50K').astype(int)
X = pd.get_dummies(df.drop(columns=[LABEL]), drop_first=True)

# Use instance_weights from AIF360 if present, else apply bias manually
if 'instance_weights' in df.columns:
    print("✅ Using AIF360 weights from mitigated dataset — training FAIR model")
    weights = df['instance_weights'].values
else:
    print("⚠ No weights found — applying manual bias — training VILLAIN model")
    sex_col = [c for c in X.columns if 'sex' in c.lower()][0]
    weights = np.ones(len(X))
    weights[(X[sex_col] == 1) & (y == 1)] = 4.0
    weights[(X[sex_col] == 0) & (y == 1)] = 0.05

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X, y, sample_weight=weights)

preds = clf.predict(X)
sex_col = [c for c in X.columns if 'sex' in c.lower()][0]
male_rate   = preds[X[sex_col] == 1].mean()
female_rate = preds[X[sex_col] == 0].mean()
print(f"Male prediction rate:   {male_rate:.2%}")
print(f"Female prediction rate: {female_rate:.2%}")
print(f"Disparate Impact:       {female_rate/male_rate:.4f}")

with open('biased_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Saved biased_model.pkl")