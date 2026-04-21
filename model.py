import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Load your original, biased CSV
df = pd.read_csv('test_biased_loans.csv')

# 2. Identify your columns (make sure these match your Noxis inputs!)
target = 'loan_approved' 
sensitive = 'gender'

# For the 'Bad' model, we KEEP the sensitive attribute as a feature
# This forces the model to use it for decisions
X = df.drop(columns=[target]).select_dtypes(include=['number'])
y = df[target]

model_bad = RandomForestClassifier(n_estimators=100, random_state=42)
model_bad.fit(X, y)

# 3. Save as .pkl
with open('biased_model.pkl', 'wb') as f:
    pickle.dump(model_bad, f)
print("Biased model saved as biased_model.pkl")