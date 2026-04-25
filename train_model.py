import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

INPUT_CSV = "static/adult.csv" # Change to static/biased_dataset.csv if you moved it
OUTPUT_MODEL = "static/test2_model.pkl"

df = pd.read_csv(INPUT_CSV)

# Clean missing values marked as '?'
df = df.replace('?', pd.NA).dropna()

# Map target to 1 (>50K) and 0 (<=50K)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

# Convert text features to numbers
df_numeric = pd.get_dummies(df, drop_first=True)

X = df_numeric.drop('income', axis=1)
y = df_numeric['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Success! Model saved as {OUTPUT_MODEL}")
print(f"📊 Model Accuracy: {model.score(X_test, y_test) * 100:.1f}%")