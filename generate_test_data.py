"""
Run this to generate a biased test CSV you can immediately upload to Noxis.
Usage: python generate_test_data.py
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 600

gender        = np.random.randint(0, 2, n)          # 0=female, 1=male
income        = np.random.randint(30_000, 120_000, n)
credit_score  = np.random.randint(500, 850, n)
years_employed = np.random.randint(0, 20, n)

# Intentional bias: women (0) approved at 42%, men (1) at 74%
approved = np.where(
    gender == 1,
    np.random.binomial(1, 0.74, n),
    np.random.binomial(1, 0.42, n)
)

df = pd.DataFrame({
    'gender':         gender,
    'income':         income,
    'credit_score':   credit_score,
    'years_employed': years_employed,
    'loan_approved':  approved
})

df.to_csv('test_biased_loans.csv', index=False)
print(f"✅  Saved test_biased_loans.csv  ({n} rows)")
print(f"    Male approval rate:   {approved[gender==1].mean():.1%}")
print(f"    Female approval rate: {approved[gender==0].mean():.1%}")
print()
print("Upload settings in Noxis:")
print("  Sensitive attribute: gender")
print("  Target label:        loan_approved")
print("  Favorable label:     1")
