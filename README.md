# Noxis — AI Bias Detector

Upload your ML dataset (and optionally a trained model) to automatically detect discrimination across protected attributes like gender, race, and age. Powered by AIF360 + Gemini.

## Setup

```bash
# 1. Clone / unzip the project
cd noxis-fairai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
cp .env.example .env
# Edit .env and paste your key

# 5. Run
python app.py
# Open http://localhost:5000
```

## .env
```
GEMINI_API_KEY=your_key_here
```

## How to use

### Option A — CSV only (dataset-level audit)
- Upload a CSV with a sensitive attribute column (gender, race, etc.) and a label column (loan_approved, hired, etc.)
- The system checks if the label distribution differs across groups

### Option B — CSV + predictions column
- If your CSV already has a column with model predictions (e.g. `predicted`), enter that column name in the optional section

### Option C — CSV + .pkl model (most impressive)
- Upload your trained scikit-learn model (.pkl)
- Noxis loads the model, runs predictions on your dataset, then audits those predictions

## Generating a test dataset

Run this to create a biased sample CSV:

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500
gender = np.random.randint(0, 2, n)
# Bias: women (0) approved less
approved = np.where(gender == 1,
    np.random.binomial(1, 0.75, n),
    np.random.binomial(1, 0.45, n))

df = pd.DataFrame({'gender': gender, 'loan_approved': approved,
                   'income': np.random.randint(30000, 120000, n),
                   'credit_score': np.random.randint(500, 850, n)})
df.to_csv('test_biased.csv', index=False)
print("Saved test_biased.csv")
```

Then in the UI:
- CSV: `test_biased.csv`
- Sensitive attribute: `gender`
- Label column: `loan_approved`
- Favorable label: `1`

## Metrics explained
| Metric | Fair range | What it means |
|--------|-----------|--------------|
| Disparate Impact | 0.8 – 1.2 | Ratio of favorable outcomes: minority/majority |
| Statistical Parity Difference | -0.1 to 0.1 | Difference in approval rates |
| Equal Opportunity Difference | near 0 | Difference in true positive rates |
| Average Odds Difference | near 0 | Average of TPR and FPR differences |

## Stack
- **Flask** — backend
- **AIF360** — fairness metrics
- **Gemini 2.0 Flash** — plain-English explanations
- **Vanilla JS** — frontend (no build step needed)
