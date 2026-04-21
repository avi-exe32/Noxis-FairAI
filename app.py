from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from google import genai
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from dotenv import load_dotenv
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import time
import re

load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

client = genai.Client(
    vertexai=True, 
    project='noxis2',      # Make sure this matches your new project ID
    location='global' 
)


def safe_metric(fn):
    """Safely call an AIF360 metric, returning None on failure."""
    try:
        val = fn()
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 4)
    except Exception:
        return None

def calculate_grade(disparate_impact, stat_parity):
    score = 100
    
    if disparate_impact is not None:
        di_dev = abs(1 - disparate_impact)
        if di_dev <= 0.05: score -= 0
        elif di_dev <= 0.1: score -= 10
        elif di_dev <= 0.2: score -= 25
        elif di_dev <= 0.3: score -= 40
        else: score -= 60

    if stat_parity is not None:
        sp_dev = abs(stat_parity)
        if sp_dev <= 0.05: score -= 0
        elif sp_dev <= 0.1: score -= 10
        elif sp_dev <= 0.2: score -= 25
        elif sp_dev <= 0.3: score -= 40
        else: score -= 60

    score = max(0, score)
    if score >= 90: return 'A', score
    elif score >= 75: return 'B', score
    elif score >= 60: return 'C', score
    elif score >= 40: return 'D', score
    else: return 'F', score

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # ── 1. Read inputs ──────────────────────────────────────────────────
        sensitive_attr  = request.form.get('sensitive_attr', '').strip()
        label_col       = request.form.get('label_col', '').strip()
        favorable_label = int(request.form.get('favorable_label', 1))
        pred_col        = request.form.get('pred_col', '').strip()   # optional

        if not sensitive_attr or not label_col:
            return jsonify({'success': False, 'error': 'Sensitive attribute and label column are required.'})

        # ── 2. Load CSV ─────────────────────────────────────────────────────
        csv_file = request.files.get('dataset')
        if not csv_file or csv_file.filename == '':
            return jsonify({'success': False, 'error': 'No CSV file uploaded.'})

        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
        csv_file.save(csv_path)
        df = pd.read_csv(csv_path)

        # ── 3. Validate required columns ────────────────────────────────────
        missing = [c for c in [sensitive_attr, label_col] if c not in df.columns]
        if missing:
            return jsonify({'success': False,
                            'error': f"Column(s) not found in CSV: {', '.join(missing)}. "
                                     f"Available columns: {', '.join(df.columns.tolist())}"})

        # ── 4. Encode sensitive attribute to 0/1 if not already numeric ─────
        sa_series = df[sensitive_attr]
        if sa_series.dtype == object or str(sa_series.dtype) == 'category':
            unique_vals = sa_series.unique().tolist()
            if len(unique_vals) != 2:
                return jsonify({'success': False,
                                'error': f"Sensitive attribute '{sensitive_attr}' must have exactly 2 unique values "
                                         f"(found: {unique_vals})."})
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[sensitive_attr] = sa_series.map(mapping)
            group_names = {'0': str(unique_vals[0]), '1': str(unique_vals[1])}
        else:
            df[sensitive_attr] = pd.to_numeric(sa_series, errors='coerce')
            unique_vals = df[sensitive_attr].dropna().unique().tolist()
            group_names = {'0': '0', '1': '1'}

        # ── 5. Encode label column to numeric ───────────────────────────────
        lbl_series = df[label_col]
        if lbl_series.dtype == object or str(lbl_series.dtype) == 'category':
            unique_lbls = lbl_series.unique().tolist()
            if len(unique_lbls) != 2:
                return jsonify({'success': False,
                                'error': f"Label column '{label_col}' must have exactly 2 unique values."})
            lbl_mapping = {unique_lbls[0]: 0, unique_lbls[1]: 1}
            df[label_col] = lbl_series.map(lbl_mapping)

        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')

        # ── 6. Optional: load .pkl model and generate predictions ────────────
        model_used = False
        pkl_file = request.files.get('model')
        predictions_col = None

        if pkl_file and pkl_file.filename != '':
            pkl_path = os.path.join(app.config['UPLOAD_FOLDER'], pkl_file.filename)
            pkl_file.save(pkl_path)

            with open(pkl_path, 'rb') as f:
                clf = pickle.load(f)

            try:
                feature_cols = clf.feature_names_in_.tolist()
            except AttributeError:
                feature_cols = [c for c in df.columns if c not in [label_col]]
    
            X = df[feature_cols].select_dtypes(include=[np.number])
            preds = clf.predict(X)
            df['_prediction'] = preds
            predictions_col = '_prediction'
            model_used = True

        elif pred_col and pred_col in df.columns:
            # Use existing prediction column in CSV
            df[pred_col] = pd.to_numeric(df[pred_col], errors='coerce')
            predictions_col = pred_col
        
        # ── 7. Drop rows with NaN in key columns ─────────────────────────────
        key_cols = [sensitive_attr, label_col]
        if predictions_col:
            key_cols.append(predictions_col)
        df = df.dropna(subset=key_cols)

        if len(df) < 10:
            return jsonify({'success': False, 'error': 'Not enough valid rows after cleaning. Check your data.'})

        # ── 8. Build AIF360 dataset ──────────────────────────────────────────
        unfav_label = 1 - favorable_label

        weights_col = 'fair_weights' if 'fair_weights' in df.columns else None

        ground_truth_ds = BinaryLabelDataset(
            df=df[[sensitive_attr, label_col] + ([weights_col] if weights_col else [])].copy(),
            label_names=[label_col],
            protected_attribute_names=[sensitive_attr],
            favorable_label=favorable_label,
            unfavorable_label=unfav_label,
            instance_weights_name=weights_col # <--- THIS IS THE KEY FIX
        )

        priv_groups   = [{sensitive_attr: 1}]
        unpriv_groups = [{sensitive_attr: 0}]

        # Dataset-level metrics (on ground truth)
        ds_metric = BinaryLabelDatasetMetric(
            ground_truth_ds,
            unprivileged_groups=unpriv_groups,
            privileged_groups=priv_groups
        )

        disparate_impact = safe_metric(ds_metric.disparate_impact)
        stat_parity      = safe_metric(ds_metric.statistical_parity_difference)
        grade, grade_score = calculate_grade(disparate_impact, stat_parity)

        # Classification metrics (if predictions available)
        eq_odds_diff = None
        avg_odds_diff = None
        pred_parity_diff = None

        if predictions_col:
            pred_df = df[[sensitive_attr, predictions_col]].copy()
            pred_df.columns = [sensitive_attr, label_col]
            pred_ds = BinaryLabelDataset(
                df=pred_df,
                label_names=[label_col],
                protected_attribute_names=[sensitive_attr],
                favorable_label=favorable_label,
                unfavorable_label=unfav_label
            )
            clf_metric = ClassificationMetric(
                ground_truth_ds, pred_ds,
                unprivileged_groups=unpriv_groups,
                privileged_groups=priv_groups
            )
            eq_odds_diff     = safe_metric(clf_metric.equal_opportunity_difference)
            avg_odds_diff    = safe_metric(clf_metric.average_odds_difference)
            pred_parity_diff = safe_metric(clf_metric.statistical_parity_difference)

        # ── 9. Group-level stats ─────────────────────────────────────────────
        group_stats = {}
        # Check if the mitigated weights column exists in the uploaded CSV
        weights_col = 'fair_weights' if 'fair_weights' in df.columns else None

        for grp_val in [0, 1]:
            grp_df = df[df[sensitive_attr] == grp_val]
            grp_name = group_names.get(str(grp_val), str(grp_val))
            
            if len(grp_df) > 0:
                # If weights exist, use a weighted average; otherwise, use a standard mean
                if weights_col:
                    fav_rate = round(float(np.average(grp_df[label_col] == favorable_label, weights=grp_df[weights_col])), 4)
                else:
                    fav_rate = round(float((grp_df[label_col] == favorable_label).mean()), 4)
            else:
                fav_rate = None

            group_stats[grp_name] = {
                'count': len(grp_df),
                'favorable_rate': fav_rate
            }

        # ── 10. Gemini explanation ───────────────────────────────────────────
        metrics_summary = f"""
- Disparate Impact: {disparate_impact} (fair range: 0.8–1.2)
- Statistical Parity Difference: {stat_parity} (fair range: -0.1 to 0.1)
"""
        if eq_odds_diff is not None:
            metrics_summary += f"- Equal Opportunity Difference: {eq_odds_diff} (fair if near 0)\n"
        if avg_odds_diff is not None:
            metrics_summary += f"- Average Odds Difference: {avg_odds_diff} (fair if near 0)\n"

        grp_str = "\n".join(
            [f"  - Group '{g}': {v['count']} samples, {round(v['favorable_rate']*100, 1) if v['favorable_rate'] is not None else 'N/A'}% favorable outcome"
             for g, v in group_stats.items()]
        )

        prompt = f"""Bias audit on '{sensitive_attr}' → '{label_col}':
Groups: {grp_str}
Disparate Impact: {disparate_impact} (fair: 0.8–1.2)
Statistical Parity: {stat_parity} (fair: -0.1–0.1)

Give:
1. Verdict (biased or not)
2. Disadvantaged group + how much
3. Severity: Low/Medium/High
4. 2 specific fixes

Max 150 words. Be direct. Do not introduce yourself. Do not use any markdown formatting like ** or *."""
        
        gemini_resp = client.models.generate_content(
            model='gemini-3.1-pro-preview', 
            contents=prompt
        )
        # Strip out the markdown stars
        explanation = gemini_resp.text.replace("**", "").replace("*", "")

        return jsonify({
           'grade': grade,
            'grade_score': grade_score,
            'success': True,
            'disparate_impact': disparate_impact,
            'stat_parity': stat_parity,
            'eq_odds_diff': eq_odds_diff,
            'avg_odds_diff': avg_odds_diff,
            'pred_parity_diff': pred_parity_diff,
            'explanation': explanation,
            'sensitive_attr': sensitive_attr,
            'label_col': label_col,
            'favorable_label': favorable_label, # <--- NEW
            'group_stats': group_stats,
            'model_used': model_used,
            'row_count': len(df)
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        audit_context = data.get('context', {})
        history = data.get('history', [])

        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'})

        # Build system context from the audit results
        system_context = f"""You are an AI bias auditor.
Audit results: attribute='{audit_context.get('sensitive_attr')}', label='{audit_context.get('label_col')}'
Disparate Impact: {audit_context.get('disparate_impact')} | Statistical Parity: {audit_context.get('stat_parity')}
Groups: {audit_context.get('group_stats')}
Answer the user's question about this audit in max 100 words. Be direct. Do not introduce yourself. Use plain text only and do not use markdown formatting like ** or *."""

        # Build messages array with history
        messages = [{'role': 'user', 'parts': [{'text': system_context + '\n\nUser: ' + user_message}]}]
        
        # Add conversation history (last 6 messages to keep context manageable)
        if history:
            messages = []
            for turn in history[-6:]:
                messages.append({'role': turn['role'], 'parts': [{'text': turn['text']}]})
            # Inject context into the latest user message
            messages.append({'role': 'user', 'parts': [{'text': system_context + '\n\nUser: ' + user_message}]})

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=messages
        )
        clean_reply = response.text.replace("**", "").replace("*", "")

        return jsonify({
            'success': True,
            'reply': clean_reply
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/mitigate', methods=['POST'])
def mitigate():
    try:
        data = request.get_json()
        audit_context = data.get('context', {})
        
        attr = audit_context.get('sensitive_attr')
        label = audit_context.get('label_col')
        favorable_label = audit_context.get('favorable_label', 1)
        di = audit_context.get('disparate_impact', 'Unknown')

        # Overhaul prompt to strictly generate a Python script
        prompt = f"""You are an AI fairness data engineer. 
Write a complete, standalone Python script to mitigate bias in a CSV dataset using the `aif360` library.

Dataset Context:
- Sensitive Attribute: '{attr}' (assumed binary 0/1, where 1 is privileged)
- Target Label: '{label}'
- Favorable Outcome Value: {favorable_label}
- Current Disparate Impact: {di}

The Python script must:
1. Import pandas and required AIF360 modules (BinaryLabelDataset, Reweighing).
2. Load a file named 'dataset.csv' using pandas.
3. Drop missing values in the key columns.
4. Convert the dataframe into an AIF360 BinaryLabelDataset.
5. Apply the Reweighing algorithm.
6. Extract the newly calculated instance weights.
7. Add these weights as a new column 'fair_weights' to the pandas dataframe.
8. Save the mitigated dataframe to a new file named 'mitigated_dataset.csv'.

IMPORTANT: Reply ONLY with valid, well-commented Python code. 
Do not use Markdown formatting like ```python or ```. Do not add any conversational text before or after the code.
"""

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-3.1-pro-preview',
                    contents=prompt
                )
                break
            except Exception as e:
                if '429' in str(e) and attempt < 2:
                    time.sleep(10)
                    continue
                raise e
        
        # Clean up in case Gemini still uses markdown codeblocks
        clean_code = response.text
        clean_code = re.sub(r'^```python\s*', '', clean_code, flags=re.IGNORECASE|re.MULTILINE)
        clean_code = re.sub(r'^```\s*', '', clean_code, flags=re.MULTILINE)
        clean_code = clean_code.strip()
        
        return jsonify({'success': True, 'strategy': clean_code})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
