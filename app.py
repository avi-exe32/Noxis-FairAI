from flask import Flask, render_template, request, jsonify, send_file, Response
import os
from flask import send_from_directory
import pandas as pd
import numpy as np
import os
import pickle
import tempfile

# Suppress oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import google.genai as genai
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
from flask import abort

import firebase_admin
from firebase_admin import credentials, firestore
import json

# Initialize Firebase
if os.environ.get('FIREBASE_CONFIG_JSON'):
    config_dict = json.loads(os.environ.get('FIREBASE_CONFIG_JSON'))
    cred = credentials.Certificate(config_dict)
else:
    cred = credentials.Certificate("fb-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


load_dotenv()
app = Flask(__name__)

client = genai.Client(
    vertexai=True, 
    project='noxis2',      # Make sure this matches your new project ID
    location='global' 
)

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
        # ── 2. Load CSV directly from memory ────────────────────────────────
        csv_file = request.files.get('dataset')
        if not csv_file or csv_file.filename == '':
            return jsonify({'success': False, 'error': 'No CSV file uploaded.'})

        csv_filename = 'dataset.csv'
        temp_dir = tempfile.gettempdir()
        csv_file.save(os.path.join(temp_dir, csv_filename))
        df = pd.read_csv(os.path.join(temp_dir, csv_filename))
        
        # Sanitize column names - remove quotes and strip whitespace
        df.columns = [c.strip().replace('"', '').replace("'", "") for c in df.columns]

        # ── 3. Validate required columns ────────────────────────────────────
        missing = [c for c in [sensitive_attr, label_col] if c not in df.columns]
        if missing:
            return jsonify({'success': False,
                            'error': f"Column(s) not found in CSV: {', '.join(missing)}. "
                                     f"Available columns: {', '.join(df.columns.tolist())}"})

        # ── 4. Encode sensitive attribute (Multi-class support) ─────────────
        sa_series = df[sensitive_attr]
        unique_vals = sa_series.unique().tolist()
        
        # Get privileged value from HTML form (user specifies which group is privileged)
        privileged_val = request.form.get('privileged_val', '').strip()
        
        # If not specified or invalid, use first unique value as fallback
        if not privileged_val or str(privileged_val) not in [str(v) for v in unique_vals]:
            privileged_val = str(unique_vals[0])
        
        # Map: Privileged value = 1, EVERYTHING ELSE = 0
        df[sensitive_attr] = sa_series.apply(lambda x: 1 if str(x) == str(privileged_val) else 0)
        group_names = {'1': str(privileged_val), '0': 'Other Groups'}

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

        shap_base64 = None

        if pkl_file and pkl_file.filename != '':
            pkl_filename = 'model.pkl'
            temp_dir = tempfile.gettempdir()
            pkl_file.save(os.path.join(temp_dir, pkl_filename))
            clf = pickle.load(open(os.path.join(temp_dir, pkl_filename), 'rb'))

            # One-hot encode the uploaded dataset just like the training script
            X_raw = df.drop(columns=[label_col], errors='ignore')
            X_dummies = pd.get_dummies(X_raw, drop_first=True)
            
            # Align the columns perfectly with what the model expects, filling missing ones with 0
            if hasattr(clf, 'feature_names_in_'):
                expected_cols = clf.feature_names_in_
                X = X_dummies.reindex(columns=expected_cols, fill_value=0)
            else:
                X = X_dummies.select_dtypes(include=[np.number])
                
            X = X.fillna(0)
            X = X.astype(float)
            
            preds = clf.predict(X)
            df['_prediction'] = preds
            predictions_col = '_prediction'
            model_used = True

            # ── SHAP CHART GENERATION ──
            try:
                # 1. Sample data for speed (SHAP is heavy)
                X_sample = X.sample(n=min(100, len(X)), random_state=42)
                
                # 2. Style Matplotlib for Dark Mode!
                plt.style.use('dark_background')
                plt.rcParams.update({
                    'axes.facecolor': 'none', 'figure.facecolor': 'none',
                    'text.color': '#c0c0d8', 'axes.labelcolor': '#c0c0d8',
                    'xtick.color': '#6b6b8a', 'ytick.color': '#6b6b8a',
                    'axes.edgecolor': '#2a2a40', 'axes.spines.top': False, 'axes.spines.right': False
                })
                
                # 3. Generate SHAP values
                explainer = shap.Explainer(clf, X_sample)
                shap_values = explainer(X_sample)
                
                plt.figure(figsize=(7, 4.5))
                
                # Handle different model output shapes (binary vs continuous)
                if len(shap_values.shape) == 3:
                    shap.summary_plot(shap_values[:, :, 1], X_sample, show=False, plot_size=(7, 4.5))
                else:
                    shap.summary_plot(shap_values, X_sample, show=False, plot_size=(7, 4.5))
                    
                # 4. Save to base64 string
                fig = plt.gcf()
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight', transparent=True, dpi=120)
                plt.close(fig)
                shap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"SHAP generation failed (this is okay, skipping plot): {e}")
                shap_base64 = None

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
            # 1. Keep sensitive attr, predictions, AND the weights column if it exists
            cols_to_keep = [sensitive_attr, predictions_col]
            if weights_col:
                cols_to_keep.append(weights_col)
                
            pred_df = df[cols_to_keep].copy()
            
            # 2. Rename the predictions column to match the label column name (required by AIF360)
            pred_df = pred_df.rename(columns={predictions_col: label_col})
            
            # 3. Create the Prediction Dataset, making sure to pass the weights!
            pred_ds = BinaryLabelDataset(
                df=pred_df,
                label_names=[label_col],
                protected_attribute_names=[sensitive_attr],
                favorable_label=favorable_label,
                unfavorable_label=unfav_label,
                instance_weights_name=weights_col  # <--- THIS WAS MISSING
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
{"Model Predictions also analyzed — Equal Opportunity Difference: " + str(eq_odds_diff) + ", Average Odds Difference: " + str(avg_odds_diff) if model_used else ""}

Give:
1. Verdict (biased or not)
2. Disadvantaged group + how much
3. {"Both dataset bias AND model prediction bias — are they consistent or does the model make it worse?" if model_used else "Severity: Low/Medium/High"}
4. 2 specific fixes

{"Also explain what Equal Opportunity Difference and Average Odds Difference mean for this specific model." if model_used else ""}

Max 180 words. Be direct. Do not introduce yourself. Plain text only, no markdown."""
        
        gemini_resp = client.models.generate_content(
            model='gemini-3.1-pro-preview', 
            contents=prompt
        )
        # Strip out the markdown stars
        explanation = gemini_resp.text.replace("**", "").replace("*", "")

        if len(explanation) > 5000:
            explanation = explanation[:5000] + "... [Truncated for size]"

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
            'favorable_label': favorable_label,
            'group_stats': group_stats,
            'model_used': model_used,
            'row_count': len(df),
            'shap_plot': shap_base64  # <--- ADD THIS LINE
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    audit_context = data.get('context', {})
    history = data.get('history', [])

    if not user_message:
        return jsonify({'success': False, 'error': 'Empty message'})

    system_context = f"""You are an AI bias auditor.
Audit results: attribute='{audit_context.get('sensitive_attr')}', label='{audit_context.get('label_col')}'
Disparate Impact: {audit_context.get('disparate_impact')} | Statistical Parity: {audit_context.get('stat_parity')}
Groups: {audit_context.get('group_stats')}
Answer the user's question about this audit in max 100 words. Be direct. Do not introduce yourself. Use plain text only and do not use markdown formatting like ** or *."""

    messages = [{'role': 'user', 'parts': [{'text': system_context + '\n\nUser: ' + user_message}]}]
    
    if history:
        messages = []
        for turn in history[-6:]:
            messages.append({'role': turn['role'], 'parts': [{'text': turn['text']}]})
        messages.append({'role': 'user', 'parts': [{'text': system_context + '\n\nUser: ' + user_message}]})

    def stream_chat():
        try:
            response_stream = client.models.generate_content_stream(
                model="gemini-3.1-pro-preview",
                contents=messages
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text.replace("**", "").replace("*", "")
        except Exception as e:
            yield f"Error: {str(e)}"

    return Response(stream_chat(), mimetype='text/plain')

@app.route('/mitigate', methods=['POST'])
def mitigate():
    data = request.get_json()
    audit_context = data.get('context', {})
    
    attr = audit_context.get('sensitive_attr')
    label = audit_context.get('label_col')
    favorable_label = audit_context.get('favorable_label', 1)
    di = audit_context.get('disparate_impact', 'Unknown')

    prompt = f"""You are an AI fairness data engineer. 
Write a complete, standalone Python script to mitigate bias in a CSV dataset using the `aif360` library.

Dataset Context:
- Sensitive Attribute: '{attr}'
- Target Label: '{label}'
- Favorable Outcome Value: {favorable_label}
- Current Disparate Impact: {di}

The Python script must strictly follow these steps to avoid AIF360 errors:
1. Import pandas, tempfile, and required AIF360 modules (BinaryLabelDataset, Reweighing).
2. Use tempfile.gettempdir() to find the system temp folder, then load the file at that path with 'dataset.csv' filename.
3. Replace '?' strings with pandas NA and drop missing values.
4. If '{attr}' or '{label}' are text/strings, map them to 0 and 1.
5. Use pd.get_dummies(drop_first=True) on the dataframe to convert all remaining text columns to numbers.
6. Force the entire dataframe to float using df = df.astype(float).
7. Convert the dataframe into an AIF360 BinaryLabelDataset.
8. Apply the Reweighing algorithm.
9. Extract the newly calculated instance weights and add them as a new column 'fair_weights'.
10. Save the mitigated dataframe to the temp folder as 'mitigated_dataset.csv' (use os.path.join(tempfile.gettempdir(), 'mitigated_dataset.csv')).

IMPORTANT: Reply ONLY with valid, well-commented Python code. 
Do not use Markdown formatting like ```python or ```. Do not add any conversational text before or after the code.
"""

    def stream_response():
        try:
            # THIS IS THE MAGIC: True streaming directly from the SDK
            response_stream = client.models.generate_content_stream(
                model='gemini-3.1-pro-preview',
                contents=prompt
            )
            
            for chunk in response_stream:
                if chunk.text:
                    # Clean up any stubborn markdown ticks on the fly
                    clean_text = chunk.text.replace("```python\n", "").replace("```python", "").replace("```", "")
                    yield clean_text
        except Exception as e:
            yield f"\nERROR: {str(e)}"
    
    return Response(stream_response(), mimetype='text/plain')


@app.route('/save_result', methods=['POST'])
def save_result():
    data = request.json
    db.collection('audits').add({
        'user_id': data['user_id'],
        'di_score': data['di_score'],
        'attribute': data['attribute'],
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    return {"status": "saved"}

@app.route('/download-mitigated')
def download_mitigated():
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, 'mitigated_dataset.csv')
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return "File not found. Please process the data first.", 404

if __name__ == '__main__':
    # Cloud Run provides the PORT environment variable. Default to 8080 if running locally.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
