from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from google import genai
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
API_KEY = os.getenv('GEMINI_API_KEY')
# 2. Initialize the 2026 Client (No more genai.configure)
client = genai.Client(api_key=API_KEY)


def safe_metric(fn):
    """Safely call an AIF360 metric, returning None on failure."""
    try:
        val = fn()
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 4)
    except Exception:
        return None


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

            feature_cols = [c for c in df.columns if c not in [label_col, sensitive_attr]]
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

        ground_truth_ds = BinaryLabelDataset(
            df=df[[sensitive_attr, label_col]].copy(),
            label_names=[label_col],
            protected_attribute_names=[sensitive_attr],
            favorable_label=favorable_label,
            unfavorable_label=unfav_label
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
        for grp_val in [0, 1]:
            grp_df = df[df[sensitive_attr] == grp_val]
            grp_name = group_names.get(str(grp_val), str(grp_val))
            fav_rate = round(float((grp_df[label_col] == favorable_label).mean()), 4) if len(grp_df) else None
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

Max 150 words. Be direct."""
        
        gemini_resp = client.models.generate_content(
            model='gemini-3.1-pro-preview', 
            contents=prompt
        )
        explanation = gemini_resp.text

        return jsonify({
            'success': True,
            'disparate_impact': disparate_impact,
            'stat_parity': stat_parity,
            'eq_odds_diff': eq_odds_diff,
            'avg_odds_diff': avg_odds_diff,
            'pred_parity_diff': pred_parity_diff,
            'explanation': explanation,
            'sensitive_attr': sensitive_attr,
            'label_col': label_col,
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
        system_context = f"""You are Noxis, an AI bias auditor.
Audit results: attribute='{audit_context.get('sensitive_attr')}', label='{audit_context.get('label_col')}'
Disparate Impact: {audit_context.get('disparate_impact')} | Statistical Parity: {audit_context.get('stat_parity')}
Groups: {audit_context.get('group_stats')}
Answer the user's question about this audit in max 100 words. Be direct."""

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
            model='gemini-3-flash-preview',
            contents=messages
        )

        return jsonify({
            'success': True,
            'reply': response.text
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
