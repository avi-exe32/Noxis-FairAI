

<div align="center">

<img width="500"  alt="Noxis" src="https://github.com/user-attachments/assets/3fc2d145-846a-4c35-8c34-70bd7a71c2bf" />

#  Noxis — AI Fairness Auditor

### *Detect. Explain. Fix. AI Bias — Before It Harms.*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gemini](https://img.shields.io/badge/Gemini-3.1_Pro-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![AIF360](https://img.shields.io/badge/IBM-AIF360-052FAD?style=for-the-badge)](https://aif360.mybluemix.net/)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Cloud Run](https://img.shields.io/badge/Google_Cloud-Cloud_Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge)](https://shap.readthedocs.io/)

<br/>

> *"Algorithms shouldn't inherit our biases. Noxis ensures they don't."*

<br/>

**[▶️ Watch Demo Video](YOUR_YOUTUBE_LINK) · [🌐 Live Demo](YOUR_WEBSITE_LINK) · [🚀 Run Locally](#1-clone-the-repo) · [🧪 Test Datasets](#option-a--built-in-sample-files)**

<br/>

**Built for [SDG 10: Reduced Inequalities] & [SDG 16: Peace, Justice & Strong Institutions]**

</div>

---

## 🌍 The Problem

Every day, AI models quietly shape human lives — deciding who gets a loan, who gets hired, who gets parole. These models are trained on historical data that carries decades of systemic bias. The model learns it. The model repeats it. Nobody notices.

**Noxis changes that.**

<div align="right"><a href="#️⃣-noxis--ai-fairness-auditor">↑ Back to top</a></div>

---

## 🔍 What is Noxis?

Noxis is an **enterprise-grade AI bias auditing platform**. Upload your dataset and your trained model. In seconds, Noxis tells you:

- **Is your AI discriminating?** And against whom?
- **By how much?** With an A–F fairness grade and industry-standard metrics
- **Why?** With a Gemini AI explanation in plain English
- **How do you fix it?** With a one-click auto-generated Python mitigation script

Think of it as **VirusTotal — but for AI bias.**

---

## ✨ Features

### 🔬 Deep Bias Detection (IBM AIF360)
Four industry-standard fairness metrics computed instantly:

| Metric | What It Measures | Fair Range |
|---|---|---|
| **Disparate Impact** | Are outcome rates proportional across groups? | 0.8 – 1.2 |
| **Statistical Parity Difference** | Do groups receive favorable outcomes equally? | -0.1 to 0.1 |
| **Equal Opportunity Difference** | Are true positive rates equal? | near 0 |
| **Average Odds Difference** | Are prediction errors balanced? | near 0 |

Every audit receives an **A–F Fairness Grade** (scored out of 100) so stakeholders understand risk at a glance.

---

### 🤖 Gemini AI Copilot
- Translates raw fairness math into **plain English** instantly
- **Streaming chat assistant** that knows your exact audit results — ask it anything
- Context-aware, multi-turn conversation with your full dashboard data

---

### ⚡ 1-Click Python Mitigation
- Gemini writes a **complete, runnable Python script** using AIF360's Reweighing algorithm
- Mathematically offsets bias by adjusting instance weights — no retraining required
- Download the fixed dataset (`mitigated_dataset.csv`) directly from the UI

---

### 📊 Interactive Visualizations
- **SHAP Feature Importance** — reveals which features drove the model's decisions (dark mode, professional output)
- **4 default charts** with editable chart types and attributes
- **Custom Chart Builder** — search chart types, pick any axis, add unlimited charts to your dashboard
- **Full PDF export** of the entire audit report

---

### ⚖️ Legal & Compliance Hub
Automatically cross-references your bias metrics against real international law:
- 🇺🇸 **US EEOC 4/5ths Rule** — employment discrimination standard
- 🗽 **NYC Local Law 144** — automated hiring tool regulation
- 🇪🇺 **EU AI Act** — high-risk AI system compliance

---

### 🔥 Adversarial Red Teaming
Acts as an **offensive security tool**. Gemini generates 3 realistic adversarial personas from the disadvantaged group who would be unfairly rejected by your model despite being highly qualified — exposing hidden vulnerabilities like zip-code proxies for race.

---

### 👤 Human Impact Simulator
Puts a face to the numbers. Generates **3 empathetic, realistic case studies** of real people from the disadvantaged group who would be hurt by this bias — making the data human for executives and compliance officers.

---

### 🧪 CEO Simulation Lab
An executive-ready interactive sandbox:
- **What-If Sliders** — drag approval rates to see live grade changes
- **CEO Strategy Board** — real-time Chart.js scatter plot showing the accuracy vs. fairness tradeoff
- Helps executives make data-driven decisions about fairness vs. performance

---

### 🌐 Live Data Connect
Fetch and audit **live CSVs directly from a raw GitHub or public API URL** — no file upload required. Paste a link, audit instantly.

---

### 🔐 Firebase Audit History
- Google Sign-In authentication
- Every audit saved to **Firestore** with timestamp, attribute, and DI score
- Full audit log accessible across sessions — 20 most recent audits

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JS, Chart.js |
| **Backend** | Python 3.9+, Flask |
| **Bias Detection** | IBM AIF360, Scikit-learn |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **AI / LLM** | Google Gemini 3.1 Pro via Vertex AI Python SDK |
| **Auth & Database** | Firebase Authentication + Cloud Firestore |
| **Deployment** | Google Cloud Run (Docker container) |
| **Visualization** | Chart.js, Matplotlib (dark mode SHAP plots) |

<div align="right"><a href="#️⃣-noxis--ai-fairness-auditor">↑ Back to top</a></div>

---

## 🚀 Local Setup

### Prerequisites
- Python 3.9+
- A Google Cloud project with **Vertex AI API** enabled
- A Firebase project with **Authentication** and **Firestore** enabled
- `gcloud` CLI installed and authenticated

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/noxis.git
cd noxis
```

### 2. Install dependencies
```bash
pip install flask pandas numpy scikit-learn aif360 shap matplotlib \
            google-genai firebase-admin python-dotenv requests
```

### 3. Firebase credentials
Download your Firebase service account key from:
> Firebase Console → Project Settings → Service Accounts → Generate New Private Key

Save it as `fb-key.json` in the root of the project.

### 4. Google Cloud / Vertex AI authentication
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Then update `app.py` lines 45–48 with your project ID:
```python
client = genai.Client(
    vertexai=True,
    project='YOUR_PROJECT_ID',
    location='global'
)
```

### 5. Run
```bash
python app.py
```

Open **http://localhost:8080** 🎉

<div align="right"><a href="#️⃣-noxis--ai-fairness-auditor">↑ Back to top</a></div>

---

## ☁️ Cloud Deployment (Google Cloud Run)

### 1. Create a `Dockerfile` in the root
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
```

### 2. Create `requirements.txt`
```
flask
pandas
numpy
scikit-learn
aif360
shap
matplotlib
google-genai
firebase-admin
python-dotenv
requests
gunicorn
```

### 3. Set Firebase credentials as environment variable
Since `fb-key.json` can't be uploaded to Cloud Run directly, use an env var instead:
```
Cloud Run Console → Edit & Deploy New Revision → Environment Variables
Key:   FIREBASE_CONFIG_JSON
Value: (paste the entire contents of fb-key.json as a single-line JSON string)
```
> The app already handles this automatically — it checks for `FIREBASE_CONFIG_JSON` first, then falls back to `fb-key.json` for local development.

### 4. Deploy
```bash
gcloud run deploy noxis \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

Cloud Run gives you a live HTTPS URL. ✅

<div align="right"><a href="#️⃣-noxis--ai-fairness-auditor">↑ Back to top</a></div>

---

## 🧪 Test Datasets

### Option A — Built-in sample files
Click **📥 Download Sample Assets** on the upload page to get:
- `biased_dataset.csv` — pre-built biased dataset
- `test_model.pkl` — a trained scikit-learn model ready to audit

Upload both and use these settings:
```
Sensitive Attr:    sex  (or gender, depending on sample)
Privileged Value:  Male
Target Label:      income
Favorable Label:   1
```

---

### Option B — Live Data Connect (paste URL directly)

**Adult Census Income — Gender Bias**
```
URL:               https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/adult.csv
Sensitive Attr:    sex
Privileged Value:  Male
Target Label:      income
Favorable Label:   1
```
Expected: Disparate Impact ~0.36, Grade F — strong gender bias detected.

---

**ProPublica COMPAS — Racial Bias**
```
URL:               https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
Sensitive Attr:    race
Privileged Value:  Caucasian
Target Label:      two_year_recid
Favorable Label:   0
```
Expected: Significant racial bias — the real dataset that sparked the global algorithmic fairness debate.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER (index.html)                                           │
│                                                                 │
│  Upload Page ──► Loading ──► Split View                         │
│                               ├── Chat Panel (Gemini stream)    │
│                               └── Report Tabs                   │
│                                    ├── Overview & Mitigation    │
│                                    ├── Visualizations           │
│                                    ├── Legal Hub                │
│                                    ├── Red Teaming              │
│                                    ├── Human Impact             │
│                                    ├── Simulation Lab           │
│                                    └── Audit Log                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP / Streaming responses
┌──────────────────────────────▼──────────────────────────────────┐
│  FLASK BACKEND (app.py)                                         │
│                                                                 │
│  POST /analyze      AIF360 metrics + SHAP + Gemini explain      │
│  POST /chat         Gemini streaming chat (context-aware)       │
│  POST /mitigate     Gemini writes Python mitigation script      │
│  POST /red_team     Gemini generates adversarial personas       │
│  POST /human_impact Gemini generates human impact cards         │
│  POST /proxy_csv    Fetches live CSVs from public URLs          │
│  GET  /download-mitigated   Download fixed dataset              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐    ┌────────▼───────┐   ┌───────▼──────┐
   │  Vertex AI  │    │  AIF360 + SHAP │   │   Firebase   │
   │  Gemini 3.1 │    │  Scikit-learn  │   │  Auth + DB   │
   └─────────────┘    └────────────────┘   └──────────────┘
```

---

## 📁 Project Structure

```
noxis/
├── app.py                   # Flask backend — all API routes
├── templates/
│   └── index.html           # Full frontend (single-page app)
├── static/
│   ├── logo.png             # Noxis logo
│   ├── biased_dataset.csv   # Sample dataset for testing
│   └── test_model.pkl       # Sample trained sklearn model
├── fb-key.json              # Firebase service account (gitignored!)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚠️ Environment Variables

| Variable | Where | Description |
|---|---|---|
| `FIREBASE_CONFIG_JSON` | Cloud Run env | Full `fb-key.json` contents as JSON string |
| `PORT` | Auto-set by Cloud Run | Server port (default: 8080) |

---

## 🌱 UN SDG Alignment

| Goal | How Noxis Contributes |
|---|---|
| **SDG 10 — Reduced Inequalities** | Detects and fixes AI systems that disproportionately harm marginalized groups in lending, hiring, and healthcare |
| **SDG 16 — Peace, Justice & Strong Institutions** | Promotes accountability and transparency in automated decision-making and helps institutions comply with AI fairness laws |

---

## 🗺️ Roadmap

- [ ] **CI/CD Integration** — GitHub Actions plugin to block biased models from reaching production
- [ ] **Multi-Attribute Auditing** — Detect intersectional bias (e.g., race × gender)
- [ ] **NLP & Vision Support** — Audit bias in text classifiers and image recognition models
- [ ] **Enterprise REST API** — Programmatic bias auditing for MLOps pipelines
- [ ] **Team Workspaces** — Collaborative audit dashboards for compliance teams
- [ ] **Scheduled Monitoring** — Auto-run audits on live production model endpoints

---

<div align="center">

**Built with ⚖️ by Team Noxis**

*Making AI fair for everyone.*

**[SDG 10: Reduced Inequalities] · [SDG 16: Peace & Justice]**

</div>
