# DataForge: Automated Insights Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Native%20UI-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)
![AI](https://img.shields.io/badge/Generative%20AI-Google%20Gemini-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**DataForge: Automated Insights Platform** is a comprehensive, end-to-end **AI Data Science & Business Analytics Platform**. 

Moving far beyond standard AutoML, this platform is engineered for rigorous business intelligence, statistical analysis, hypothesis testing, and explainable AI (XAI). It empowers data analysts and business stakeholders to seamlessly transition from raw data ingestion to executive reporting without writing a single line of code.

---

## 🌟 Executive Summary

Designed with the rigorous analytical standards of top-tier financial institutions and enterprise data teams in mind, this platform automates the entire data science lifecycle. It features a 10-module sequential architecture that handles Data Quality, Exploratory Data Analysis (EDA), advanced Statistical Hypothesis Testing, automated Feature Engineering, Multi-threaded AutoML, SHAP-based Explainability, and AI-driven business insights powered by Google Gemini.

## 🏗️ The 10-Module Architecture

The application is structured into a logical, sequential workflow:

1. **Data Ingestion:** Upload CSV/Excel files and define target optimization metrics.
2. **Exploratory Data Analysis (EDA):** Automated data profiling, univariate distributions, and bivariate relationships with interactive Plotly graphs.
3. **Statistics & Hypothesis Testing:** Formal distribution testing (Shapiro-Wilk) and a smart Hypothesis Testing engine that automatically runs the correct test (t-test, ANOVA, Chi-Square, Pearson/Spearman) and outputs strict business decisions (Reject/Accept H₀).
4. **Business Analytics:** Interactive dashboards featuring dynamic KPI cards and Deep Dive Analytics (Bar charts, Pie charts, Histograms, 3D Scatters, and Treemaps).
5. **Feature Engineering:** UI-driven configuration for missing value imputation, scaling (Standard/Robust), categorical encoding, and outlier clipping.
6. **AutoML Engine:** Multi-threaded training pipeline supporting Random Forests, XGBoost, LightGBM, and Gradient Boosting. Automatically evaluates and ranks models on a unified leaderboard.
7. **Explainable AI (XAI):** Unbox the "black box" models with Global Feature Importance, dense SHAP Summary Plots, and interactive SHAP Dependence Plots.
8. **Prediction Engine:** Batch score new, unseen datasets using the winning trained pipeline and download the predictions.
9. **AI Business Analyst (Chatbot):** Powered by **Google Gemini 2.5 Flash**, chat directly with your dataset and model metrics to extract proactive business recommendations and insights.
10. **Executive Reporting:** Automatically compile your data profiling, SHAP plots, model performance, and AI-generated insights into a downloadable, professional PDF report.

---

## 🛠️ Technology Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (with custom CSS injection for a premium, dark-mode React-style aesthetic)
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Plotly Express, Matplotlib
- **Statistical Analysis:** SciPy, Statsmodels
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
- **Explainable AI:** SHAP (SHapley Additive exPlanations)
- **Generative AI / LLM:** Google Gemini API (`langchain-google-genai`)
- **PDF Generation:** ReportLab

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/analytics-copilot-ai.git
cd analytics-copilot-ai
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
To utilize the AI Chatbot and automated AI insights for the PDF report, you need a Google Gemini API Key.
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```
The platform will automatically launch in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
analytics-copilot-ai/
├── .streamlit/                 # Global Streamlit config (Dark Theme)
├── app.py                      # Main landing page
├── theme.py                    # Custom CSS injection for premium sidebar UI
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API Keys)
├── pages/                      # The 10-Module Workflow
│   ├── 1_Upload.py
│   ├── 2_EDA.py
│   ├── 3_Statistics_Testing.py
│   ├── 4_Business_Analytics.py
│   ├── 5_Feature_Engineering.py
│   ├── 6_AutoML.py
│   ├── 7_Explainability.py
│   ├── 8_Prediction.py
│   ├── 9_Chatbot.py
│   └── 10_Report.py
└── backend/                    # Core Python Logic
    ├── pipeline.py             # ML Training pipelines
    ├── explainability.py       # SHAP integration
    └── report_generator.py     # PDF generation engine
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/analytics-copilot-ai/issues).

## 📝 License
This project is licensed under the MIT License.
