# 💳 Credit Approval Model

A machine learning–driven **Credit Approval System** designed to evaluate credit applications, optimize lender profitability, and reduce bias in decision-making.

The system integrates a **Gradient Boosting classification model**, cost–benefit–driven decision thresholding, advanced explainability techniques, and a **GenAI-powered Loan Assistant** to deliver automated, transparent, and user-centric credit decisions.

---

## 🧾 1. Project Description

This project implements a **Credit Approval Model** that predicts whether a loan application should be approved or denied based on applicant characteristics.

Key capabilities include:
- Predicting a **probability of default**
- Allowing lenders to control risk via a configurable decision threshold
- Providing model analytics to support business decisions
- Delivering applicant-facing explanations and recommendations through GenAI

The system is designed to reflect **real-world lending constraints**, where maximizing expected profit while minimizing costly errors is critical.

---

## 🎯 2. Motivation and Business Context

Loan application decisions are traditionally made through **manual, person-to-person processes**, which can introduce:
- Human bias
- Inconsistent decision criteria
- Operational inefficiencies
- Limited transparency for applicants

This project demonstrates how an **ML-driven credit approval pipeline** can:
- Standardize and scale decision-making
- Reduce bias and human error
- Optimize outcomes using cost–benefit analysis
- Provide applicants with **on-demand, explainable feedback** via GenAI

---

## 🧠 3. Modeling Approach and Decision Framework

### 3.1 🤖 Machine Learning Model
- **Algorithm:** GradientBoostingClassifier (scikit-learn)
- **Task:** Binary classification (Approve / Deny)
- **Output:** Estimated probability of default

### 3.2 📈 Decision Threshold Optimization
Rather than relying on a fixed probability cutoff, the system enables:
- Adjustable decision thresholds
- Explicit control over acceptable default risk
- Optimization using **Expected Profit vs. Threshold** analysis

This approach aligns model outputs directly with business objectives.

---

## 📊 4. Model Evaluation and Analytics

The application provides a comprehensive suite of analytical tools, including:

- Expected Profit vs. Decision Threshold
- Confusion Matrix
- Learning Curve
- False Positive and False Negative analysis

The optimization objective is to:
> **Minimize false positives while approving the maximum number of profitable loans**

---

## 🤖✨ 5. GenAI-Powered Loan Assistant

Following a credit decision, applicants receive **personalized consultation** from a **GenAI Loan Assistant** powered by the OpenAI SDK.

The assistant:
- Explains approval or denial decisions in natural language
- Identifies features that can be improved
- Respects immutable or protected attributes (e.g., gender)
- Provides actionable, constraint-aware recommendations

This component enhances transparency, user trust, and accessibility while avoiding black-box decisioning.

---

## 🚀 6. Application Access and Execution

The system is deployed as an interactive **Streamlit web application**.

**Access the application here:**  
👉 https://creditapprovalmodel-punnd7v6ievhdxc5gykbtt.streamlit.app

No local setup is required to explore the application.

---

## 🛠️ 7. Technical Capabilities

This project demonstrates end-to-end applied machine learning with a strong emphasis on **GenAI-enabled explainability and decision support**.

### 7.1 📐 Machine Learning and Optimization
- Gradient Boosting classification (scikit-learn)
- Mixed numerical and categorical feature handling
- Cost–benefit–based decision threshold tuning
- Expected Profit–driven evaluation strategy

### 7.2 🔍 Explainability and Transparency
- **SHAP** for global and local feature importance
- **DiCE (Diverse Counterfactual Explanations)** for actionable counterfactual insights
- Error analysis via confusion matrices and learning curves

### 7.3 🧠 GenAI Integration
- **OpenAI SDK**–powered natural-language explanations
- Personalized, constraint-aware guidance for applicants
- Human-centered interpretation of ML outputs

### 7.4 🧩 Systems and Application Design
- Streamlit-based interactive front end
- Modular, API-style inference architecture
- **JSON-based input/output** for interoperability
- CLI utilities for batch inference and testing

---

## 🗂️ 8. Repository Structure

```text
credit_approval_model/
├── data/
│   └── loan_data.csv              # Credit application dataset
├── pages/
│   └── 1_Model_Analytics.py       # Streamlit model analytics page
├── cli.py                         # Command-line inference interface
├── Loan_Application.py            # Core loan application logic
├── loan_assistant.py              # GenAI-powered loan consultation assistant
├── model_service.py               # Model loading and prediction service
├── credit_approval_model.ipynb    # EDA, training, and evaluation
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md
```

---

⚠️ 9. Limitations and Considerations
- Limited representation of certain features (e.g., income, age, percent income) may cause some variables, such as loan interest rate, to carry disproportionate influence.
- Decision thresholds are optimized on historical data and may require recalibration as economic conditions change.
- Class imbalance may affect detection of rare default outcomes.
- Important risk factors (e.g., credit history length, employment stability) are not included.
- The model is not fully explainable, and GenAI outputs should be treated as advisory.
- Fairness metrics and human-in-the-loop review are not explicitly enforced.

---

📄 10. License

This project is licensed under the MIT License.

Note: This project is intended for educational and demonstration purposes.
