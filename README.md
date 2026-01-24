# Credit Approval Model

A machine learning–driven **Credit Approval System** that approves or denies credit applications based on customer criteria, while optimizing lender profitability and reducing bias in loan decisions.

The project combines a **Gradient Boosting Classifier**, cost-benefit decision analysis, and a **GenAI Loan Assistant** to provide both automated decisions and human-readable explanations for applicants.

---

## 📌 What Is This?

This project builds a **Credit Approval Model** that predicts whether a loan application should be approved or denied based on applicant features.

- The core model is a **GradientBoostingClassifier (scikit-learn)**
- Outputs a **probability of default**
- A configurable **decision threshold** allows lenders to control risk tolerance
- Includes advanced **model analytics** to support business decision-making
- After a decision is made, a **GenAI Loan Assistant** explains the result and suggests actionable improvements within allowed constraints

The system is designed for **real-world lending scenarios**, where maximizing profit and minimizing costly errors is critical.

---

## 🎯 Why Does It Exist?

Traditional loan applications are often handled **person-to-person**, which introduces:
- Human bias
- Inconsistent decision criteria
- Slower turnaround times
- Limited transparency for applicants

This project demonstrates how a **machine learning–based loan approval system** can:

- Standardize decision-making
- Reduce bias and human error
- Optimize profitability using cost-benefit analysis
- Provide applicants with **on-demand, explainable feedback** through a GenAI assistant

---

## 🧠 Model & Decision Strategy

### Machine Learning Model
- **Algorithm:** GradientBoostingClassifier (scikit-learn)
- **Problem Type:** Binary classification (Approve / Deny)
- **Output:** Probability of default

### Decision Threshold Optimization
Rather than using a fixed 0.5 cutoff, the business can:
- Adjust the **decision threshold**
- Control the probability of default they are willing to assume
- Trade off risk vs. loan volume using **Expected Profit**

---

## 📊 Model Analytics

The project includes multiple analytical tools to support decision-making:

- **Expected Profit vs. Threshold**
- **Confusion Matrix**
- **Learning Curve**
- **False Positive / False Negative analysis**

The primary optimization goal is to:
> **Reduce False Positives as much as possible while approving the maximum number of profitable loans**

---

## 🤖 GenAI Loan Assistant

Once a recommendation is made, the applicant receives **personalized consultation** from a **GenAI Loan Assistant**.

The assistant:
- Analyzes the loan application
- Explains *why* the application was approved or denied
- Identifies **features that can be improved**
- Respects **non-modifiable constraints** (e.g., gender or protected attributes)
- Provides actionable, understandable guidance

This bridges the gap between **black-box ML models** and **user trust**.

---

## 🖥 How to Run the App

The project is deployed as an interactive **Streamlit application**.

👉 **Launch the app here:**  
**[Streamlit App URL]**

No local setup required.

---

## ⚠️ Limitations

- The training data has limited coverage for features such as **income**, **age**, and **percent income**, which may cause certain variables (e.g., **loan interest rate**) to carry disproportionate weight in approval decisions.
- The decision threshold is optimized on historical data and may require **recalibration** as borrower behavior or economic conditions change.
- The dataset may exhibit **class imbalance**, potentially impacting the model’s ability to capture rare default outcomes.
- Important risk factors such as **credit history length** and **employment stability** are not included, limiting predictive accuracy.
- While feature importance and the GenAI Loan Assistant provide interpretability, the model is **not fully explainable**, and GenAI outputs should be treated as advisory.
- Fairness metrics and human review processes are not explicitly enforced; this project is intended for **educational and demonstration purposes** only.


## 🛠 Technical Features

- Gradient Boosting classification model
- Mixed numerical and categorical feature handling
- Cost-benefit–driven decision thresholding
- Model evaluation and visualization tools
- Streamlit front-end
- CLI utilities for inference
- Modular model service architecture
- GenAI-powered loan consultation assistant

---

## 📁 Repository Structure

```text
credit_approval_model/
├── data/
│   └── loan_data.csv              # Credit application dataset
├── pages/
│   └── 1_Model_Analytics.py       # Streamlit page for model analytics
├── cli.py                         # Command-line interface for inference
├── Loan_Application.py            # Core loan application logic
├── loan_assistant.py              # GenAI-powered loan consultation assistant
├── model_service.py               # Model loading and prediction service
├── credit_approval_model.ipynb    # EDA, model training, and evaluation
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md
```
