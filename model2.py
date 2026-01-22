import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

import shap
import dice_ml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
df = pd.read_csv(DATA_PATH)

rename_map = {
    "person_age": "age",
    "person_gender": "gender",
    "person_education": "education",
    "person_income": "income",
    "person_emp_exp": "emp_exp",
    "person_home_ownership": "home_ownership",
    "loan_amnt": "loan_amount",
    "loan_intent": "loan_intent",
    "loan_int_rate": "loan_rate",
    "loan_percent_income": "loan_perc_income",
    "cb_person_cred_hist_length": "credit_history",
    "credit_score": "credit_score",
    "previous_loan_defaults_on_file": "loan_defaults",
    "loan_status": "loan_status",
}

df = df.rename(columns=rename_map)

TARGET_COL = "loan_status"

feature_cols = [
    "age", "gender", "education", "income", "emp_exp", "home_ownership",
    "loan_amount", "loan_intent", "loan_rate", "loan_perc_income",
    "credit_history", "credit_score", "loan_defaults"
]

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_features = [
    "gender", "education", "home_ownership", "loan_intent", "loan_defaults"
]
numeric_features = [c for c in feature_cols if c not in categorical_features]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

model = GradientBoostingClassifier(random_state=42)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

clf.fit(X_train, y_train)

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a number.")

def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter an integer.")

def get_str(prompt):
    return input(prompt).strip()

user_instance = {
    "age": get_int("age: "),
    "gender": get_str("gender: "),
    "education": get_str("education: "),
    "income": get_float("income: "),
    "emp_exp": get_int("emp_exp (years): "),
    "home_ownership": get_str("home_ownership: "),
    "loan_amount": get_float("loan_amount: "),
    "loan_intent": get_str("loan_intent: "),
    "loan_rate": get_float("loan_rate: "),
    "loan_perc_income": get_float("loan_perc_income: "),
    "credit_history": get_int("credit_history: "),
    "credit_score": get_int("credit_score: "),
    "loan_defaults": get_str("loan_defaults (Yes/No): "),
}

X_test = pd.DataFrame([user_instance], columns=feature_cols)

pred = clf.predict(X_test)[0]

prob_class1 = None
if hasattr(clf, "predict_proba"):
    proba = clf.predict_proba(X_test)
    if proba.shape[1] == 2:
        prob_class1 = float(proba[0, 1])

X_train_transformed = clf.named_steps["preprocess"].transform(X_train)
X_test_transformed = clf.named_steps["preprocess"].transform(X_test)

ohe = clf.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([cat_feature_names, np.array(numeric_features)])

underlying_model = clf.named_steps["model"]
explainer = shap.TreeExplainer(underlying_model)

shap_values = explainer.shap_values(X_test_transformed)

if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values_to_show = shap_values[1]
else:
    shap_values_to_show = shap_values

shap_series = (
    pd.Series(shap_values_to_show.flatten(), index=all_feature_names)
      .sort_values(key=np.abs, ascending=False)
)

fixed_features = ["age", "gender", "loan_defaults"]

def _shap_fallback_on_feature_cols(shap_series, feature_cols, fixed_features):
    impact = {}
    for col in feature_cols:
        if col in fixed_features:
            continue
        if col in shap_series.index:
            impact[col] = float(abs(shap_series.loc[col]))
        else:
            pref = col + "_"
            vals = shap_series[shap_series.index.astype(str).str.startswith(pref)].abs()
            if len(vals) > 0:
                impact[col] = float(vals.max())
    if len(impact) == 0:
        return pd.Series(dtype=float)
    return pd.Series(impact).sort_values(ascending=False)

def _suggest_directional_changes(target_class, X_test, feature_cols, numeric_features, fixed_features, clf, df):
    x0 = X_test.iloc[0].copy()
    suggestions = []
    if not hasattr(clf, "predict_proba"):
        return pd.Series(dtype=float)

    def p1(row):
        return float(clf.predict_proba(pd.DataFrame([row], columns=feature_cols))[0, 1])

    base = p1(x0)

    for col in feature_cols:
        if col in fixed_features:
            continue

        if col in numeric_features:
            q_low = float(df[col].quantile(0.10))
            q_high = float(df[col].quantile(0.90))

            row_low = x0.copy()
            row_low[col] = q_low
            p_low = p1(row_low)

            row_high = x0.copy()
            row_high[col] = q_high
            p_high = p1(row_high)

            best_p = max(p_low, p_high)
            delta = best_p - base
            suggestions.append((col, delta))
        else:
            vals = [v for v in df[col].dropna().unique().tolist() if str(v) != str(x0[col])]
            best = base
            for v in vals[:50]:
                row = x0.copy()
                row[col] = v
                pv = p1(row)
                if pv > best:
                    best = pv
            delta = best - base
            suggestions.append((col, delta))

    return pd.Series({k: v for k, v in suggestions}).sort_values(ascending=False)

dice_impact_series = _shap_fallback_on_feature_cols(shap_series, feature_cols, fixed_features)

if dice_impact_series.empty:
    dice_impact_series = _suggest_directional_changes(1, X_test, feature_cols, numeric_features, fixed_features, clf, df)

shap_top_k = 10
dice_top_k = 10

shap_items = []
for feat, val in shap_series.head(shap_top_k).items():
    effect = "increases_class_1" if val > 0 else "decreases_class_1"
    shap_items.append({"feature": str(feat), "effect": effect, "value": float(val)})

dice_items = []
for feat, val in dice_impact_series.head(dice_top_k).items():
    dice_items.append({"feature": str(feat), "impact_score": float(val)})

api_payload = {
    "application_id": None,
    "model": {
        "name": "GradientBoostingClassifier",
        "version": "local",
        "target": TARGET_COL,
        "positive_class": 1
    },
    "prediction": {
        "loan_status": int(pred) if isinstance(pred, (int, np.integer)) else pred,
        "probability_class_1": prob_class1
    },
    "explanations": {
        "shap": {
            "top_drivers": shap_items,
            "units": "model_output_space",
            "note": "Values are local contributions for this single instance."
        },
        "dice_impact": {
            "goal": {
                "target_class": 1,
                "fixed_features": fixed_features
            },
            "ranked_levers": dice_items,
            "method": "impact-ranking",
            "note": "Impact scores rank which features most influence moving toward class 1; not guaranteed to flip outcome."
        }
    },
    "validation": {
        "normalized_features": user_instance,
        "warnings": []
    }
}

print(json.dumps(api_payload, indent=2))
