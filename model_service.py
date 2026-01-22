# model_service.py
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")

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

TARGET_COL = "loan_status"

feature_cols = [
    "age", "gender", "education", "income", "emp_exp", "home_ownership",
    "loan_amount", "loan_intent", "loan_rate", "loan_perc_income",
    "credit_history", "credit_score", "loan_defaults"
]

categorical_features = ["gender", "education", "home_ownership", "loan_intent", "loan_defaults"]


def _normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip + lowercase categorical columns to ensure training-serving consistency.
    Keeps NaNs as NaNs.
    """
    df = df.copy()
    for col in categorical_features:
        s = df[col]
        mask = s.notna()
        df.loc[mask, col] = s.loc[mask].astype(str).str.strip().str.lower()
    return df


def find_best_threshold_by_profit(
    y_true,
    y_proba,
    profit_per_good_loan: float,
    loss_per_default: float,
    thresholds=None
):
    """
    y_true: true labels (1 = good loan, 0 = bad loan)
    y_proba: predicted probability of class 1
    profit = TP * profit_per_good_loan - FP * loss_per_default
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    N = len(y_true)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        expected_profit = tp * profit_per_good_loan - fp * loss_per_default
        approval_rate = (tp + fp) / N

        rows.append({
            "threshold": float(t),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "approval_rate": float(approval_rate),
            "expected_profit": float(expected_profit),
        })

    results = pd.DataFrame(rows)
    best = results.loc[results["expected_profit"].idxmax()]
    return results, float(best["threshold"]), best


def get_profit_optimal_threshold(
    profit_per_good_loan: float = 10000.0,
    loss_per_default: float = 80000.0,
    thresholds=None
):
    """
    Compute the profit-optimal decision threshold using validation data stored on the model.
    Returns (best_threshold, best_row_dict).
    """
    # Pull stored validation artifacts from the trained pipeline
    y_val = getattr(clf, "_y_val_", None)
    val_proba = getattr(clf, "_val_proba_", None)

    if y_val is None or val_proba is None:
        return 0.5, {
            "note": "Validation probabilities not available; using default threshold 0.5",
            "profit_per_good_loan": float(profit_per_good_loan),
            "loss_per_default": float(loss_per_default),
        }

    _, best_threshold, best_row = find_best_threshold_by_profit(
        y_true=np.asarray(y_val),
        y_proba=np.asarray(val_proba),
        profit_per_good_loan=float(profit_per_good_loan),
        loss_per_default=float(loss_per_default),
        thresholds=thresholds,
    )

    return float(best_threshold), best_row.to_dict()


def train_pipeline(
    df: pd.DataFrame,
    *,
    profit_per_good_loan: float = 10000.0,
    loss_per_default: float = 80000.0,
    threshold_grid=None,
    random_state: int = 42
):
    df = df.rename(columns=rename_map)

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Holdout split (kept in case you want it later)
    X_train_all, _, y_train_all, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Validation split (for threshold selection)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=random_state, stratify=y_train_all
    )

    numeric_features = [c for c in feature_cols if c not in categorical_features]

    preprocess = Pipeline(steps=[
        ("normalize_cat", FunctionTransformer(_normalize_categoricals, validate=False)),
        ("encode", ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("num", "passthrough", numeric_features),
            ]
        )),
    ])

    model = GradientBoostingClassifier(random_state=random_state)
    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    # Fit on train split (not val)
    clf.fit(X_train, y_train)

    # --- Profit-optimal threshold on validation set ---
    if not hasattr(clf, "predict_proba"):
        clf.best_threshold_ = 0.5
        clf._y_val_ = None
        clf._val_proba_ = None
        clf.threshold_selection_ = {
            "note": "Model has no predict_proba; defaulted threshold to 0.5",
            "profit_per_good_loan": float(profit_per_good_loan),
            "loss_per_default": float(loss_per_default),
        }
    else:
        val_proba = clf.predict_proba(X_val)[:, 1]

        # ✅ Store validation artifacts so Streamlit/CLI can recompute threshold fast
        clf._y_val_ = np.asarray(y_val)
        clf._val_proba_ = np.asarray(val_proba)

        results, best_threshold, best_row = find_best_threshold_by_profit(
            y_true=np.asarray(y_val),
            y_proba=np.asarray(val_proba),
            profit_per_good_loan=float(profit_per_good_loan),
            loss_per_default=float(loss_per_default),
            thresholds=threshold_grid
        )
        clf.best_threshold_ = float(best_threshold)
        clf.threshold_selection_ = {
            "profit_per_good_loan": float(profit_per_good_loan),
            "loss_per_default": float(loss_per_default),
            "best_threshold": float(best_threshold),
            "best_row": best_row.to_dict(),
        }
        # Optionally keep curve: clf.threshold_curve_ = results

    return clf, df, numeric_features


# Train once on import
_df = pd.read_csv(DATA_PATH)
clf, df_full, numeric_features = train_pipeline(_df)


def build_payload(
    user_instance: dict,
    fixed_features=None,
    threshold: float | None = None,
    profit_per_good_loan: float = 10000.0,
    loss_per_default: float = 80000.0
) -> dict:
    """
    threshold:
      - None -> uses profit-optimal threshold computed from (profit_per_good_loan, loss_per_default)
      - float -> uses user-provided threshold (Streamlit slider)
    """
    if fixed_features is None:
        fixed_features = ["age", "gender", "loan_defaults"]

    X_test = pd.DataFrame([user_instance], columns=feature_cols)

    # Probability for class 1
    prob_class1 = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        if proba.shape[1] == 2:
            prob_class1 = float(proba[0, 1])

    # Compute profit-optimal default threshold for current business settings
    default_best_threshold, best_row = get_profit_optimal_threshold(
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default
    )

    threshold_used = float(default_best_threshold) if threshold is None else float(threshold)

    # Final decision using threshold on prob_class1
    if prob_class1 is None:
        pred = clf.predict(X_test)[0]
    else:
        pred = int(prob_class1 >= threshold_used)

    # --- SHAP in transformed feature space ---
    X_test_transformed = clf.named_steps["preprocess"].transform(X_test)

    encoder_ct = clf.named_steps["preprocess"].named_steps["encode"]
    ohe = encoder_ct.named_transformers_["cat"]
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

    shap_items = []
    for feat, val in shap_series.head(10).items():
        effect = "increases_class_1" if val > 0 else "decreases_class_1"
        shap_items.append({"feature": str(feat), "effect": effect, "value": float(val)})

    normalized_df = _normalize_categoricals(X_test)
    normalized_features = normalized_df.iloc[0].to_dict()

    api_payload = {
        "model": {
            "name": "GradientBoostingClassifier",
            "version": "local",
            "target": TARGET_COL,
            "positive_class": 1
        },
        "prediction": {
            "loan_status": int(pred) if isinstance(pred, (int, np.integer)) else pred,
            "probability_class_1": prob_class1,
            "threshold_used": threshold_used,
            "default_best_threshold": float(default_best_threshold),
        },
        "policy": {
            "profit_per_good_loan": float(profit_per_good_loan),
            "loss_per_default": float(loss_per_default),
            "threshold_best_row": best_row,
            "note": "default_best_threshold computed from validation using profit/loss."
        },
        "explanations": {
            "shap": {
                "top_drivers": shap_items,
                "units": "model_output_space",
                "note": "Values are local contributions for this single instance."
            }
        },
        "validation": {
            "normalized_features": normalized_features,
            "warnings": []
        }
    }
    return api_payload
