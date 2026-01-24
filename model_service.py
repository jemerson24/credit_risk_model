# model_service.py
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    ConfusionMatrixDisplay,
)

import shap
import matplotlib.pyplot as plt


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

    # ✅ 20% HOLDOUT TEST SET
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # ✅ VALIDATION SET (for threshold selection only)
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

    clf.fit(X_train, y_train)

    # Store splits for analytics
    clf._X_train_ = X_train
    clf._y_train_ = y_train
    clf._X_val_ = X_val
    clf._y_val_raw_ = y_val
    clf._X_test_ = X_test
    clf._y_test_ = y_test

    # Threshold selection on validation
    if hasattr(clf, "predict_proba"):
        val_proba = clf.predict_proba(X_val)[:, 1]
        clf._y_val_ = np.asarray(y_val)
        clf._val_proba_ = np.asarray(val_proba)

        _, best_threshold, best_row = find_best_threshold_by_profit(
            y_true=clf._y_val_,
            y_proba=clf._val_proba_,
            profit_per_good_loan=profit_per_good_loan,
            loss_per_default=loss_per_default,
            thresholds=threshold_grid,
        )

        clf.best_threshold_ = float(best_threshold)
        clf.threshold_selection_ = best_row.to_dict()
    else:
        clf.best_threshold_ = 0.5
        clf._y_val_ = None
        clf._val_proba_ = None

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
    if fixed_features is None:
        fixed_features = ["age", "gender", "loan_defaults"]

    # ✅ SINGLE USER ROW
    X_user = pd.DataFrame([user_instance], columns=feature_cols)

    prob_class1 = None
    if hasattr(clf, "predict_proba"):
        prob_class1 = float(clf.predict_proba(X_user)[0, 1])

    default_best_threshold, best_row = get_profit_optimal_threshold(
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default
    )

    threshold_used = float(default_best_threshold) if threshold is None else float(threshold)
    pred = int(prob_class1 >= threshold_used) if prob_class1 is not None else int(clf.predict(X_user)[0])

    X_user_transformed = clf.named_steps["preprocess"].transform(X_user)

    encoder_ct = clf.named_steps["preprocess"].named_steps["encode"]
    ohe = encoder_ct.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([cat_feature_names, np.array(numeric_features)])

    explainer = shap.TreeExplainer(clf.named_steps["model"])
    shap_values = explainer.shap_values(X_user_transformed)
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap_series = (
        pd.Series(shap_values.flatten(), index=all_feature_names)
        .sort_values(key=np.abs, ascending=False)
    )

    shap_items = [
        {
            "feature": str(f),
            "effect": "increases_class_1" if v > 0 else "decreases_class_1",
            "value": float(v),
        }
        for f, v in shap_series.head(10).items()
    ]

    normalized_features = _normalize_categoricals(X_user).iloc[0].to_dict()

    return {
        "prediction": {
            "loan_status": int(pred),
            "probability_class_1": prob_class1,
            "threshold_used": threshold_used,
        },
        "explanations": {
            "shap": {"top_drivers": shap_items}
        },
        "validation": {
            "normalized_features": normalized_features
        }
    }


# -------------------------
# Analytics (CLI-safe)
# -------------------------

def evaluate_on_test(
    threshold: float | None = None,
    profit_per_good_loan: float = 10000.0,
    loss_per_default: float = 80000.0
) -> dict:
    """
    Evaluate the trained pipeline on the stored holdout 20% split (X_test/y_test).

    threshold:
      - None: uses profit-optimal threshold computed on validation for given profit/loss
      - float: uses provided threshold

    Returns dict with metrics + confusion matrix + profit + approval_rate.
    """
    X_test = getattr(clf, "_X_test_", None)
    y_test = getattr(clf, "_y_test_", None)
    if X_test is None or y_test is None:
        raise RuntimeError("Holdout split not available. Ensure train_pipeline stores _X_test_/_y_test_.")

    default_best_threshold, _ = get_profit_optimal_threshold(
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default
    )
    thr = float(default_best_threshold) if threshold is None else float(threshold)

    if not hasattr(clf, "predict_proba"):
        y_pred = clf.predict(X_test)
        y_proba = None
    else:
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # business metrics
    expected_profit = float(tp) * float(profit_per_good_loan) - float(fp) * float(loss_per_default)
    approval_rate = float((tp + fp) / len(y_test))

    out = {
        "threshold_used": thr,
        "profit_per_good_loan": float(profit_per_good_loan),
        "loss_per_default": float(loss_per_default),
        "expected_profit": float(expected_profit),
        "approval_rate": float(approval_rate),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),

        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        out["log_loss"] = float(log_loss(y_test, y_proba))
    else:
        out["roc_auc"] = None
        out["log_loss"] = None

    return out


def get_threshold_curve(
    profit_per_good_loan: float = 10000.0,
    loss_per_default: float = 80000.0,
    thresholds=None
):
    """
    Validation-set threshold curve for expected profit.
    Uses stored validation labels/probabilities on clf.
    Returns: (results_df, best_threshold, best_row_dict)
    """
    y_val = getattr(clf, "_y_val_", None)
    val_proba = getattr(clf, "_val_proba_", None)

    if y_val is None or val_proba is None:
        raise RuntimeError("Validation artifacts not available on clf (_y_val_/_val_proba_).")

    results_df, best_threshold, best_row = find_best_threshold_by_profit(
        y_true=np.asarray(y_val),
        y_proba=np.asarray(val_proba),
        profit_per_good_loan=float(profit_per_good_loan),
        loss_per_default=float(loss_per_default),
        thresholds=thresholds,
    )
    return results_df, float(best_threshold), best_row.to_dict()


def evaluate_on_holdout(
    threshold: float | None = None,
    profit_per_good_loan: float = 10000.0,
    loss_per_default: float = 80000.0
) -> dict:
    """
    Alias for evaluate_on_test() to match the Streamlit analytics page naming.
    Evaluates performance on the 20% holdout (X_test/y_test).
    """
    return evaluate_on_test(
        threshold=threshold,
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default,
    )
