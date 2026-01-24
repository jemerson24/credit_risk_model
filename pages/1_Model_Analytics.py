# pages/1_Model_Analytics.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import model_service as ms  # lets us access ms.clf, stored splits, etc.
from model_service import (
    get_profit_optimal_threshold,
    get_threshold_curve,
    evaluate_on_holdout,  # alias -> evaluate_on_test in your model_service.py
)

from sklearn.model_selection import learning_curve

st.set_page_config(page_title="Model Analytics", page_icon="📊", layout="wide")

st.title("📊 Model Analytics")
st.write(
    "This page reports **offline performance** measured on the original **20% test dataset** "
    "(stored as `X_test/y_test` in `model_service.py`). "
    "It does **not** use applicant inputs from the Loan Application page."
)

# -------------------------
# Sidebar controls (no Run button here anymore)
# -------------------------
with st.sidebar:
    st.subheader("Business assumptions")

    # Optional: share profit/loss between pages via session_state
    if "profit_per_good_loan" not in st.session_state:
        st.session_state["profit_per_good_loan"] = 10000.0
    if "loss_per_default" not in st.session_state:
        st.session_state["loss_per_default"] = 80000.0

    profit_per_good_loan = st.number_input(
        "Profit per good loan",
        min_value=0.0,
        value=float(st.session_state["profit_per_good_loan"]),
        step=1000.0,
    )
    st.session_state["profit_per_good_loan"] = float(profit_per_good_loan)

    loss_per_default = st.number_input(
        "Loss per default",
        min_value=0.0,
        value=float(st.session_state["loss_per_default"]),
        step=5000.0,
    )
    st.session_state["loss_per_default"] = float(loss_per_default)

    st.divider()

    st.subheader("Threshold on test")

    best_threshold, _ = get_profit_optimal_threshold(
        profit_per_good_loan=float(profit_per_good_loan),
        loss_per_default=float(loss_per_default),
    )

    threshold_mode = st.radio(
        "Choose threshold",
        options=["Profit-optimal", "Manual"],
        index=0,
    )

    manual_threshold = st.slider(
        "Manual threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(best_threshold),
        step=0.01,
        disabled=(threshold_mode != "Manual"),
    )

    threshold_used = (
        float(best_threshold)
        if threshold_mode == "Profit-optimal"
        else float(manual_threshold)
    )

# -------------------------
# Main-page Run button (moved from sidebar)
# -------------------------
run = st.button("Run analytics", type="primary")

# -------------------------
# Helpers (plotting)
# -------------------------
def plot_profit_curve(results_df: pd.DataFrame, best_t: float, chosen_t: float):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        results_df["threshold"],
        results_df["expected_profit"],
        label="Expected profit",
    )
    ax.axvline(
        x=float(best_t),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Ideal threshold = {best_t:.2f}",
    )
    # show chosen threshold too (if different) but keep ideal red as requested
    if abs(float(chosen_t) - float(best_t)) > 1e-9:
        ax.axvline(
            x=float(chosen_t),
            linestyle=":",
            linewidth=2,
            label=f"Chosen threshold = {chosen_t:.2f}",
        )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected profit")
    ax.set_title("Expected Profit vs Threshold")
    ax.grid(True)
    ax.legend()
    return fig


def plot_confusion_heatmap(cm_2x2):
    """
    Model output format:
    [[TN, FP],
     [FN, TP]]

    Display format (matches standard diagram):
    [[TP, FN],
     [FP, TN]]
    """

    tn, fp = cm_2x2[0]
    fn, tp = cm_2x2[1]

    # Reorder for display
    cm_display = np.array([
        [tp, fn],   # Actual Positive
        [fp, tn],   # Actual Negative
    ], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(cm_display)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticklabels(["Actual Positive", "Actual Negative"])

    labels = [
        ["TP", "FN"],
        ["FP", "TN"]
    ]

    for i in range(2):        # rows = Actual
        for j in range(2):    # cols = Predicted
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{cm_display[i, j]}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig




def plot_learning_curve_fig(train_sizes, train_mean, train_std, val_mean, val_std):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training accuracy")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)

    ax.plot(train_sizes, val_mean, marker="o", label="CV accuracy")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

    ax.set_title("Learning Curve")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend()
    return fig


# -------------------------
# Main content
# -------------------------
if not run:
    st.info("Click **Run analytics** above to compute plots and metrics.")
    st.stop()

# 1) Threshold profit curve (VALIDATION)
with st.spinner("Computing expected profit curve..."):
    curve_df, curve_best_t, curve_best_row = get_threshold_curve(
        profit_per_good_loan=float(profit_per_good_loan),
        loss_per_default=float(loss_per_default),
    )

# 2) Test metrics (X_test / y_test)
with st.spinner("Evaluating on test ..."):
    metrics = evaluate_on_holdout(
        threshold=float(threshold_used),
        profit_per_good_loan=float(profit_per_good_loan),
        loss_per_default=float(loss_per_default),
    )

# 3) Learning curve (refits internally)
with st.spinner("Computing learning curve (this may take a bit)..."):
    X_train = getattr(ms.clf, "_X_train_", None)
    y_train = getattr(ms.clf, "_y_train_", None)

    if X_train is None or y_train is None:
        learning_curve_fig = None
        learning_curve_note = "Training split not available on clf."
    else:
        sizes, train_scores, val_scores = learning_curve(
            ms.clf,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        learning_curve_fig = plot_learning_curve_fig(sizes, train_mean, train_std, val_mean, val_std)
        learning_curve_note = None

# -------------------------
# Layout: KPIs + Plots
# -------------------------
st.subheader("Test metrics")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
k2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
k3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
k4.metric("F1", f"{metrics.get('f1', 0):.3f}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("ROC-AUC", "—" if metrics.get("roc_auc") is None else f"{metrics.get('roc_auc', 0):.3f}")
k6.metric("Log loss", "—" if metrics.get("log_loss") is None else f"{metrics.get('log_loss', 0):.3f}")
k7.metric("Approval rate", f"{metrics.get('approval_rate', 0) * 100:.1f}%")
k8.metric("Expected profit", f"{metrics.get('expected_profit', 0):,.0f}")

st.caption(
    f"Threshold used on test: **{metrics.get('threshold_used', threshold_used):.2f}** "
    f"(Ideal profit-optimal threshold from validation: **{curve_best_t:.2f}**)"
)

st.divider()

# Expected profit vs threshold FIRST (stacked layout)
st.subheader("Expected profit vs threshold")
fig_profit = plot_profit_curve(curve_df, best_t=curve_best_t, chosen_t=threshold_used)
st.pyplot(fig_profit, clear_figure=True)

with st.expander("What is the decision threshold?"):
    st.markdown(
        """
The **decision threshold** is the cutoff that turns the model’s predicted
probability into an **approve or reject** decision.

- Probability ≥ threshold → **approve**
- Probability < threshold → **reject**

A **higher** threshold is more conservative (fewer approvals, lower risk).  
A **lower** threshold is more lenient (more approvals, higher risk).

The *profit-optimal* threshold maximizes expected profit based on your
profit and loss assumptions.
"""
    )

with st.expander("Top 10 thresholds by expected profit (validation)"):
    st.dataframe(
        curve_df.sort_values("expected_profit", ascending=False).head(10).reset_index(drop=True),
        use_container_width=True,
    )

st.divider()

# Confusion matrix SECOND (not side-by-side)
st.subheader("Confusion matrix heatmap")
cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
fig_cm = plot_confusion_heatmap(cm)
st.pyplot(fig_cm, clear_figure=True)

with st.expander("What is the confusion matrix?"):
    st.markdown(
        """
The **confusion matrix** summarizes how the model’s predictions compare
to the true outcomes on the test dataset.

- **True Positive (TP):** Approved and the loan was repaid  
- **False Positive (FP):** Approved but the loan defaulted  
- **False Negative (FN):** Rejected but the loan would have been repaid  
- **True Negative (TN):** Rejected and the loan would have defaulted  

Rows represent the **actual outcome**, and columns represent the
**model’s prediction**.
"""
    )

with st.expander("Confusion matrix values"):
    st.write(
    "Displayed layout:\n"
    "- Rows = Actual\n"
    "- Columns = Predicted\n\n"
    "Top-left: True Positive (TP)\n"
    "Top-right: False Negative (FN)\n"
    "Bottom-left: False Positive (FP)\n"
    "Bottom-right: True Negative (TN)"
)
    st.json(cm)



st.divider()

st.subheader("Learning curve")
if learning_curve_fig is None:
    st.warning(learning_curve_note or "Learning curve not available.")
else:
    st.pyplot(learning_curve_fig, clear_figure=True)

with st.expander("What is the learning curve?"):
    st.markdown(
        """
The **learning curve** shows how the model’s performance changes as it is
trained on more data.

- **Training accuracy:** Performance on data the model has already seen  
- **Validation accuracy:** Performance on unseen data  

If both curves are low, the model is **underfitting**.  
If training is high but validation is much lower, the model is **overfitting**.

Learning curves help diagnose whether the model would benefit from
more data or a different level of complexity.
"""
    )


with st.expander("Raw test metrics (JSON)"):
    st.json(metrics)

