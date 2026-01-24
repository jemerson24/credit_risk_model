# app.py
import os
import streamlit as st

from model_service import build_payload, get_profit_optimal_threshold
from loan_assistant import generate_assistant_explanation

st.set_page_config(page_title="Loan Risk Assistant", page_icon="💳", layout="centered")

st.title("💳 Loan Application Assistant")
st.write(
    "Enter your application details below. After you submit, the model will estimate the outcome "
    "and the assistant will explain the key factors in plain language."
)

# ---- Feature config ----
DROPDOWN_OPTIONS = {
    "gender": ["female", "male", "other"],
    "education": ["high school", "associate", "bachelor", "master", "doctorate", "other"],
    "home_ownership": ["rent", "own", "mortgage", "other"],
    "loan_intent": ["education", "medical", "personal", "venture", "homeimprovement", "debtconsolidation", "other"],
    "loan_defaults": ["no", "yes"],
}

def display_label(s: str) -> str:
    return str(s).strip().upper()

def norm_cat(x: str) -> str:
    return str(x).strip().lower()

def format_probability(p):
    if p is None:
        return None
    return f"{p * 100:.2f}%"

def opt_index(options, default_value):
    try:
        return options.index(default_value)
    except ValueError:
        return 0


# ---- Sidebar controls (policy knobs) ----
with st.sidebar:
    st.subheader("Decision settings")

    # ✅ Shared across pages (optional): store in session_state
    if "profit_per_good_loan" not in st.session_state:
        st.session_state["profit_per_good_loan"] = 10000.0
    if "loss_per_default" not in st.session_state:
        st.session_state["loss_per_default"] = 80000.0

    profit_per_good_loan = st.number_input(
        "Profit per good loan",
        min_value=0.0,
        value=float(st.session_state["profit_per_good_loan"]),
        step=1000.0,
        help="Used to compute the profit-optimal default threshold."
    )
    st.session_state["profit_per_good_loan"] = float(profit_per_good_loan)

    loss_per_default = st.number_input(
        "Loss per default",
        min_value=0.0,
        value=float(st.session_state["loss_per_default"]),
        step=5000.0,
        help="Used to compute the profit-optimal default threshold."
    )
    st.session_state["loss_per_default"] = float(loss_per_default)

    # Compute the best threshold for current profit/loss
    best_threshold, best_row = get_profit_optimal_threshold(
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default
    )

    st.caption(f"Best threshold (profit-optimal): {best_threshold:.2f}")

    # Slider default should be the best threshold. To keep it stable across reruns,
    # store it in session_state when profit/loss changes.
    key = (round(float(profit_per_good_loan), 2), round(float(loss_per_default), 2))
    if st.session_state.get("policy_key") != key:
        st.session_state["policy_key"] = key
        st.session_state["threshold"] = float(best_threshold)

    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("threshold", best_threshold)),
        step=0.01,
        help="If probability of approval >= threshold, classify as approved."
    )
    st.session_state["threshold"] = float(threshold)

    st.divider()

    show_payload = st.checkbox("Show debug payload (JSON)", value=False)

    st.caption("Requires OPENAI_API_KEY in your environment for assistant explanations.")
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not found. The assistant step may fail.")


# ---- Input form ----
with st.form("loan_form"):
    st.subheader("Application details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
        income = st.number_input("Annual income", min_value=0.0, value=10000.0, step=1000.0)
        emp_exp = st.number_input("Employment experience (years)", min_value=0, max_value=60, value=2, step=1)
        loan_amount = st.number_input("Loan amount", min_value=0.0, value=10000.0, step=500.0)

    with col2:
        loan_rate = st.number_input("Loan interest rate (%)", min_value=0.0, max_value=100.0, value=17.0, step=0.1)
        loan_perc_income = st.number_input("Loan percent of income (0–1)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        credit_history = st.number_input("Credit history length (years)", min_value=0, max_value=80, value=3, step=1)
        credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=650, step=1)

    col3, col4 = st.columns(2)
    with col3:
        gender_raw = st.selectbox(
            "Gender",
            DROPDOWN_OPTIONS["gender"],
            format_func=display_label,
            index=opt_index(DROPDOWN_OPTIONS["gender"], "female"),
        )
        education_raw = st.selectbox(
            "Education",
            DROPDOWN_OPTIONS["education"],
            format_func=display_label,
            index=opt_index(DROPDOWN_OPTIONS["education"], "bachelor"),
        )
        home_ownership_raw = st.selectbox(
            "Home ownership",
            DROPDOWN_OPTIONS["home_ownership"],
            format_func=display_label,
            index=opt_index(DROPDOWN_OPTIONS["home_ownership"], "rent"),
        )

    with col4:
        loan_intent_raw = st.selectbox(
            "Loan intent",
            DROPDOWN_OPTIONS["loan_intent"],
            format_func=display_label,
            index=opt_index(DROPDOWN_OPTIONS["loan_intent"], "education"),
        )
        loan_defaults_raw = st.selectbox(
            "Previous loan defaults",
            DROPDOWN_OPTIONS["loan_defaults"],
            format_func=display_label,
            index=opt_index(DROPDOWN_OPTIONS["loan_defaults"], "no"),
        )

    submitted = st.form_submit_button("Submit")


# ---- Prediction + Assistant ----
if submitted:
    user_instance = {
        "age": int(age),
        "gender": norm_cat(gender_raw),
        "education": norm_cat(education_raw),
        "income": float(income),
        "emp_exp": int(emp_exp),
        "home_ownership": norm_cat(home_ownership_raw),
        "loan_amount": float(loan_amount),
        "loan_intent": norm_cat(loan_intent_raw),
        "loan_rate": float(loan_rate),
        "loan_perc_income": float(loan_perc_income),
        "credit_history": int(credit_history),
        "credit_score": int(credit_score),
        "loan_defaults": norm_cat(loan_defaults_raw),
    }

    with st.spinner("Running model prediction..."):
        payload = build_payload(
            user_instance=user_instance,
            threshold=float(threshold),
            profit_per_good_loan=float(profit_per_good_loan),
            loss_per_default=float(loss_per_default),
        )

    pred = payload.get("prediction", {}).get("loan_status", None)
    p1 = payload.get("prediction", {}).get("probability_class_1", None)
    p_str = format_probability(p1)

    st.subheader("Model result")
    if pred is None:
        st.error("No prediction returned.")
    else:
        if int(pred) == 1:
            st.success("Congratulations! Your loan has been approved!")
        else:
            st.error("Unfortunately, your loan application has been rejected.")

        if p_str is not None:
            st.caption(f"Estimated chance of approval: {p_str}  |  Threshold used: {float(threshold):.2f}")

    st.subheader("Loan Assistant Recommendation")
    try:
        with st.spinner("Generating assistant explanation..."):
            explanation = generate_assistant_explanation(payload)
        st.write(explanation)
    except Exception as e:
        st.error("Assistant generation failed.")
        st.code(str(e))

    st.info("To view overall model performance on the 20% holdout set (X_test/y_test), open **Model Analytics** from the sidebar.")

    if show_payload:
        with st.expander("Debug payload (JSON)"):
            st.json(payload)
