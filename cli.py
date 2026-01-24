# cli.py
import os

from model_service import (
    build_payload,
    get_profit_optimal_threshold,
    get_threshold_curve,
    plot_expected_profit_curve,
    evaluate_on_holdout,
    plot_confusion_matrix_holdout,
    plot_learning_curve_accuracy,
)
from loan_assistant import generate_assistant_explanation


def get_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def get_int(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def get_str(prompt, default=None):
    raw = input(prompt).strip()
    if raw == "" and default is not None:
        return str(default)
    return raw


def normalize_user_instance(ui: dict) -> dict:
    def norm(s):
        return str(s).strip().lower()

    ui["gender"] = norm(ui["gender"])
    ui["education"] = norm(ui["education"])
    ui["home_ownership"] = norm(ui["home_ownership"])
    ui["loan_intent"] = norm(ui["loan_intent"])
    ui["loan_defaults"] = norm(ui["loan_defaults"])
    return ui


def run_analytics(profit_per_good_loan: float, loss_per_default: float) -> None:
    print("\n" + "=" * 60)
    print("MODEL ANALYTICS (OFFLINE EVALUATION)")
    print("=" * 60)
    print("Note: Analytics are computed on the original 20% holdout set (X_test/y_test),")
    print("not on the single applicant you enter later.\n")

    # 1) Threshold curve + plot (validation-based)
    print("Computing profit curve on validation set...")
    results_df, best_threshold, best_row = get_threshold_curve(
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default,
    )
    print(f"Profit-optimal threshold (validation): {best_threshold:.3f}")
    print("Best row (validation):")
    for k, v in best_row.items():
        print(f"  {k}: {v}")

    plot_expected_profit_curve(results_df, best_threshold)

    # 2) Holdout evaluation (20% test split)
    print("\nEvaluating on holdout test set (20%)...")
    metrics = evaluate_on_holdout(
        threshold=None,  # use profit-optimal threshold for these business knobs
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default,
    )

    print("\nHoldout results:")
    ordered = [
        "threshold_used",
        "expected_profit",
        "approval_rate",
        "tp", "fp", "tn", "fn",
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "log_loss",
    ]
    for k in ordered:
        if k in metrics:
            print(f"  {k}: {metrics[k]}")

    print("  confusion_matrix:", metrics.get("confusion_matrix"))

    plot_confusion_matrix_holdout(
        threshold=None,
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default,
    )

    # 3) Learning curve
    plot_learning_curve_accuracy(cv=5)


def main():
    # --- Business knobs ---
    print("\nBusiness settings (press Enter to use defaults)")
    profit_per_good_loan = get_float("Profit per good loan [default 10000]: ", default=10000)
    loss_per_default = get_float("Loss per default [default 80000]: ", default=80000)

    # --- Analytics (offline, uses stored validation + holdout) ---
    run_analytics(profit_per_good_loan=profit_per_good_loan, loss_per_default=loss_per_default)

    # --- Threshold selection for applicant decision ---
    best_threshold, _best_row = get_profit_optimal_threshold(
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default
    )
    print(f"\nProfit-optimal threshold (validation): {best_threshold:.2f}")
    use_best = get_str("Use this threshold for the applicant decision? [Y/n]: ", default="y").lower()

    threshold_override = None
    if use_best in ("n", "no"):
        threshold_override = get_float("Enter a threshold between 0 and 1 (e.g., 0.35): ")
        threshold_override = max(0.0, min(1.0, float(threshold_override)))

    # --- Application inputs (single applicant; NOT used for analytics) ---
    print("\nEnter application details (single applicant)")
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
    user_instance = normalize_user_instance(user_instance)

    payload = build_payload(
        user_instance=user_instance,
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default,
        threshold=threshold_override,
    )

    print("\n" + "=" * 60)
    print("LOAN ASSISTANT (APPLICANT EXPLANATION)")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. Assistant explanation may fail.\n")

    try:
        explanation = generate_assistant_explanation(payload)
        print(explanation)
    except Exception as e:
        print("Assistant generation failed.")
        print(str(e))

    # Optional: show key policy info
    thr_used = payload.get("prediction", {}).get("threshold_used", None)
    p1 = payload.get("prediction", {}).get("probability_class_1", None)
    decision = payload.get("prediction", {}).get("loan_status", None)

    print("\n(For reference)")
    if decision is not None:
        print(f"- Decision (loan_status): {decision}  (1=APPROVED, 0=NOT APPROVED)")
    if p1 is not None:
        print(f"- Estimated approval probability: {p1:.4f}")
    if thr_used is not None:
        print(f"- Threshold used: {float(thr_used):.2f}")
    print(f"- Profit per good loan: {profit_per_good_loan:.2f}")
    print(f"- Loss per default: {loss_per_default:.2f}")


if __name__ == "__main__":
    main()
