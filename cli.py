# cli.py
import numpy as np

from model_service import build_payload, get_profit_optimal_threshold
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


def main():
    # --- Business knobs ---
    print("\nBusiness settings (press Enter to use defaults)")
    profit_per_good_loan = get_float("Profit per good loan [default 10000]: ", default=10000)
    loss_per_default = get_float("Loss per default [default 80000]: ", default=80000)

    best_threshold, best_row = get_profit_optimal_threshold(
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default
    )

    print(f"\nProfit-optimal threshold (computed from validation): {best_threshold:.2f}")
    use_best = get_str("Use this threshold? [Y/n]: ", default="y").lower()

    threshold_override = None
    if use_best in ("n", "no"):
        threshold_override = get_float("Enter a threshold between 0 and 1 (e.g., 0.35): ")
        # basic clamp
        threshold_override = max(0.0, min(1.0, threshold_override))

    # --- Application inputs ---
    print("\nEnter application details")
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

    # Build payload using chosen profit/loss and threshold (auto or override)
    payload = build_payload(
        user_instance=user_instance,
        profit_per_good_loan=profit_per_good_loan,
        loss_per_default=loss_per_default,
        threshold=threshold_override
    )

    # Print only assistant (no JSON)
    print("\n" + "=" * 60)
    print("LOAN ASSISTANT")
    print("=" * 60)

    explanation = generate_assistant_explanation(payload)
    print(explanation)

    # Optional: show key policy info
    thr_used = payload.get("prediction", {}).get("threshold_used", None)
    p1 = payload.get("prediction", {}).get("probability_class_1", None)
    if p1 is not None and thr_used is not None:
        print("\n(For reference)")
        print(f"- Probability of approval (class 1): {p1:.4f}")
        print(f"- Threshold used: {thr_used:.2f}")
        print(f"- Profit per good loan: {profit_per_good_loan:.2f}")
        print(f"- Loss per default: {loss_per_default:.2f}")


if __name__ == "__main__":
    main()
