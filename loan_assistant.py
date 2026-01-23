# loan_assistant.py
import os
import json
from typing import Dict, Any, List, Optional

from openai import OpenAI

# You can override the model with an env var if you want:
# export OPENAI_MODEL="gpt-5.2-mini"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _safe_get(payload: Dict[str, Any], path: List[str], default=None):
    cur: Any = payload
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize_payload(payload: Dict[str, Any]) -> str:
    pred = _safe_get(payload, ["prediction", "loan_status"], None)
    p1 = _safe_get(payload, ["prediction", "probability_class_1"], None)

    shap_drivers: List[Dict[str, Any]] = _safe_get(payload, ["explanations", "shap", "top_drivers"], []) or []

    # dice_impact is optional (your current payload doesn’t include it)
    dice_block = _safe_get(payload, ["explanations", "dice_impact"], None)
    levers: List[Dict[str, Any]] = (dice_block or {}).get("ranked_levers", []) if isinstance(dice_block, dict) else []
    fixed: List[str] = ((dice_block or {}).get("goal", {}) or {}).get("fixed_features", []) if isinstance(dice_block, dict) else []

    lines = []
    lines.append(f"Prediction loan_status: {pred}")
    if p1 is not None:
        try:
            lines.append(f"Probability of class 1: {float(p1):.6f}")
        except Exception:
            lines.append(f"Probability of class 1: {p1}")

    lines.append("\nTop SHAP drivers (first 8):")
    for item in shap_drivers[:8]:
        feat = item.get("feature", "?")
        eff = str(item.get("effect", "")).replace("_", " ")
        val = item.get("value", 0.0)
        lines.append(f"- {feat}: {eff} (contribution={val})")

    if levers:
        lines.append("\nRanked levers toward class 1 (first 5):")
        lines.append(f"Fixed: {', '.join(map(str, fixed))}" if fixed else "Fixed: (none provided)")
        for item in levers[:5]:
            lines.append(f"- {item.get('feature','?')}: impact={item.get('impact_score')}")
    else:
        lines.append("\nNo dice_impact levers provided in payload.")

    return "\n".join(lines)


# def generate_assistant_explanation(payload: Dict[str, Any]) -> str:
#     """
#     Calls OpenAI to produce a user-friendly explanation from the model payload.
#     Requires OPENAI_API_KEY in your environment.
#     """
#     client = OpenAI()  # reads OPENAI_API_KEY by default :contentReference[oaicite:2]{index=2}

#     brief = summarize_payload(payload)

#     # You can include extra guardrails here to avoid over-claiming.
#     system_instructions = (
#         "You are a helpful assistant explaining a loan model result to a user.\n"
#         "Use plain language, no ML jargon. Be concise.\n"
#         "Explain what most influenced the result and what could plausibly improve odds.\n"
#         "Do NOT claim certainty; this is a model estimate.\n"
#         "Do NOT give legal/financial advice; provide general educational guidance only.\n"
#     )

#     user_message = (
#         "Here is a structured model payload (JSON-like summary). "
#         "Write a short explanation for the applicant.\n\n"
#         f"{brief}\n\n"
#         "Output format:\n"
#         "1) One-sentence result summary\n"
#         "2) 3–5 bullet points: biggest factors\n"
#         "3) 2–4 bullet points: general improvement ideas (if available from payload; otherwise general)\n"
#     )

#     resp = client.responses.create(
#         model=DEFAULT_MODEL,
#         input=[
#             {"role": "system", "content": system_instructions},
#             {"role": "user", "content": user_message},
#         ],
#     )  # Responses API :contentReference[oaicite:3]{index=3}

#     # The SDK provides a convenience accessor for the text output
#     return resp.output_text

def generate_assistant_explanation(payload: Dict[str, Any]) -> str:
    """
    Calls OpenAI to produce a user-friendly explanation from the model payload.
    Requires OPENAI_API_KEY in your environment.
    """
    client = OpenAI()

    brief = summarize_payload(payload)

    system_instructions = (
        "You are a helpful assistant explaining a loan decision to an applicant on behalf of a lender.\n"
        "Use clear, plain language with a professional, conversational tone.\n"
        "Avoid machine learning jargon and internal model terminology.\n"
        "Do not mention or imply a model, algorithm, scoring system, automation, AI, or predictions.\n"
        "Write as if you are communicating the lender's decision rationale.\n"
        "Do not claim certainty; describe results as estimates based on patterns in prior applications.\n"
        "Do not provide legal, financial, or personalized advice—only general educational guidance.\n"
        "Do not contradict the provided decision outcome.\n\n"

        "Label mapping (MUST follow exactly):\n"
        "- loan_status = 1 means APPROVED\n"
        "- loan_status = 0 means NOT APPROVED\n\n"

        "Payload structure (for your understanding):\n"
        "- prediction.loan_status: the decision label (0 or 1)\n"
        "- prediction.probability_class_1: estimated likelihood of approval if present\n"
        "- explanations.shap.top_drivers: strongest contributing factors with fields:\n"
        "  feature, effect (increases_class_1 / decreases_class_1), value (strength)\n"
        "- explanations.dice_impact (optional): ranked_levers and goal.fixed_features\n"
        "- validation.normalized_features: the applicant’s normalized inputs\n\n"

        "Output constraints:\n"
        "- Never mention internal feature names, encoded variables, or one-hot categories.\n"
        "- Convert any feature names into human-friendly phrases.\n"
        "- In the 'factors that influenced the decision' bullet list: ONLY list the factor phrases.\n"
        "  Do NOT add explanations, numbers, or commentary in those bullets.\n"
        "- If dice_impact is missing, do not invent specific levers; give clearly general improvement ideas.\n"
    )

    user_message = (
        "Below is a structured summary derived from the payload.\n\n"
        f"{brief}\n\n"
        "Write an explanation for the applicant with this format and style:\n"
        "- Start with one short paragraph stating the lender's decision (approved or not approved) and include the "
        "estimated approval likelihood as a percentage if it is available.\n"
        "- Do not mention a model, prediction, probability model, or automation; frame it as the lender's decision context.\n"
        "- Then list the factors that most influenced the decision using non-numbered bullet points.\n"
        "  IMPORTANT: in this section, bullets must be ONLY short factor names/phrases (no explanations).\n"
        "- Then list potential improvement ideas using non-numbered bullet points. If payload levers exist, base ideas on them; "
        "otherwise keep them general and clearly non-personalized.\n"
        "Do not use numbered lists or section headers. Keep it concise.\n"
    )

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_message},
        ],
    )

    return resp.choices[0].message.content


if __name__ == "__main__":
    # Optional: allow testing with stdin
    import sys
    data = sys.stdin.read().strip()
    if not data:
        print("Pipe a JSON payload into stdin, e.g. `python loan_assistant.py < payload.json`")
        sys.exit(0)

    payload = json.loads(data)
    print(generate_assistant_explanation(payload))
