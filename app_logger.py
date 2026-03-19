"""
app_logger.py
─────────────────────────────────────────────────────────────────────
Persists every underwriting submission to data/applications.json
so the Analytics Dashboard can read aggregate stats.

Called from app.py after every successful agent run.
"""

import json
import os
import re
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "applications.json")


def _load() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save(records: list):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(records, f, indent=2)


def _parse_decision(oracle_text: str) -> str:
    """Extract decision type from ORACLE output text."""
    u = oracle_text.upper()
    if "REJECTED"     in u or "REJECT"       in u: return "Rejected"
    if "COUNTER-OFFER" in u or "COUNTER OFFER" in u: return "Counter-Offer"
    if "ESCALATED"    in u or "ESCALATE"      in u: return "Escalated"
    return "Approved"


def _parse_score(text: str, label: str) -> float:
    """
    Try to extract a numeric score from agent output text.
    e.g. label="TOTAL SALES SCORE" → looks for "95/100" → returns 95.0
    """
    pattern = rf"{re.escape(label)}\s*[:\-]\s*(\d+(?:\.\d+)?)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def log_application(
    applicant_name:  str,
    monthly_income:  int,
    loan_amount:     int,
    loan_purpose:    str,
    credit_score:    int,
    existing_emis:   int,
    emp_type:        str,
    years_employed:  int,
    bio:             dict,
    results:         dict,
    new_emi:         int,
    dti:             float,
    lti:             float,
):
    """
    Append one underwriting record to the log file.

    Parameters
    ----------
    bio     : BioSentinel result dict (fraud_score, signal, class_probs, etc.)
    results : Agent output dict (aria, rex, clara, oracle text strings)
    """
    records = _load()

    decision = _parse_decision(results.get("oracle", ""))

    # Try to extract agent scores from output text
    aria_score  = _parse_score(results.get("aria",   ""), "TOTAL SALES SCORE")
    rex_score   = _parse_score(results.get("rex",    ""), "TOTAL RISK SCORE")

    record = {
        "id":              f"LN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp":       datetime.now().isoformat(),
        "applicant_name":  applicant_name or "Anonymous",
        "monthly_income":  monthly_income,
        "loan_amount":     loan_amount,
        "loan_purpose":    loan_purpose or "Not specified",
        "credit_score":    credit_score,
        "existing_emis":   existing_emis,
        "emp_type":        emp_type,
        "years_employed":  years_employed,
        "new_emi":         new_emi,
        "dti":             dti,
        "lti":             lti,
        "decision":        decision,
        "aria_score":      aria_score,
        "rex_score":       rex_score,
        "fraud_score":     bio.get("fraud_score", 0.0),
        "fraud_signal":    bio.get("signal", "UNKNOWN"),
        "predicted_class": bio.get("predicted_class", "human"),
        "p_human":         bio.get("class_probs", {}).get("human",   0.0),
        "p_bot":           bio.get("class_probs", {}).get("bot",     0.0),
        "p_duress":        bio.get("class_probs", {}).get("duress",  0.0),
        "p_coached":       bio.get("class_probs", {}).get("coached", 0.0),
    }

    records.append(record)
    _save(records)
    return record


def load_all() -> list:
    """Return all logged application records."""
    return _load()


def clear_log():
    """Wipe the log — used by the analytics reset button."""
    _save([])