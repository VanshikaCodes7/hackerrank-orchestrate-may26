"""
escalation.py
Rule-based escalation guard that overrides LLM decisions.
These patterns always force status=escalated regardless of what Claude says.
"""

import re
from typing import Tuple, List

# High-risk keyword patterns → always escalate
ESCALATION_PATTERNS: List[Tuple[str, str]] = [
    # Account security
    (r"\b(hacked|compromised|unauthorized.{0,20}access|account.{0,20}stolen)\b", "account_security"),
    (r"\b(someone.{0,20}(logged in|accessing).{0,20}account)\b", "account_security"),
    (r"\b(change.{0,20}password.{0,20}(urgently|immediately|asap))\b", "account_security"),

    # Fraud & financial
    (r"\b(fraud|fraudulent|chargeback|dispute.{0,20}charge|unauthorized.{0,20}(charge|transaction|payment))\b", "fraud_financial"),
    (r"\b(money.{0,20}(stolen|missing|deducted|charged))\b", "fraud_financial"),
    (r"\b(refund.{0,20}(not received|denied|rejected))\b", "fraud_financial"),

    # Legal & compliance
    (r"\b(lawsuit|legal.{0,20}action|attorney|lawyer|court|gdpr|data.{0,20}(breach|leak|stolen))\b", "legal_compliance"),
    (r"\b(sue|suing|litigation|subpoena)\b", "legal_compliance"),
    (r"\b(right.{0,20}to.{0,20}(erasure|deletion|be forgotten))\b", "legal_compliance"),

    # Identity & PII
    (r"\b(identity.{0,20}theft|personal.{0,20}data.{0,20}(exposed|leaked|stolen))\b", "identity_pii"),
    (r"\b(ssn|social security|passport.{0,20}number|bank.{0,20}account.{0,20}number)\b", "identity_pii"),

    # Visa-specific high risk
    (r"\b(card.{0,20}(stolen|lost|missing|cloned|skimmed))\b", "card_security"),
    (r"\b(pin.{0,20}(compromised|stolen|exposed))\b", "card_security"),

    # Abuse / threats
    (r"\b(threatening|harassment|abuse|discriminat(ion|ory))\b", "abuse"),

    # Prompt injection attempts
    (r"(ignore.{0,30}(previous|above|prior|all).{0,30}instructions?)", "prompt_injection"),
    (r"(you are now|pretend (you are|to be)|act as (a|an) [a-z]+\s*(ai|bot|assistant))", "prompt_injection"),
    (r"(system prompt|jailbreak|DAN mode)", "prompt_injection"),
]

# Patterns that suggest the ticket is completely out of scope
OUT_OF_SCOPE_PATTERNS = [
    r"\b(weather|recipe|sports|stock(s| market)|lottery|horoscope)\b",
    r"\b(write (me |a )?(poem|story|essay|code))\b",
    r"\b(who (is|was) [A-Z][a-z]+|what is the capital of)\b",
]


def check_escalation(issue: str, subject: str = "") -> Tuple[bool, str]:
    """
    Check if a ticket should be hard-escalated based on rules.

    Returns:
        (should_escalate: bool, reason: str)
    """
    full_text = f"{subject} {issue}".lower()

    for pattern, category in ESCALATION_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            return True, category

    return False, ""


def check_out_of_scope(issue: str, subject: str = "") -> bool:
    """Returns True if the ticket appears completely out of scope."""
    full_text = f"{subject} {issue}".lower()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            return True
    return False


def check_no_corpus_match(has_good_match: bool) -> Tuple[bool, str]:
    """Escalate if retrieval found no relevant documentation."""
    if not has_good_match:
        return True, "no_corpus_match"
    return False, ""


def evaluate_escalation(
    issue: str,
    subject: str,
    has_good_match: bool,
    llm_status: str,
) -> Tuple[str, str]:
    """
    Final escalation decision combining rule-based checks + LLM output.

    Returns:
        (final_status: str, escalation_reason: str)
    """
    # 1. Hard rules first
    should_escalate, reason = check_escalation(issue, subject)
    if should_escalate:
        return "escalated", f"Hard rule triggered: {reason}"

    # 2. No corpus match
    no_match, match_reason = check_no_corpus_match(has_good_match)
    if no_match:
        return "escalated", "No relevant documentation found in corpus"

    # 3. Trust LLM decision
    return llm_status, ""
