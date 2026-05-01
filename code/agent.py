"""
agent.py
Core triage agent: retrieves docs, calls Groq, applies escalation rules,
and returns a structured triage decision for each support ticket.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from retriever import retrieve, format_context
from escalation import evaluate_escalation, check_out_of_scope
from prompts import SYSTEM_PROMPT, build_user_prompt

# ── Logging setup ────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
logger.propagate = False

_fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

for _filename in ("logs/agent.log", "logs/agent.txt"):
    _handler = logging.FileHandler(_filename, encoding="utf-8")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

FALLBACK_RESPONSE = {
    "status": "escalated",
    "product_area": "Unknown",
    "response": "We've received your request and a human support agent will review it shortly.",
    "justification": "Agent encountered an error processing this ticket; escalated for safety.",
    "request_type": "product_issue",
}


def _call_groq(system: str, user: str) -> str:
    """Call Groq API and return raw text response."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()

    logger.info("=" * 60)
    logger.info("PROMPT:\n%s", user[:500])
    logger.info("RESPONSE:\n%s", raw[:500])
    logger.info("=" * 60)

    return raw


def _parse_json_response(raw: str) -> Optional[Dict]:
    """Extract and parse JSON from the LLM response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_fix(result: Dict, llm_status: str, issue: str, subject: str, has_good_match: bool) -> Dict:
    """Apply escalation rules and ensure all fields are valid."""
    from config import ALLOWED_STATUS, ALLOWED_REQUEST_TYPES

    final_status, escalation_reason = evaluate_escalation(
        issue=issue,
        subject=subject,
        has_good_match=has_good_match,
        llm_status=result.get("status", "escalated"),
    )

    result["status"] = final_status

    if escalation_reason and final_status == "escalated" and llm_status != "escalated":
        result["justification"] = (
            result.get("justification", "") + f" [Auto-escalated: {escalation_reason}]"
        ).strip()

    if result["status"] == "escalated" and "human" not in result.get("response", "").lower():
        result["response"] = (
            result.get("response", "").rstrip() +
            " A human support agent will review your case and follow up shortly."
        ).strip()

    if result.get("status") not in ALLOWED_STATUS:
        result["status"] = "escalated"

    if result.get("request_type") not in ALLOWED_REQUEST_TYPES:
        result["request_type"] = "product_issue"

    for field in ["status", "product_area", "response", "justification", "request_type"]:
        if not result.get(field):
            result[field] = FALLBACK_RESPONSE[field]

    return result


def triage(
    issue: str,
    subject: str,
    company: str,
    collection,
    vectorizer,
    matrix,
    chunks,
) -> Dict:
    """Full triage pipeline for one support ticket."""
    issue = (issue or "").strip()
    subject = (subject or "").strip()
    company = (company or "None").strip()

    logger.info("TICKET | company=%s | subject=%s", company, subject)

    if check_out_of_scope(issue, subject):
        logger.info("RESULT | status=escalated | reason=out_of_scope")
        return {
            "status": "escalated",
            "product_area": "Out of Scope",
            "response": "This request appears to be outside the scope of our support services. A human agent will review and route it appropriately.",
            "justification": "Ticket content does not match any supported product domain.",
            "request_type": "invalid",
        }

    retrieval = retrieve(
        query=f"{subject} {issue}".strip(),
        collection=collection,
        vectorizer=vectorizer,
        matrix=matrix,
        chunks=chunks,
        company=company,
    )

    context = format_context(retrieval["hits"])
    has_good_match = retrieval["has_good_match"]

    logger.info("RETRIEVAL | best_score=%.4f | has_good_match=%s", retrieval["best_score"], has_good_match)

    user_prompt = build_user_prompt(
        issue=issue,
        subject=subject,
        company=company,
        context=context,
    )

    try:
        raw_response = _call_groq(SYSTEM_PROMPT, user_prompt)
        result = _parse_json_response(raw_response)

        if result is None:
            print(f"  [WARN] Could not parse Groq response. Raw: {raw_response[:200]}")
            logger.warning("Could not parse Groq response. Raw: %s", raw_response[:200])
            result = dict(FALLBACK_RESPONSE)
    except Exception as e:
        print(f"  [ERROR] Groq API call failed: {e}")
        logger.error("Groq API call failed: %s", e)
        result = dict(FALLBACK_RESPONSE)

    llm_status = result.get("status", "escalated")

    result = _validate_and_fix(
        result=result,
        llm_status=llm_status,
        issue=issue,
        subject=subject,
        has_good_match=has_good_match,
    )

    logger.info("RESULT | status=%s | product_area=%s | request_type=%s", result["status"], result["product_area"], result["request_type"])

    return result
