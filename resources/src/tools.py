"""Classroom-safe tools for LangGraph agent examples."""
from __future__ import annotations
import csv, json, math, os
from pathlib import Path
# from datetime import datetime
import datetime
from typing import Dict, List, Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def calculator(expression: str) -> str:
    # timestamp = datetime.now().isoformat(timespec="seconds")
    """Evaluate simple arithmetic only. Deliberately blocks names, imports and attributes."""
    allowed = set("0123456789+-*/(). %")
    timestamp = datetime.datetime.now()
    if any(ch not in allowed for ch in expression):
        return f"Blocked at {timestamp}: calculator accepts only simple arithmetic characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {})) + f": {timestamp}"
    except Exception as exc:
        return f"{timestamp}: Calculator error: {exc}"


def pick_rate(items_picked: float, hours: float) -> Dict[str, Any]:
    if hours <= 0:
        return {"error": "hours must be greater than zero"}
    rate = items_picked / hours
    return {
        "items_picked": items_picked,
        "hours": hours,
        "pick_rate_per_hour": round(rate, 2),
        "interpretation_warning": "Do not use this metric alone for judging a person. Add safety, quality, task mix, training, fatigue, equipment, and human review."
    }


def load_knowledge_base() -> List[Dict[str, str]]:
    with open(DATA_DIR / "agentic_ai_knowledge_base.csv", newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def knowledge_lookup(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    rows = load_knowledge_base()
    tokens = {t.lower().strip('.,;:()') for t in query.split() if len(t) > 2}
    scored = []
    for row in rows:
        text = f"{row['topic']} {row['source_type']} {row['content']}".lower()
        score = sum(1 for t in tokens if t in text)
        if score:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]] or rows[:top_k]


def retrieve_observations(area: str = "all") -> List[Dict[str, str]]:
    with open(DATA_DIR / "amazon_visit_observation_examples.csv", newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if area.lower() == "all":
        return rows
    return [r for r in rows if area.lower() in r['area'].lower()]


def risk_assessment(user_request: str) -> Dict[str, str]:
    risky_terms = ["performance", "disciplinary", "fire", "dismiss", "approve", "override", "safety", "personal data", "employee", "named"]
    lower = user_request.lower()
    hits = [t for t in risky_terms if t in lower]
    risk = "high" if hits else "medium" if any(w in lower for w in ["policy", "exception", "quality", "escalate"]) else "low"
    return {"risk": risk, "signals": ", ".join(hits) or "none", "recommendation": "human_review" if risk == "high" else "continue_with_logging"}


def human_review_packet(proposed_action: str, evidence: str, risk: str) -> Dict[str, str]:
    return {
        "status": "requires_human_review",
        "risk": risk,
        "proposed_action": proposed_action,
        "evidence": evidence,
        "review_options": "approve | edit | reject",
        "note": "In class, the student or facilitator acts as the reviewer. In production, connect this to an authenticated review workflow."
    }
