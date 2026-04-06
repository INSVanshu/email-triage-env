"""
Deterministic graders for the Email Triage environment.

Each grader returns a float score in [0.0, 1.0] and a breakdown dict.

Scoring philosophy
──────────────────
• Partial credit is given at every level — agents always receive *some* signal.
• Higher-level tasks (triage, full_triage) stack lower-level rewards so the
  agent benefits from getting the basics right even if advanced fields fail.
• Penalty (-0.1) is applied for nonsensical actions like marking a WORK HIGH
  priority email as SPAM or leaving action_items empty on a full_triage task.
"""
from __future__ import annotations
from typing import Dict, Tuple
from difflib import SequenceMatcher

from models import EmailCategory, EmailPriority, TriageAction, ActionType
from data import EMAIL_BY_ID


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    """Simple ratio-based text similarity [0,1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    """Fraction of required keywords present in text (case-insensitive)."""
    if not keywords:
        return 1.0  # no keywords required → full marks
    if not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def _action_item_coverage(submitted: list[str], reference: list[str]) -> float:
    """
    Greedy best-match coverage: for each reference item, find the best
    similarity match in submitted items.  Returns average best-match score.
    """
    if not reference:
        return 1.0
    if not submitted:
        return 0.0
    scores = []
    for ref in reference:
        best = max(_text_similarity(ref, s) for s in submitted)
        scores.append(best)
    return sum(scores) / len(scores)


# ─────────────────────────────────────────────────────────────────
# Task 1 – Classify
# ─────────────────────────────────────────────────────────────────

def grade_classify(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    """
    Score [0,1] for a CLASSIFY action.

    • Correct category  → 1.0
    • Off-by-one class  → 0.5  (e.g. WORK vs SUPPORT, FINANCE vs WORK)
    • SPAM vs non-SPAM  → 0.0  (harmful misclassification)
    • Any other wrong   → 0.2  (partial – at least tried)
    """
    email = EMAIL_BY_ID[email_id]
    gt: EmailCategory = email["gt_category"]
    pred: EmailCategory = action.category

    breakdown = {"gt_category": gt.value, "pred_category": pred.value}

    if pred == gt:
        score = 1.0
        breakdown["reason"] = "Exact match"
    elif (pred == EmailCategory.SPAM) != (gt == EmailCategory.SPAM):
        # Spam/not-spam confusion is the most harmful
        score = 0.0
        breakdown["reason"] = "Spam vs non-spam confusion"
    elif {pred, gt} in (
        {EmailCategory.WORK, EmailCategory.SUPPORT},
        {EmailCategory.WORK, EmailCategory.FINANCE},
        {EmailCategory.NEWSLETTER, EmailCategory.PERSONAL},
    ):
        score = 0.5
        breakdown["reason"] = "Related-class confusion"
    else:
        score = 0.2
        breakdown["reason"] = "Wrong category"

    return round(score, 4), breakdown


# ─────────────────────────────────────────────────────────────────
# Task 2 – Triage  (classify + priority + suggested_action)
# ─────────────────────────────────────────────────────────────────

def grade_triage(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    """
    Score [0,1] for a TRIAGE action.

    Weights:
    • Category accuracy   30%
    • Priority accuracy   40%
    • Suggested action    30%  (keyword coverage)
    """
    email = EMAIL_BY_ID[email_id]
    gt_cat: EmailCategory = email["gt_category"]
    gt_pri: EmailPriority = email["gt_priority"]
    gt_kw: list[str] = email["gt_reply_keywords"]

    classify_score, classify_bd = grade_classify(email_id, action)

    # Priority scoring
    pred_pri = action.priority or EmailPriority.MEDIUM
    if pred_pri == gt_pri:
        priority_score = 1.0
        pri_reason = "Exact match"
    elif (gt_pri == EmailPriority.HIGH and pred_pri == EmailPriority.MEDIUM) or \
         (gt_pri == EmailPriority.MEDIUM and pred_pri in (EmailPriority.HIGH, EmailPriority.LOW)):
        priority_score = 0.5
        pri_reason = "Adjacent priority"
    else:
        priority_score = 0.0
        pri_reason = "Opposite priority"

    # Suggested action scoring (keyword coverage against reply keywords)
    action_text = action.suggested_action or ""
    action_kw_score = _keyword_coverage(action_text, gt_kw) if gt_kw else (1.0 if action_text else 0.3)

    # Penalty: no suggested action when there clearly should be one
    if not action.suggested_action and gt_pri == EmailPriority.HIGH:
        action_kw_score = max(0.0, action_kw_score - 0.2)

    score = 0.30 * classify_score + 0.40 * priority_score + 0.30 * action_kw_score

    breakdown = {
        "category": classify_bd,
        "priority": {"gt": gt_pri.value, "pred": pred_pri.value, "score": priority_score, "reason": pri_reason},
        "suggested_action": {"score": action_kw_score, "text": action_text[:80]},
        "weighted_score": round(score, 4),
    }

    return round(score, 4), breakdown


# ─────────────────────────────────────────────────────────────────
# Task 3 – Full Triage
# ─────────────────────────────────────────────────────────────────

def grade_full_triage(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    """
    Score [0,1] for a FULL_TRIAGE action.

    Weights:
    • Triage sub-score (category + priority + suggested_action)   40%
    • Action items coverage                                        35%
    • Draft reply quality (keyword coverage)                       25%
    """
    email = EMAIL_BY_ID[email_id]
    gt_items: list[str] = email["gt_action_items"]
    gt_reply_kw: list[str] = email["gt_reply_keywords"]

    triage_score, triage_bd = grade_triage(email_id, action)

    # Action items
    submitted_items = action.action_items or []
    items_score = _action_item_coverage(submitted_items, gt_items)

    # Penalize empty action items on high-importance emails
    if not submitted_items and email["gt_priority"] == EmailPriority.HIGH:
        items_score = max(0.0, items_score - 0.3)

    # Draft reply
    draft = action.draft_reply or ""
    if gt_reply_kw:
        reply_score = _keyword_coverage(draft, gt_reply_kw)
        # Bonus if draft is substantive (>30 chars)
        if len(draft) > 30 and reply_score > 0:
            reply_score = min(1.0, reply_score + 0.1)
    else:
        # Email doesn't need a reply (SPAM, newsletter) — penalize unnecessary drafts
        reply_score = 0.0 if (draft and len(draft) > 20) else 1.0

    score = 0.40 * triage_score + 0.35 * items_score + 0.25 * reply_score

    breakdown = {
        "triage": triage_bd,
        "action_items": {
            "submitted": submitted_items,
            "reference_count": len(gt_items),
            "score": round(items_score, 4),
        },
        "draft_reply": {
            "score": round(reply_score, 4),
            "draft_preview": draft[:100],
        },
        "weighted_score": round(score, 4),
    }

    return round(score, 4), breakdown


# ─────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────

GRADER_MAP = {
    "task_classify":    grade_classify,
    "task_triage":      grade_triage,
    "task_full_triage": grade_full_triage,
}

ACTION_TYPE_MAP = {
    "task_classify":    ActionType.CLASSIFY,
    "task_triage":      ActionType.TRIAGE,
    "task_full_triage": ActionType.FULL_TRIAGE,
}


def grade(task_id: str, email_id: str, action: TriageAction) -> Tuple[float, dict]:
    """Main entry point: dispatches to the correct grader."""
    grader = GRADER_MAP.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return grader(email_id, action)
