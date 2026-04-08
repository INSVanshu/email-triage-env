"""
Deterministic graders for the Email Triage environment.
All scores are clamped to strictly (0.001, 0.999) — never exactly 0.0 or 1.0.
"""
from __future__ import annotations
from typing import Dict, Tuple
from difflib import SequenceMatcher

from models import EmailCategory, EmailPriority, TriageAction, ActionType
from data import EMAIL_BY_ID


# ── Score clamping — required by validator ────────────────────────
def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (never 0.0 or 1.0)."""
    return round(max(0.001, min(0.999, score)), 4)


# ── Helpers ───────────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 0.999
    if not a or not b:
        return 0.001
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _keyword_coverage(text: str, keywords: list) -> float:
    if not keywords:
        return 0.999
    if not text:
        return 0.001
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    raw = hits / len(keywords)
    # Never return exact 0 or 1
    return max(0.001, min(0.999, raw))


def _action_item_coverage(submitted: list, reference: list) -> float:
    if not reference:
        return 0.999
    if not submitted:
        return 0.001
    scores = []
    for ref in reference:
        best = max(_text_similarity(ref, s) for s in submitted)
        scores.append(best)
    raw = sum(scores) / len(scores)
    return max(0.001, min(0.999, raw))


# ── Task 1 — Classify ─────────────────────────────────────────────

def grade_classify(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    email = EMAIL_BY_ID[email_id]
    gt: EmailCategory = email["gt_category"]
    pred: EmailCategory = action.category

    breakdown = {"gt_category": gt.value, "pred_category": pred.value}

    if pred == gt:
        raw = 0.95                          # correct — but not exactly 1.0
        breakdown["reason"] = "Correct category"
    elif (pred == EmailCategory.SPAM) != (gt == EmailCategory.SPAM):
        raw = 0.05                          # spam confusion — but not exactly 0.0
        breakdown["reason"] = "Spam vs non-spam confusion"
    elif {pred, gt} in (
        {EmailCategory.WORK, EmailCategory.SUPPORT},
        {EmailCategory.WORK, EmailCategory.FINANCE},
        {EmailCategory.NEWSLETTER, EmailCategory.PERSONAL},
    ):
        raw = 0.50
        breakdown["reason"] = "Related-class confusion"
    else:
        raw = 0.20
        breakdown["reason"] = "Wrong category"

    score = _clamp(raw)
    return score, breakdown


# ── Task 2 — Triage ───────────────────────────────────────────────

def grade_triage(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    email = EMAIL_BY_ID[email_id]
    gt_pri: EmailPriority = email["gt_priority"]
    gt_kw: list = email["gt_reply_keywords"]

    classify_score, classify_bd = grade_classify(email_id, action)

    pred_pri = action.priority or EmailPriority.MEDIUM
    if pred_pri == gt_pri:
        priority_score = 0.95
        pri_reason = "Correct priority"
    elif (gt_pri == EmailPriority.HIGH and pred_pri == EmailPriority.MEDIUM) or \
         (gt_pri == EmailPriority.MEDIUM and pred_pri in (EmailPriority.HIGH, EmailPriority.LOW)):
        priority_score = 0.50
        pri_reason = "Adjacent priority"
    else:
        priority_score = 0.05
        pri_reason = "Opposite priority"

    action_text = action.suggested_action or ""
    action_kw_score = _keyword_coverage(action_text, gt_kw) if gt_kw else (0.80 if action_text else 0.20)

    if not action.suggested_action and gt_pri == EmailPriority.HIGH:
        action_kw_score = max(0.001, action_kw_score - 0.20)

    raw = 0.30 * classify_score + 0.40 * priority_score + 0.30 * action_kw_score
    score = _clamp(raw)

    breakdown = {
        "category": classify_bd,
        "priority": {"gt": gt_pri.value, "pred": pred_pri.value,
                     "score": priority_score, "reason": pri_reason},
        "suggested_action": {"score": action_kw_score, "text": action_text[:80]},
        "weighted_score": score,
    }
    return score, breakdown


# ── Task 3 — Full Triage ──────────────────────────────────────────

def grade_full_triage(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    email = EMAIL_BY_ID[email_id]
    gt_items: list = email["gt_action_items"]
    gt_reply_kw: list = email["gt_reply_keywords"]

    triage_score, triage_bd = grade_triage(email_id, action)

    submitted_items = action.action_items or []
    items_score = _action_item_coverage(submitted_items, gt_items)
    if not submitted_items and email["gt_priority"] == EmailPriority.HIGH:
        items_score = max(0.001, items_score - 0.30)

    draft = action.draft_reply or ""
    if gt_reply_kw:
        reply_score = _keyword_coverage(draft, gt_reply_kw)
        if len(draft) > 30 and reply_score > 0.001:
            reply_score = min(0.999, reply_score + 0.10)
    else:
        # Spam/newsletter — penalise non-empty draft
        reply_score = 0.05 if (draft and len(draft) > 20) else 0.95

    raw = 0.40 * triage_score + 0.35 * items_score + 0.25 * reply_score
    score = _clamp(raw)

    breakdown = {
        "triage": triage_bd,
        "action_items": {"submitted": submitted_items,
                         "reference_count": len(gt_items),
                         "score": round(items_score, 4)},
        "draft_reply": {"score": round(reply_score, 4), "draft_preview": draft[:100]},
        "weighted_score": score,
    }
    return score, breakdown


# ── Dispatcher ────────────────────────────────────────────────────

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
    grader = GRADER_MAP.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    return grader(email_id, action)
