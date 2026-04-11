"""
graders.py - Email Triage deterministic graders.

Score formula:  safe_score = 0.1 + raw_score * 0.8
This maps [0, 1] -> [0.1, 0.9] mathematically.
It is IMPOSSIBLE to return exactly 0.0 or 1.0.
"""
from __future__ import annotations
from difflib import SequenceMatcher
from typing import Tuple
from models import EmailCategory, EmailPriority, TriageAction, ActionType
from data import EMAIL_BY_ID


def _safe(raw: float) -> float:
    """
    Map any raw score in [0,1] to a safe score in [0.1, 0.9].
    Mathematically impossible to return 0.0 or 1.0.
    """
    clamped = max(0.0, min(1.0, float(raw)))
    return round(0.1 + clamped * 0.8, 4)


def _sim(a: str, b: str) -> float:
    if not a and not b: return 0.85
    if not a or not b:  return 0.1
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _kw(text: str, keywords: list) -> float:
    if not keywords: return 0.85
    if not text:     return 0.1
    hits = sum(1 for k in keywords if k.lower() in text.lower())
    return max(0.1, min(0.9, hits / len(keywords)))


def _items(submitted: list, reference: list) -> float:
    if not reference:  return 0.85
    if not submitted:  return 0.1
    scores = [max(_sim(ref, s) for s in submitted) for ref in reference]
    return max(0.1, min(0.9, sum(scores) / len(scores)))


# ── Task 1 ────────────────────────────────────────────────────────
def grade_classify(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    email = EMAIL_BY_ID[email_id]
    gt    = email["gt_category"]
    pred  = action.category

    if pred == gt:
        raw, reason = 0.98, "Correct"
    elif (pred == EmailCategory.SPAM) != (gt == EmailCategory.SPAM):
        raw, reason = 0.02, "Spam/non-spam confusion"
    elif {pred, gt} in ({EmailCategory.WORK, EmailCategory.SUPPORT},
                        {EmailCategory.WORK, EmailCategory.FINANCE},
                        {EmailCategory.NEWSLETTER, EmailCategory.PERSONAL}):
        raw, reason = 0.55, "Related class"
    else:
        raw, reason = 0.20, "Wrong category"

    score = _safe(raw)
    return score, {"gt": gt.value, "pred": pred.value, "reason": reason, "score": score}


# ── Task 2 ────────────────────────────────────────────────────────
def grade_triage(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    email  = EMAIL_BY_ID[email_id]
    gt_pri = email["gt_priority"]
    gt_kw  = email["gt_reply_keywords"]

    cls_score, cls_bd = grade_classify(email_id, action)

    pred_pri = action.priority or EmailPriority.MEDIUM
    if pred_pri == gt_pri:
        pri_raw, pri_r = 0.98, "Correct"
    elif (gt_pri == EmailPriority.HIGH   and pred_pri == EmailPriority.MEDIUM) or \
         (gt_pri == EmailPriority.MEDIUM and pred_pri != EmailPriority.MEDIUM):
        pri_raw, pri_r = 0.55, "Adjacent"
    else:
        pri_raw, pri_r = 0.02, "Opposite"
    pri_score = _safe(pri_raw)

    act_text  = action.suggested_action or ""
    act_score = _safe(_kw(act_text, gt_kw)) if gt_kw else (_safe(0.75) if act_text else _safe(0.2))

    raw   = 0.30 * cls_score + 0.40 * pri_score + 0.30 * act_score
    score = _safe(raw)

    return score, {
        "category": cls_bd,
        "priority": {"gt": gt_pri.value, "pred": pred_pri.value,
                     "score": pri_score, "reason": pri_r},
        "action":   {"score": act_score, "text": act_text[:60]},
        "score":    score,
    }


# ── Task 3 ────────────────────────────────────────────────────────
def grade_full_triage(email_id: str, action: TriageAction) -> Tuple[float, dict]:
    email      = EMAIL_BY_ID[email_id]
    gt_items   = email["gt_action_items"]
    gt_kw      = email["gt_reply_keywords"]

    tri_score, tri_bd = grade_triage(email_id, action)

    submitted   = action.action_items or []
    item_score  = _safe(_items(submitted, gt_items))

    draft = action.draft_reply or ""
    if gt_kw:
        reply_score = _safe(_kw(draft, gt_kw) + (0.1 if len(draft) > 30 else 0))
    else:
        reply_score = _safe(0.15) if (draft and len(draft) > 20) else _safe(0.85)

    raw   = 0.40 * tri_score + 0.35 * item_score + 0.25 * reply_score
    score = _safe(raw)

    return score, {
        "triage":       tri_bd,
        "action_items": {"count": len(submitted), "score": item_score},
        "reply":        {"score": reply_score, "preview": draft[:80]},
        "score":        score,
    }


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
    fn = GRADER_MAP.get(task_id)
    if fn is None:
        raise ValueError(f"Unknown task_id: {task_id!r}")
    score, bd = fn(email_id, action)
    # Final safety net — should never be needed given _safe() above
    score = round(max(0.001, min(0.999, float(score))), 4)
    return score, bd
