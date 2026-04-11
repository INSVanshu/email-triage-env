"""
baseline.py – Rule-based baseline agent.
All scores clamped strictly to (0.001, 0.999) — never 0.0 or 1.0.
"""
from __future__ import annotations
import os, sys, time, argparse, re, json
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TriageAction, ActionType, EmailCategory, EmailPriority
from data import TASK_EMAILS, EMAIL_BY_ID
from server.environment import EmailTriageEnvironment

# ── Score clamping ────────────────────────────────────────────────
def _clamp(v):
    try:    return round(max(0.001, min(0.999, float(v))), 4)
    except: return 0.001

# ── Rule-based heuristics ─────────────────────────────────────────
def rule_based_action(task_id: str, obs_dict: dict) -> TriageAction:
    subject = obs_dict.get("subject", "")
    body    = obs_dict.get("body",    "")
    text    = (subject + " " + body).lower()

    if any(w in text for w in ["won","prize","lottery","processing fee","bank details"]):
        cat = EmailCategory.SPAM
    elif any(w in text for w in ["invoice","billing","overdue","aws bill","charge"]):
        cat = EmailCategory.FINANCE
    elif any(w in text for w in ["outage","incident","alert","sign-in","security","unreachable"]):
        cat = EmailCategory.SUPPORT
    elif any(w in text for w in ["newsletter","unsubscribe","tips"]):
        cat = EmailCategory.NEWSLETTER
    elif any(w in text for w in ["lunch","dinner","friend","weekend","saturday"]):
        cat = EmailCategory.PERSONAL
    else:
        cat = EmailCategory.WORK

    if any(w in text for w in ["urgent","critical","action required","immediately","overdue",
                                "outage","friday","asap","15th","security alert"]):
        pri = EmailPriority.HIGH
    elif any(w in text for w in ["newsletter","tips","lunch","prize","congratulations","poll"]):
        pri = EmailPriority.LOW
    else:
        pri = EmailPriority.MEDIUM

    if task_id == "task_classify":
        return TriageAction(action_type=ActionType.CLASSIFY, category=cat)

    if task_id == "task_triage":
        hints = {
            EmailCategory.SPAM:       "Mark as spam and delete",
            EmailCategory.FINANCE:    f"Process payment/invoice – {pri.value} priority",
            EmailCategory.SUPPORT:    "Acknowledge and investigate immediately",
            EmailCategory.WORK:       f"Review and respond – {pri.value} priority",
            EmailCategory.NEWSLETTER: "Archive or unsubscribe",
            EmailCategory.PERSONAL:   "Reply at your convenience",
        }
        return TriageAction(action_type=ActionType.TRIAGE, category=cat, priority=pri,
                            suggested_action=hints.get(cat, f"Handle with {pri.value} priority"))

    # task_full_triage
    lines   = [l.strip() for l in body.split("\n") if l.strip()]
    items   = [l for l in lines if any(t in l.lower() for t in
               ["please","must","submit","review","confirm","pay","check","acknowledge"])][:3]
    if not items:
        items = ["Review and take appropriate action"]

    draft = ""
    if cat not in (EmailCategory.SPAM, EmailCategory.NEWSLETTER):
        draft = (f"Thank you for your email regarding '{subject[:40]}'. "
                 f"I have noted this as {pri.value} priority and will respond accordingly.")

    return TriageAction(action_type=ActionType.FULL_TRIAGE, category=cat, priority=pri,
                        suggested_action=f"Handle {cat.value} – {pri.value} priority",
                        action_items=items, draft_reply=draft)


# ── Episode runner ────────────────────────────────────────────────
def run_episode(task_id: str, agent_fn) -> Tuple[float, list]:
    env     = EmailTriageEnvironment()
    obs     = env.reset(task_id=task_id)
    details = []
    step_n  = 0

    while not obs.done:
        step_n += 1
        action = agent_fn(task_id, obs.model_dump())
        obs, reward, done, info = env.step(action)
        details.append({"step": step_n, "reward": _clamp(reward),
                         "breakdown": info.get("breakdown", {})})
        if done:
            break

    # Clamp the final episode score before returning
    return _clamp(env.episode_score()), details


# ── run_baseline_in_process — called by FastAPI /baseline ─────────
def run_baseline_in_process():
    """
    Run rule-based baseline on all 3 tasks.
    Returns (scores_dict, details_dict) with ALL scores clamped to (0.001, 0.999).
    """
    tasks   = ["task_classify", "task_triage", "task_full_triage"]
    scores  = {}
    details = {}
    for task_id in tasks:
        raw_score, ep_details = run_episode(task_id, rule_based_action)
        scores[task_id]  = _clamp(raw_score)   # ← clamped here
        details[task_id] = ep_details
    return scores, details


# ── CLI ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None,
                        choices=["task_classify","task_triage","task_full_triage"])
    args = parser.parse_args()

    tasks = [args.task] if args.task else ["task_classify","task_triage","task_full_triage"]

    print(f"\n{'='*55}\n  Email Triage Baseline\n{'='*55}\n")
    all_scores = {}
    for task_id in tasks:
        score, ep = run_episode(task_id, rule_based_action)
        all_scores[task_id] = score
        for d in ep:
            bar = "█"*int(d["reward"]*10) + "░"*(10-int(d["reward"]*10))
            print(f"  Step {d['step']:02d}  [{bar}]  {d['reward']:.4f}")
        print(f"  Episode score: {score:.4f}  (strictly in (0,1): {0 < score < 1})\n")

    mean = _clamp(sum(all_scores.values()) / len(all_scores))
    print(f"Mean: {mean:.4f}")

if __name__ == "__main__":
    main()
