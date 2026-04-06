"""
baseline.py – Baseline agent for the Email Triage OpenEnv environment.

Two modes
─────────
1. RULE-BASED baseline (no API key needed)
   Uses hand-crafted heuristics (keyword matching) to triage emails.
   Run:  python baseline.py

2. LLM baseline (requires OPENAI_API_KEY)
   Uses the OpenAI API client to triage emails via GPT.
   Run:  OPENAI_API_KEY=sk-... python baseline.py --llm

The script runs all 3 tasks and prints a score table.
It also exposes run_baseline_in_process() for the FastAPI /baseline endpoint.
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import re
from typing import Tuple

# Make repo root importable when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    TriageAction, ActionType, EmailCategory, EmailPriority,
)
from data import TASK_EMAILS, EMAIL_BY_ID
from server.environment import EmailTriageEnvironment
from graders import grade


# ─────────────────────────────────────────────────────────────────
# Rule-based heuristic baseline
# ─────────────────────────────────────────────────────────────────

SPAM_SIGNALS     = ["won", "prize", "lottery", "claim", "processing fee", "bank details", "million"]
WORK_SIGNALS     = ["budget", "contract", "meeting", "team", "project", "review", "deadline", "q3", "q4"]
FINANCE_SIGNALS  = ["invoice", "payment", "billing", "overdue", "aws bill", "charge", "settle"]
SUPPORT_SIGNALS  = ["outage", "alert", "incident", "unreachable", "error rate", "sign-in", "security"]
NEWSLETTER_SIG   = ["unsubscribe", "newsletter", "tips", "you are receiving"]
PERSONAL_SIG     = ["lunch", "dinner", "coffee", "friend", "family", "weekend"]

HIGH_SIG         = ["urgent", "critical", "action required", "immediately", "outage", "overdue",
                    "security alert", "down", "friday", "asap", "deadline", "15th"]
LOW_SIG          = ["newsletter", "tips", "poll", "unsubscribe", "lunch", "prize", "congratulations"]


def _classify_heuristic(subject: str, body: str) -> EmailCategory:
    text = (subject + " " + body).lower()
    if any(s in text for s in SPAM_SIGNALS):
        return EmailCategory.SPAM
    if any(s in text for s in FINANCE_SIGNALS):
        return EmailCategory.FINANCE
    if any(s in text for s in SUPPORT_SIGNALS):
        return EmailCategory.SUPPORT
    if any(s in text for s in NEWSLETTER_SIG):
        return EmailCategory.NEWSLETTER
    if any(s in text for s in PERSONAL_SIG):
        return EmailCategory.PERSONAL
    if any(s in text for s in WORK_SIGNALS):
        return EmailCategory.WORK
    return EmailCategory.UNKNOWN


def _priority_heuristic(subject: str, body: str) -> EmailPriority:
    text = (subject + " " + body).lower()
    if any(s in text for s in HIGH_SIG):
        return EmailPriority.HIGH
    if any(s in text for s in LOW_SIG):
        return EmailPriority.LOW
    return EmailPriority.MEDIUM


def _extract_action_items(body: str) -> list[str]:
    """Very naive action item extractor — looks for imperative sentences."""
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    items = []
    triggers = ["please", "you need", "must", "submit", "review", "confirm",
                "send", "respond", "complete", "check", "pay", "acknowledge"]
    for line in lines:
        if any(t in line.lower() for t in triggers) and len(line) > 15:
            items.append(line[:120])
    return items[:4] if items else ["Review the email and take appropriate action"]


def _draft_reply(category: EmailCategory, priority: EmailPriority, subject: str) -> str:
    if category == EmailCategory.SPAM:
        return ""  # No reply to spam
    if category in (EmailCategory.NEWSLETTER,):
        return ""
    if priority == EmailPriority.HIGH:
        return (
            f"Thank you for your message regarding '{subject[:40]}'. "
            "I have received this and will address it as a priority. "
            "I will follow up shortly with the required information."
        )
    return (
        f"Thank you for your email about '{subject[:40]}'. "
        "I will review and respond accordingly."
    )


def rule_based_action(task_id: str, obs_dict: dict) -> TriageAction:
    """Generate an action using rule-based heuristics."""
    subject = obs_dict.get("subject", "")
    body = obs_dict.get("body", "")
    category = _classify_heuristic(subject, body)
    priority = _priority_heuristic(subject, body)

    if task_id == "task_classify":
        return TriageAction(action_type=ActionType.CLASSIFY, category=category)

    elif task_id == "task_triage":
        action_hints = {
            EmailCategory.SPAM: "Mark as spam and delete",
            EmailCategory.FINANCE: f"Process payment/invoice – priority {priority.value}",
            EmailCategory.SUPPORT: "Acknowledge and investigate immediately",
            EmailCategory.WORK: f"Review and respond by deadline – {priority.value} priority",
            EmailCategory.NEWSLETTER: "Archive or unsubscribe",
            EmailCategory.PERSONAL: "Reply at your convenience",
        }
        return TriageAction(
            action_type=ActionType.TRIAGE,
            category=category,
            priority=priority,
            suggested_action=action_hints.get(category, f"Handle with {priority.value} priority"),
        )

    else:  # task_full_triage
        action_items = _extract_action_items(body)
        draft = _draft_reply(category, priority, subject)
        return TriageAction(
            action_type=ActionType.FULL_TRIAGE,
            category=category,
            priority=priority,
            suggested_action=f"Handle {category.value} email – {priority.value} priority",
            action_items=action_items,
            draft_reply=draft,
        )


# ─────────────────────────────────────────────────────────────────
# LLM baseline (OpenAI)
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant. For each email you receive,
you must analyze it and respond ONLY with a valid JSON object following the schema provided.
Do not include any explanation or markdown — pure JSON only."""


def _build_user_prompt(task_id: str, obs_dict: dict) -> str:
    subject = obs_dict.get("subject", "")
    sender = obs_dict.get("sender", "")
    body = obs_dict.get("body", "")

    base = f"""Email to triage:
Subject: {subject}
From: {sender}
Body:
{body}

"""

    if task_id == "task_classify":
        schema = '{"action_type": "classify", "category": "<spam|work|personal|newsletter|finance|support|unknown>"}'
        return base + f"Return JSON matching this schema:\n{schema}"

    elif task_id == "task_triage":
        schema = json.dumps({
            "action_type": "triage",
            "category": "<spam|work|personal|newsletter|finance|support|unknown>",
            "priority": "<high|medium|low>",
            "suggested_action": "<one-line action recommendation>",
        }, indent=2)
        return base + f"Return JSON matching this schema:\n{schema}"

    else:
        schema = json.dumps({
            "action_type": "full_triage",
            "category": "<spam|work|personal|newsletter|finance|support|unknown>",
            "priority": "<high|medium|low>",
            "suggested_action": "<one-line action recommendation>",
            "action_items": ["<item1>", "<item2>"],
            "draft_reply": "<short reply to sender, empty string if not needed>",
        }, indent=2)
        return base + f"Return JSON matching this schema:\n{schema}"


def llm_action(task_id: str, obs_dict: dict, client, model: str = "gpt-4o-mini") -> TriageAction:
    """Generate an action using the OpenAI API."""
    user_prompt = _build_user_prompt(task_id, obs_dict)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
        return TriageAction(**data)
    except Exception as e:
        print(f"  ⚠ LLM JSON parse error: {e}\n  Raw: {raw[:200]}")
        # Fallback to heuristic
        return rule_based_action(task_id, obs_dict)


# ─────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────

def run_episode(task_id: str, agent_fn) -> Tuple[float, list]:
    """Run one full episode and return (mean_score, per_step_details)."""
    env = EmailTriageEnvironment()
    obs = env.reset(task_id=task_id)
    details = []
    step_num = 0

    while not obs.done:
        step_num += 1
        action = agent_fn(task_id, obs.model_dump())
        obs, reward, done, info = env.step(action)
        details.append({
            "step": step_num,
            "email_id": info.get("breakdown", {}).get("email_id", ""),
            "reward": round(reward, 4),
            "score_breakdown": info.get("breakdown", {}),
        })
        if done:
            break

    return env.episode_score(), details


# ─────────────────────────────────────────────────────────────────
# run_baseline_in_process  (called by FastAPI /baseline endpoint)
# ─────────────────────────────────────────────────────────────────

def run_baseline_in_process():
    """Run rule-based baseline on all tasks; return (scores_dict, details_dict)."""
    tasks = ["task_classify", "task_triage", "task_full_triage"]
    scores = {}
    details = {}
    for task_id in tasks:
        score, ep_details = run_episode(task_id, rule_based_action)
        scores[task_id] = score
        details[task_id] = ep_details
    return scores, details


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Email Triage baseline agent")
    parser.add_argument("--llm", action="store_true", help="Use OpenAI LLM instead of rule-based")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--task", default=None,
                        choices=["task_classify", "task_triage", "task_full_triage"],
                        help="Run a single task only")
    args = parser.parse_args()

    tasks = (
        [args.task] if args.task
        else ["task_classify", "task_triage", "task_full_triage"]
    )

    if args.llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY environment variable to use --llm mode.")
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: Run `pip install openai` to use LLM mode.")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        def agent_fn(task_id, obs_dict):
            return llm_action(task_id, obs_dict, client, model=args.model)
        agent_label = f"LLM ({args.model})"
    else:
        agent_fn = rule_based_action
        agent_label = "Rule-based heuristic"

    print(f"\n{'='*60}")
    print(f"  Email Triage OpenEnv — Baseline Evaluation")
    print(f"  Agent: {agent_label}")
    print(f"{'='*60}\n")

    all_scores = {}
    for task_id in tasks:
        print(f"▶ Task: {task_id}")
        start = time.time()
        score, ep_details = run_episode(task_id, agent_fn)
        elapsed = time.time() - start
        all_scores[task_id] = score

        for d in ep_details:
            step = d["step"]
            rew = d["reward"]
            bar = "█" * int(rew * 10) + "░" * (10 - int(rew * 10))
            print(f"  Step {step:02d}  [{bar}]  reward={rew:.3f}")

        print(f"  ──────────────────────────────")
        print(f"  Episode score: {score:.4f}  ({elapsed:.1f}s)\n")

    print(f"{'='*60}")
    print("  SUMMARY")
    print(f"{'─'*60}")
    for task_id, s in all_scores.items():
        difficulty = {"task_classify": "easy  ", "task_triage": "medium", "task_full_triage": "hard  "}
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  [{difficulty.get(task_id, '      ')}] {task_id:<20} [{bar}] {s:.4f}")
    mean = sum(all_scores.values()) / len(all_scores)
    print(f"{'─'*60}")
    print(f"  Mean score: {mean:.4f}")
    print(f"{'='*60}\n")

    return all_scores


if __name__ == "__main__":
    main()
