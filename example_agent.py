"""
example_agent.py – Minimal example showing how to write an agent
that interacts with the Email Triage OpenEnv via HTTP.

This is what a student/participant would write to interact with
the deployed Hugging Face Space.

Run:
    python example_agent.py --task task_full_triage --llm

Or against the deployed HF Space:
    python example_agent.py --url https://YOUR-HF-SPACE.hf.space
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import re
import requests

BASE_URL = "http://localhost:7860"


# ─────────────────────────────────────────────
# Minimal rule-based agent (no imports needed)
# ─────────────────────────────────────────────

def rule_action(task_id: str, obs: dict) -> dict:
    """Dead-simple keyword-based agent."""
    subject = (obs.get("subject") or "").lower()
    body    = (obs.get("body")    or "").lower()
    text    = subject + " " + body

    # Category
    if any(w in text for w in ["won", "prize", "lottery", "processing fee"]):
        cat = "spam"
    elif any(w in text for w in ["invoice", "billing", "overdue", "aws bill"]):
        cat = "finance"
    elif any(w in text for w in ["outage", "incident", "alert", "sign-in", "security"]):
        cat = "support"
    elif any(w in text for w in ["newsletter", "unsubscribe", "tips"]):
        cat = "newsletter"
    elif any(w in text for w in ["lunch", "dinner", "friend", "weekend"]):
        cat = "personal"
    else:
        cat = "work"

    # Priority
    if any(w in text for w in ["urgent", "critical", "action required", "immediately",
                                "overdue", "outage", "friday", "15th"]):
        pri = "high"
    elif any(w in text for w in ["newsletter", "tips", "lunch", "prize"]):
        pri = "low"
    else:
        pri = "medium"

    if task_id == "task_classify":
        return {"action_type": "classify", "category": cat}

    if task_id == "task_triage":
        return {
            "action_type": "triage",
            "category": cat,
            "priority": pri,
            "suggested_action": f"Handle {cat} email with {pri} priority",
        }

    # task_full_triage
    lines = [l.strip() for l in body.split("\n") if len(l.strip()) > 15]
    items = [l for l in lines if any(t in l.lower() for t in
             ["please", "must", "submit", "review", "confirm", "pay", "check"])][:3]
    if not items:
        items = ["Review and take appropriate action"]

    draft = ""
    if cat not in ("spam", "newsletter"):
        draft = (
            f"Thank you for your email regarding '{obs.get('subject', '')[:40]}'. "
            f"I have noted this as {pri} priority and will address it accordingly."
        )

    return {
        "action_type": "full_triage",
        "category": cat,
        "priority": pri,
        "suggested_action": f"Handle {cat} email – {pri} priority",
        "action_items": items,
        "draft_reply": draft,
    }


# ─────────────────────────────────────────────
# LLM agent (OpenAI)
# ─────────────────────────────────────────────

def llm_action(task_id: str, obs: dict, client, model="gpt-4o-mini") -> dict:
    schemas = {
        "task_classify": '{"action_type":"classify","category":"<spam|work|personal|newsletter|finance|support|unknown>"}',
        "task_triage":   '{"action_type":"triage","category":"...","priority":"<high|medium|low>","suggested_action":"<one-line action>"}',
        "task_full_triage": json.dumps({
            "action_type": "full_triage",
            "category": "<spam|work|personal|newsletter|finance|support|unknown>",
            "priority": "<high|medium|low>",
            "suggested_action": "<one-line recommendation>",
            "action_items": ["<item1>", "<item2>"],
            "draft_reply": "<reply or empty string for spam/newsletters>",
        })
    }
    prompt = (
        f"Triage this email and return ONLY valid JSON matching this schema:\n"
        f"{schemas[task_id]}\n\n"
        f"Subject: {obs['subject']}\nFrom: {obs['sender']}\nBody:\n{obs['body']}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an email triage expert. Return only JSON, no markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        print(f"  ⚠ JSON parse error, falling back to rule-based")
        return rule_action(task_id, obs)


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

def run_episode(task_id: str, agent_fn, base_url: str = BASE_URL):
    """Run one episode against the HTTP server."""
    # Reset
    r = requests.post(f"{base_url}/reset", json={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()

    print(f"\n  ▶ Task: {task_id}  ({obs['emails_remaining']+1} emails)")
    step = 0

    while not obs.get("done"):
        step += 1
        action = agent_fn(task_id, obs)

        r = requests.post(f"{base_url}/step", json={"action": action})
        r.raise_for_status()
        result = r.json()

        reward = result["reward"]
        obs = result["observation"]
        bar = "█" * int(reward * 10) + "░" * (10 - int(reward * 10))
        print(f"  Step {step:02d}  [{bar}]  {reward:.3f}  {obs.get('feedback','')[:60]}")

        if result["done"]:
            break

    # Final state
    r = requests.get(f"{base_url}/state")
    state = r.json()
    final_score = state.get("cumulative_score", 0.0)
    print(f"  ─────────────────────────────")
    print(f"  Episode score: {final_score:.4f}")
    return final_score


def main():
    parser = argparse.ArgumentParser(description="Email Triage agent example")
    parser.add_argument("--url", default=BASE_URL, help="Server base URL")
    parser.add_argument("--task", default=None,
                        choices=["task_classify", "task_triage", "task_full_triage"])
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    tasks = [args.task] if args.task else ["task_classify", "task_triage", "task_full_triage"]

    # Verify server is up
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        r.raise_for_status()
        print(f"✅ Server at {args.url} is healthy")
    except Exception as e:
        print(f"❌ Could not reach server at {args.url}: {e}")
        sys.exit(1)

    if args.llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Set OPENAI_API_KEY for LLM mode.")
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        agent_fn = lambda t, o: llm_action(t, o, client, args.model)
        label = f"LLM ({args.model})"
    else:
        agent_fn = rule_action
        label = "Rule-based"

    print(f"\n{'='*55}")
    print(f"  Email Triage Agent  —  {label}")
    print(f"{'='*55}")

    all_scores = {}
    for task_id in tasks:
        all_scores[task_id] = run_episode(task_id, agent_fn, base_url=args.url)

    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'─'*55}")
    for tid, sc in all_scores.items():
        bar = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
        print(f"  {tid:<22} [{bar}] {sc:.4f}")
    mean = sum(all_scores.values()) / len(all_scores)
    print(f"{'─'*55}")
    print(f"  Mean: {mean:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
