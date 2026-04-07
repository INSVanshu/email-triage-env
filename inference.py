"""
inference.py – Hackathon submission inference script.
Follows the required structured logging format: [START] / [STEP] / [END]
Uses OpenAI client configured via environment variables.
"""
import os
import sys
import json
import re

# ── Required environment variables ──────────────────────────────
API_BASE_URL     = os.environ["API_BASE_URL"]          # injected by hackathon validator – required
API_KEY          = os.environ["API_KEY"]               # injected by hackathon validator – required
MODEL_NAME       = os.getenv("MODEL_NAME", "gpt-4o-mini")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")       # optional – no default

# ── OpenAI client routed through the hackathon LiteLLM proxy ────
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

# ── Environment HTTP client ──────────────────────────────────────
import urllib.request
import urllib.error

def env_request(path: str, method: str = "GET", body: dict = None) -> dict:
    """Make HTTP request to the OpenEnv environment server."""
    url = API_BASE_URL.rstrip("/") + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        raise RuntimeError(f"HTTP {e.code} on {url}: {body_text}")


# ── LLM triage agent ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert email triage assistant. "
    "Respond ONLY with a valid JSON object — no markdown, no explanation."
)

SCHEMAS = {
    "task_classify": {
        "action_type": "classify",
        "category": "<spam|work|personal|newsletter|finance|support|unknown>"
    },
    "task_triage": {
        "action_type": "triage",
        "category": "<spam|work|personal|newsletter|finance|support|unknown>",
        "priority": "<high|medium|low>",
        "suggested_action": "<one-line recommendation>"
    },
    "task_full_triage": {
        "action_type": "full_triage",
        "category": "<spam|work|personal|newsletter|finance|support|unknown>",
        "priority": "<high|medium|low>",
        "suggested_action": "<one-line recommendation>",
        "action_items": ["<item1>", "<item2>"],
        "draft_reply": "<short reply or empty string for spam/newsletters>"
    }
}


def llm_action(task_id: str, obs: dict) -> dict:
    """Call the LLM to produce a triage action for the current email."""
    schema = json.dumps(SCHEMAS[task_id], indent=2)
    user_prompt = (
        f"Triage this email. Return JSON matching this schema:\n{schema}\n\n"
        f"Subject: {obs.get('subject', '')}\n"
        f"From: {obs.get('sender', '')}\n"
        f"Body:\n{obs.get('body', '')}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except Exception:
        # Fallback: rule-based if LLM returns unparseable output
        return _rule_fallback(task_id, obs)


def _rule_fallback(task_id: str, obs: dict) -> dict:
    """Simple keyword fallback used when LLM output can't be parsed."""
    text = (obs.get("subject","") + " " + obs.get("body","")).lower()
    if any(w in text for w in ["won","prize","lottery","processing fee"]):
        cat = "spam"
    elif any(w in text for w in ["invoice","billing","overdue","aws bill"]):
        cat = "finance"
    elif any(w in text for w in ["outage","incident","alert","sign-in","security"]):
        cat = "support"
    elif any(w in text for w in ["newsletter","unsubscribe","tips"]):
        cat = "newsletter"
    elif any(w in text for w in ["lunch","dinner","friend","weekend"]):
        cat = "personal"
    else:
        cat = "work"

    pri = "high" if any(w in text for w in ["urgent","critical","action required","overdue","outage"]) else "medium"

    if task_id == "task_classify":
        return {"action_type": "classify", "category": cat}
    if task_id == "task_triage":
        return {"action_type": "triage", "category": cat, "priority": pri,
                "suggested_action": f"Handle {cat} email with {pri} priority"}
    return {
        "action_type": "full_triage", "category": cat, "priority": pri,
        "suggested_action": f"Handle {cat} email – {pri} priority",
        "action_items": ["Review and take appropriate action"],
        "draft_reply": "" if cat in ("spam","newsletter") else
                       f"Thank you for your email. I will respond with {pri} priority."
    }


# ── Episode runner with required structured logging ───────────────

def run_episode(task_id: str) -> float:
    """
    Run one full episode against the environment server.
    Logs in the required [START] / [STEP] / [END] structured format.
    """
    step_num = 0
    rewards = []
    last_error = None

    # ── [START] ──────────────────────────────────────────────────
    print(f"[START] task={task_id} env={API_BASE_URL} model={MODEL_NAME}", flush=True)

    try:
        # Reset environment
        obs = env_request("/reset", method="POST", body={"task_id": task_id})

        while not obs.get("done", False):
            step_num += 1

            # Get LLM action
            action = llm_action(task_id, obs)
            # Compact single-line action repr for the log
            action_str = json.dumps(action, separators=(',', ':'))

            # Submit to environment
            result = env_request("/step", method="POST", body={"action": action})
            reward  = result.get("reward", 0.0)
            obs     = result.get("observation", result)
            done    = result.get("done", obs.get("done", False))
            last_error = obs.get("last_action_error", None)

            rewards.append(reward)

            # ── [STEP] ───────────────────────────────────────────
            error_val = last_error if last_error else "null"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={error_val}",
                flush=True
            )

            if done:
                break

        # Get final state
        state = env_request("/state")
        final_score = state.get("cumulative_score", 0.0)
        success = final_score >= 0.5

    except Exception as e:
        last_error = str(e)
        final_score = 0.0
        success = False
        # Emit an error STEP so the validator sees at least one STEP line
        print(
            f"[STEP] step={step_num + 1} action=null reward=0.00 "
            f"done=true error={last_error}",
            flush=True
        )

    # ── [END] ────────────────────────────────────────────────────
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] task={task_id} success={'true' if success else 'false'} "
        f"steps={step_num} rewards={rewards_str} score={final_score:.2f}",
        flush=True
    )

    return final_score


# ── Main ─────────────────────────────────────────────────────────

def main():
    tasks = ["task_classify", "task_triage", "task_full_triage"]
    scores = {}

    for task_id in tasks:
        score = run_episode(task_id)
        scores[task_id] = score

    mean_score = round(sum(scores.values()) / len(scores), 4)
    # Final summary to stderr so it doesn't interfere with stdout parsing
    print(
        f"[END] event=inference_run mean_score={mean_score:.2f} "
        f"passed={'true' if mean_score >= 0.5 else 'false'}",
        flush=True
    )

    return scores


if __name__ == "__main__":
    main()
