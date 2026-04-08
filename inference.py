"""
inference.py – Hackathon submission inference script.
Uses API_BASE_URL and API_KEY injected by the validator.
Prints required [START] / [STEP] / [END] structured stdout blocks.
"""
import os
import sys
import json
import re

# ── Environment variables — injected by the hackathon validator ──
API_BASE_URL     = os.getenv("API_BASE_URL", "https://Vansh051201-email-triage-env.hf.space")
MODEL_NAME       = os.getenv("MODEL_NAME",   "gpt-4o-mini")
API_KEY          = os.getenv("API_KEY")           # injected by validator — no default
HF_TOKEN         = os.getenv("HF_TOKEN")          # optional — no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional — no default

# ── OpenAI client — uses API_BASE_URL and API_KEY from env ───────
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY or HF_TOKEN or "dummy-key",
    base_url=API_BASE_URL,
)

# ── Environment HTTP client (no extra deps) ──────────────────────
import urllib.request
import urllib.error

ENV_BASE = os.getenv("API_BASE_URL", "https://Vansh051201-email-triage-env.hf.space")

def env_post(path, body=None):
    url  = ENV_BASE.rstrip("/") + path
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def env_get(path):
    url = ENV_BASE.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


# ── LLM triage via the validator's proxy ─────────────────────────

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

def llm_action(task_id, obs):
    """Call the LLM proxy to produce a triage action."""
    schema = json.dumps(SCHEMAS[task_id], indent=2)
    user_prompt = (
        f"Triage this email. Return JSON matching this schema:\n{schema}\n\n"
        f"Subject: {obs.get('subject','')}\n"
        f"From: {obs.get('sender','')}\n"
        f"Body:\n{obs.get('body','')}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        return _rule_fallback(task_id, obs)

def _rule_fallback(task_id, obs):
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


# ── Episode runner with required structured stdout logging ────────

def run_episode(task_id):
    # [START] block
    print(f"[START] task={task_id} model={MODEL_NAME}", flush=True)

    obs = env_post("/reset", {"task_id": task_id})
    step_num = 0

    while not obs.get("done", False):
        step_num += 1
        action = llm_action(task_id, obs)
        result = env_post("/step", {"action": action})
        reward  = result.get("reward", 0.0)
        obs     = result.get("observation", result)
        done    = result.get("done", obs.get("done", False))

        # [STEP] block
        print(
            f"[STEP] task={task_id} step={step_num} "
            f"reward={round(reward,4)} done={done}",
            flush=True
        )

        if done:
            break

    state = env_get("/state")
    final_score = state.get("cumulative_score", 0.0)

    # [END] block
    print(
        f"[END] task={task_id} score={round(final_score,4)} "
        f"steps={step_num} passed={final_score >= 0.5}",
        flush=True
    )

    return final_score


# ── Main ──────────────────────────────────────────────────────────

def main():
    tasks = ["task_classify", "task_triage", "task_full_triage"]
    scores = {}

    print(f"[START] event=inference_run tasks={tasks} model={MODEL_NAME}", flush=True)

    for task_id in tasks:
        try:
            scores[task_id] = run_episode(task_id)
        except Exception as e:
            print(f"[STEP] task={task_id} error={str(e)}", flush=True)
            scores[task_id] = 0.0

    mean_score = round(sum(scores.values()) / len(scores), 4)

    print(
        f"[END] event=inference_run "
        f"task_classify={scores.get('task_classify',0)} "
        f"task_triage={scores.get('task_triage',0)} "
        f"task_full_triage={scores.get('task_full_triage',0)} "
        f"mean={mean_score}",
        flush=True
    )

    return scores


if __name__ == "__main__":
    main()
