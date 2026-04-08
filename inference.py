import os
import re
import json
import urllib.request
from openai import OpenAI

# ── Environment variables (exactly matching sample) ──────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client (exactly matching sample) ──────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── Environment server URL ────────────────────────────────────────
ENV_URL = "https://Vansh051201-email-triage-env.hf.space"

def env_post(path, body=None):
    url  = ENV_URL + path
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def env_get(path):
    url = ENV_URL + path
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


# ── LLM triage function ───────────────────────────────────────────

SCHEMAS = {
    "task_classify": '{"action_type":"classify","category":"<spam|work|personal|newsletter|finance|support|unknown>"}',
    "task_triage":   '{"action_type":"triage","category":"<spam|work|personal|newsletter|finance|support|unknown>","priority":"<high|medium|low>","suggested_action":"<one-line recommendation>"}',
    "task_full_triage": '{"action_type":"full_triage","category":"<spam|work|personal|newsletter|finance|support|unknown>","priority":"<high|medium|low>","suggested_action":"<one-line>","action_items":["<item1>"],"draft_reply":"<reply or empty>"}',
}

def run_inference(prompt: str) -> str:
    """Call the LLM via the injected API_BASE_URL proxy."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_action(task_id: str, obs: dict) -> dict:
    schema = SCHEMAS[task_id]
    prompt = (
        f"Triage this email. Return ONLY valid JSON matching: {schema}\n\n"
        f"Subject: {obs.get('subject','')}\n"
        f"From: {obs.get('sender','')}\n"
        f"Body: {obs.get('body','')}"
    )
    raw = run_inference(prompt)
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        # Safe fallback
        text = (obs.get("subject","") + obs.get("body","")).lower()
        cat  = "spam" if "lottery" in text or "prize" in text else \
               "finance" if "invoice" in text or "billing" in text else \
               "support" if "outage" in text or "alert" in text else \
               "newsletter" if "unsubscribe" in text else "work"
        pri  = "high" if any(w in text for w in ["urgent","critical","overdue","outage"]) else "medium"
        if task_id == "task_classify":
            return {"action_type": "classify", "category": cat}
        if task_id == "task_triage":
            return {"action_type": "triage", "category": cat, "priority": pri,
                    "suggested_action": f"Handle {cat} email"}
        return {"action_type": "full_triage", "category": cat, "priority": pri,
                "suggested_action": f"Handle {cat} email",
                "action_items": ["Review and respond appropriately"],
                "draft_reply": "" if cat in ("spam","newsletter") else "Thank you, I will follow up shortly."}


# ── Episode runner with required structured stdout output ─────────

def run_episode(task_id: str) -> float:
    print(f"[START] task={task_id} model={MODEL_NAME}", flush=True)

    obs      = env_post("/reset", {"task_id": task_id})
    step_num = 0

    while not obs.get("done", False):
        step_num += 1
        action = get_action(task_id, obs)
        result = env_post("/step", {"action": action})
        reward = result.get("reward", 0.0)
        obs    = result.get("observation", result)
        done   = result.get("done", obs.get("done", False))

        print(f"[STEP] task={task_id} step={step_num} reward={round(reward,4)} done={done}", flush=True)

        if done:
            break

    state       = env_get("/state")
    final_score = state.get("cumulative_score", 0.0)

    print(f"[END] task={task_id} score={round(final_score,4)} steps={step_num}", flush=True)
    return final_score


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks  = ["task_classify", "task_triage", "task_full_triage"]
    scores = {}

    print(f"[START] event=inference_run model={MODEL_NAME}", flush=True)

    for task_id in tasks:
        try:
            scores[task_id] = run_episode(task_id)
        except Exception as e:
            print(f"[STEP] task={task_id} error={e}", flush=True)
            scores[task_id] = 0.0

    mean = round(sum(scores.values()) / len(scores), 4)

    print(
        f"[END] event=inference_run "
        f"task_classify={scores.get('task_classify',0)} "
        f"task_triage={scores.get('task_triage',0)} "
        f"task_full_triage={scores.get('task_full_triage',0)} "
        f"mean={mean}",
        flush=True
    )
