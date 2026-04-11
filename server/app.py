"""
server/app.py - Email Triage OpenEnv — SINGLE FILE IMPLEMENTATION
All logic inlined: models, data, graders, environment, API.
Scores guaranteed in (0.1, 0.9) — mathematically impossible to be 0.0 or 1.0.
REBUILD: 2026-04-12
"""
from __future__ import annotations
import os, sys, time, copy
from uuid import uuid4
from typing import Optional, List, Tuple
from difflib import SequenceMatcher
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════
# SCORE SAFETY — mathematical guarantee: output always in (0.1, 0.9)
# ═══════════════════════════════════════════════════════════════════
def S(raw: float) -> float:
    """Map any float → (0.1, 0.9). Impossible to return 0.0 or 1.0."""
    try:    v = max(0.0, min(1.0, float(raw)))
    except: v = 0.5
    return round(0.1 + v * 0.8, 4)

# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════
class EmailCategory(str, Enum):
    SPAM="spam"; WORK="work"; PERSONAL="personal"
    NEWSLETTER="newsletter"; FINANCE="finance"
    SUPPORT="support"; UNKNOWN="unknown"

class EmailPriority(str, Enum):
    HIGH="high"; MEDIUM="medium"; LOW="low"

class ActionType(str, Enum):
    CLASSIFY="classify"; TRIAGE="triage"; FULL_TRIAGE="full_triage"

# ═══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════
class TriageAction(BaseModel):
    action_type:      ActionType
    category:         EmailCategory   = EmailCategory.UNKNOWN
    priority:         Optional[EmailPriority] = None
    suggested_action: Optional[str]   = None
    action_items:     Optional[List[str]] = None
    draft_reply:      Optional[str]   = None

class EmailObservation(BaseModel):
    email_id:         str
    subject:          str
    sender:           str
    body:             str
    thread_length:    int   = 1
    reward:           float = 0.5
    done:             bool  = False
    success:          bool  = False
    emails_remaining: int   = 0
    cumulative_score: float = 0.5
    feedback:         Optional[str] = None

class TriageState(BaseModel):
    episode_id:       str
    task_id:          str
    step_count:       int
    total_emails:     int
    emails_completed: int
    cumulative_score: float
    actions_log:      List[dict] = []

class GraderResult(BaseModel):
    task_id:    str
    episode_id: str
    score:      float
    breakdown:  dict
    passed:     bool

# ═══════════════════════════════════════════════════════════════════
# EMAIL DATA (inline — no import needed)
# ═══════════════════════════════════════════════════════════════════
EMAILS = [
    {"email_id":"e001","subject":"Q3 Budget Review – Action Required by Friday",
     "sender":"cfo@acmecorp.com","thread_length":1,
     "body":"Please review the Q3 budget spreadsheet and confirm your department allocations before Friday COB. Variances above 5% need written justification submitted to finance@acmecorp.com.\n\nRegards,\nSarah (CFO)",
     "gt_category":EmailCategory.WORK,"gt_priority":EmailPriority.HIGH,
     "gt_action_items":["Review Q3 budget spreadsheet","Confirm allocations by Friday COB","Submit justification for variances >5%"],
     "gt_reply_keywords":["confirm","budget","friday","allocations"]},
    {"email_id":"e002","subject":"CONGRATULATIONS! You've won $1,000,000!!!",
     "sender":"lottery.winner99@prizeclaim.biz","thread_length":1,
     "body":"You have been selected as the grand prize winner. Send your bank details and a $50 processing fee to claim your prize.",
     "gt_category":EmailCategory.SPAM,"gt_priority":EmailPriority.LOW,
     "gt_action_items":["Mark as spam","Do not reply"],"gt_reply_keywords":[]},
    {"email_id":"e003","subject":"Server outage – production API down",
     "sender":"alerts@monitoring.internal","thread_length":1,
     "body":"CRITICAL ALERT: The production API server has been unreachable for 15 minutes. Error rate: 100%. Incident ID: INC-4821.",
     "gt_category":EmailCategory.SUPPORT,"gt_priority":EmailPriority.HIGH,
     "gt_action_items":["Acknowledge incident INC-4821","Investigate outage","Notify affected teams"],
     "gt_reply_keywords":["acknowledge","incident","investigating"]},
    {"email_id":"e004","subject":"Monthly newsletter – Top 10 productivity tips",
     "sender":"noreply@productivityweekly.io","thread_length":1,
     "body":"Here are this month's top 10 productivity tips. You are receiving this because you subscribed. Unsubscribe anytime.",
     "gt_category":EmailCategory.NEWSLETTER,"gt_priority":EmailPriority.LOW,
     "gt_action_items":["Archive or unsubscribe"],"gt_reply_keywords":[]},
    {"email_id":"e005","subject":"Invoice #INV-2024-0892 overdue – 30 days",
     "sender":"billing@cloudstorage.com","thread_length":1,
     "body":"Invoice #INV-2024-0892 for $349.00 is 30 days overdue. Please settle payment to avoid service suspension.",
     "gt_category":EmailCategory.FINANCE,"gt_priority":EmailPriority.HIGH,
     "gt_action_items":["Pay invoice #INV-2024-0892","Confirm payment"],
     "gt_reply_keywords":["payment","invoice","settle"]},
    {"email_id":"e006","subject":"Lunch plans Saturday?",
     "sender":"alex.friend@gmail.com","thread_length":1,
     "body":"Hey! Are you free for lunch on Saturday around noon? Thinking of trying that new Thai place. Let me know!",
     "gt_category":EmailCategory.PERSONAL,"gt_priority":EmailPriority.LOW,
     "gt_action_items":["Reply to confirm or decline Saturday lunch"],
     "gt_reply_keywords":["saturday","lunch","thai"]},
    {"email_id":"e007","subject":"Re: Contract renewal – legal review needed",
     "sender":"legal@partnerfirm.com","thread_length":3,
     "body":"Our legal team flagged clauses in Section 4 (Liability) and Section 7 (IP Rights). We need a response by the 15th for the January 1 go-live.",
     "gt_category":EmailCategory.WORK,"gt_priority":EmailPriority.HIGH,
     "gt_action_items":["Forward to in-house counsel","Review Sections 4 and 7","Respond by the 15th"],
     "gt_reply_keywords":["counsel","review","15th","contract"]},
    {"email_id":"e008","subject":"Your AWS bill – $4,287 this month",
     "sender":"billing@amazonaws.com","thread_length":1,
     "body":"Your AWS account has been charged $4,287.43, which is 340% above your normal spend. Largest cost: EC2 us-east-1 $3,100.",
     "gt_category":EmailCategory.FINANCE,"gt_priority":EmailPriority.HIGH,
     "gt_action_items":["Investigate EC2 us-east-1 charges","Review data transfer costs","Set up billing alerts"],
     "gt_reply_keywords":["investigate","EC2","billing","costs"]},
    {"email_id":"e009","subject":"Team offsite – venue poll",
     "sender":"hr@acmecorp.com","thread_length":1,
     "body":"We are planning the Q4 team offsite and need your input. Please fill in the quick poll by EOD Thursday.",
     "gt_category":EmailCategory.WORK,"gt_priority":EmailPriority.MEDIUM,
     "gt_action_items":["Complete venue poll by EOD Thursday"],
     "gt_reply_keywords":["poll","completed","thursday"]},
    {"email_id":"e010","subject":"Security alert: new sign-in from unknown device",
     "sender":"security@accounts.google.com","thread_length":1,
     "body":"We noticed a new sign-in from an unrecognized device in Lagos, Nigeria at 03:14 AM UTC. If this was not you, secure your account immediately.",
     "gt_category":EmailCategory.SUPPORT,"gt_priority":EmailPriority.HIGH,
     "gt_action_items":["Verify if login was authorized","Change password if suspicious","Enable 2FA"],
     "gt_reply_keywords":["password","secure","2FA","unauthorized"]},
]
EMAIL_BY_ID = {e["email_id"]: e for e in EMAILS}
TASK_EMAILS = {
    "task_classify":    ["e001","e002","e003","e004","e006"],
    "task_triage":      ["e001","e003","e005","e007","e009","e010"],
    "task_full_triage": [e["email_id"] for e in EMAILS],
}
ACTION_TYPE_MAP = {
    "task_classify":    ActionType.CLASSIFY,
    "task_triage":      ActionType.TRIAGE,
    "task_full_triage": ActionType.FULL_TRIAGE,
}

# ═══════════════════════════════════════════════════════════════════
# GRADERS (inline)
# ═══════════════════════════════════════════════════════════════════
def _sim(a,b):
    if not a and not b: return 0.85
    if not a or not b:  return 0.15
    return SequenceMatcher(None,a.lower(),b.lower()).ratio()

def _kw(text, keywords):
    if not keywords: return 0.85
    if not text:     return 0.15
    hits = sum(1 for k in keywords if k.lower() in text.lower())
    return max(0.15, min(0.85, hits/len(keywords)))

def _items_score(submitted, reference):
    if not reference:  return 0.85
    if not submitted:  return 0.15
    scores = [max(_sim(ref,s) for s in submitted) for ref in reference]
    return max(0.15, min(0.85, sum(scores)/len(scores)))

def _grade_classify(email_id, action):
    e    = EMAIL_BY_ID[email_id]
    gt   = e["gt_category"]
    pred = action.category
    if pred == gt:
        raw = 0.95
    elif (pred == EmailCategory.SPAM) != (gt == EmailCategory.SPAM):
        raw = 0.05
    elif {pred,gt} in ({EmailCategory.WORK,EmailCategory.SUPPORT},
                       {EmailCategory.WORK,EmailCategory.FINANCE},
                       {EmailCategory.NEWSLETTER,EmailCategory.PERSONAL}):
        raw = 0.50
    else:
        raw = 0.20
    score = S(raw)
    return score, {"gt":gt.value,"pred":pred.value,"score":score}

def _grade_triage(email_id, action):
    e      = EMAIL_BY_ID[email_id]
    gt_pri = e["gt_priority"]
    gt_kw  = e["gt_reply_keywords"]
    cs, cb = _grade_classify(email_id, action)
    pp     = action.priority or EmailPriority.MEDIUM
    if pp == gt_pri:                                    pri_raw = 0.95
    elif abs(["low","medium","high"].index(pp.value) -
             ["low","medium","high"].index(gt_pri.value)) == 1: pri_raw = 0.50
    else:                                               pri_raw = 0.05
    ps  = S(pri_raw)
    act = action.suggested_action or ""
    acs = S(_kw(act,gt_kw)) if gt_kw else (S(0.75) if act else S(0.20))
    raw = 0.30*cs + 0.40*ps + 0.30*acs
    score = S(raw)
    return score, {"category":cb,"priority":{"gt":gt_pri.value,"pred":pp.value,"score":ps},
                   "action":{"score":acs},"score":score}

def _grade_full(email_id, action):
    e      = EMAIL_BY_ID[email_id]
    gt_i   = e["gt_action_items"]
    gt_kw  = e["gt_reply_keywords"]
    ts, tb = _grade_triage(email_id, action)
    iscore = S(_items_score(action.action_items or [], gt_i))
    draft  = action.draft_reply or ""
    if gt_kw:
        rs = S(_kw(draft,gt_kw) + (0.10 if len(draft)>30 else 0))
    else:
        rs = S(0.10) if (draft and len(draft)>20) else S(0.90)
    raw = 0.40*ts + 0.35*iscore + 0.25*rs
    score = S(raw)
    return score, {"triage":tb,"items":{"score":iscore},"reply":{"score":rs},"score":score}

GRADER_MAP = {
    "task_classify":    _grade_classify,
    "task_triage":      _grade_triage,
    "task_full_triage": _grade_full,
}

def do_grade(task_id, email_id, action):
    fn = GRADER_MAP.get(task_id)
    if not fn: raise ValueError(f"Unknown task: {task_id}")
    score, bd = fn(email_id, action)
    # Absolute last safety net
    score = round(max(0.001, min(0.999, float(score))), 4)
    return score, bd

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT (inline)
# ═══════════════════════════════════════════════════════════════════
class EmailTriageEnvironment:
    def __init__(self):
        self._state = None
        self._queue = []
        self._current = None
        self._scores = []

    def reset(self, task_id="task_classify"):
        if task_id not in TASK_EMAILS: task_id="task_classify"
        self._queue   = [copy.deepcopy(EMAIL_BY_ID[eid]) for eid in TASK_EMAILS[task_id]]
        self._scores  = []
        self._current = None
        self._state   = TriageState(
            episode_id=str(uuid4()), task_id=task_id, step_count=0,
            total_emails=len(self._queue), emails_completed=0,
            cumulative_score=S(0.5), actions_log=[])
        return self._make_obs(S(0.5), False, True, "Episode started.")

    def step(self, action):
        if not self._state: raise RuntimeError("Call reset() first.")
        expected = ACTION_TYPE_MAP.get(self._state.task_id)
        if action.action_type != expected:
            action = action.model_copy(update={"action_type": expected})
        if not self._current:
            return self._make_obs(S(0.1),True,False,"No email."), S(0.1), True, {}
        score, bd = do_grade(self._state.task_id, self._current["email_id"], action)
        score = round(max(0.001, min(0.999, float(score))), 4)
        self._scores.append(score)
        self._state.step_count      += 1
        self._state.emails_completed += 1
        cum = round(max(0.001,min(0.999, sum(self._scores)/len(self._scores))),4)
        self._state.cumulative_score  = cum
        self._state.actions_log.append({"step":self._state.step_count,
            "email_id":self._current["email_id"],"score":score})
        done = len(self._queue)==0
        obs  = self._make_obs(score, done, True, f"Score: {score:.3f}")
        return obs, score, done, {"breakdown":bd,"episode_id":self._state.episode_id}

    @property
    def state(self): return self._state

    def _make_obs(self, reward, done, success, feedback):
        if self._queue and not done:
            self._current = self._queue.pop(0)
        elif done:
            self._current = None
        e   = self._current
        cum = round(max(0.001,min(0.999,
              self._state.cumulative_score if self._state else S(0.5))),4)
        reward = round(max(0.001, min(0.999, float(reward))), 4)
        if e is None:
            return EmailObservation(email_id="",subject="[Done]",sender="",
                body="Episode complete.",thread_length=0,reward=reward,done=True,
                success=success,emails_remaining=0,cumulative_score=cum,feedback=feedback)
        return EmailObservation(email_id=e["email_id"],subject=e["subject"],
            sender=e["sender"],body=e["body"],thread_length=e["thread_length"],
            reward=reward,done=done,success=success,
            emails_remaining=len(self._queue),cumulative_score=cum,feedback=feedback)

    def episode_score(self):
        if not self._scores: return S(0.5)
        return round(max(0.001,min(0.999, sum(self._scores)/len(self._scores))),4)

# ═══════════════════════════════════════════════════════════════════
# BASELINE (inline rule-based agent)
# ═══════════════════════════════════════════════════════════════════
def _rule_action(task_id, obs_dict):
    text = (obs_dict.get("subject","") + " " + obs_dict.get("body","")).lower()
    cat  = (EmailCategory.SPAM      if any(w in text for w in ["lottery","prize","won","processing fee"]) else
            EmailCategory.FINANCE   if any(w in text for w in ["invoice","billing","overdue","aws bill"]) else
            EmailCategory.SUPPORT   if any(w in text for w in ["outage","alert","incident","sign-in","security"]) else
            EmailCategory.NEWSLETTER if any(w in text for w in ["newsletter","unsubscribe","tips"]) else
            EmailCategory.PERSONAL  if any(w in text for w in ["lunch","dinner","friend","saturday"]) else
            EmailCategory.WORK)
    pri  = (EmailPriority.HIGH if any(w in text for w in ["urgent","critical","action required","overdue","outage","friday","15th"]) else
            EmailPriority.LOW  if any(w in text for w in ["newsletter","lunch","prize","poll"]) else
            EmailPriority.MEDIUM)
    if task_id=="task_classify":
        return TriageAction(action_type=ActionType.CLASSIFY, category=cat)
    if task_id=="task_triage":
        return TriageAction(action_type=ActionType.TRIAGE, category=cat, priority=pri,
                            suggested_action=f"Handle {cat.value} with {pri.value} priority")
    lines = [l.strip() for l in obs_dict.get("body","").split("\n") if len(l.strip())>15]
    items = [l for l in lines if any(t in l.lower() for t in
             ["please","must","submit","review","confirm","pay","check"])][:3] or ["Review and respond"]
    draft = ("" if cat in (EmailCategory.SPAM, EmailCategory.NEWSLETTER) else
             f"Thank you for your email. I will handle this with {pri.value} priority.")
    return TriageAction(action_type=ActionType.FULL_TRIAGE, category=cat, priority=pri,
                        suggested_action=f"Handle {cat.value} – {pri.value}",
                        action_items=items, draft_reply=draft)

def run_baseline_in_process():
    scores, details = {}, {}
    for task_id in ["task_classify","task_triage","task_full_triage"]:
        env = EmailTriageEnvironment()
        obs = env.reset(task_id=task_id)
        eps = []
        while not obs.done:
            action = _rule_action(task_id, obs.model_dump())
            obs, reward, done, info = env.step(action)
            # Triple-clamp every reward
            reward = round(max(0.001, min(0.999, float(reward))), 4)
            eps.append({"step": env._state.step_count, "reward": reward})
            if done: break
        final = round(max(0.001, min(0.999, float(env.episode_score()))), 4)
        scores[task_id]  = final
        details[task_id] = eps
    return scores, details

# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(title="Email Triage OpenEnv", version="1.0.0", docs_url="/docs")
_env = EmailTriageEnvironment()

class ResetRequest(BaseModel):
    task_id: str = "task_classify"
class StepRequest(BaseModel):
    action: TriageAction
class GraderRequest(BaseModel):
    task_id: str; email_id: str; action: TriageAction

@app.get("/")
def root():
    return {"environment":"Email Triage OpenEnv","version":"1.0.0",
            "tasks":list(TASK_EMAILS.keys()),
            "endpoints":["/reset","/step","/state","/tasks","/grader","/baseline","/docs"]}

@app.get("/health")
def health():
    return {"status":"ok","timestamp":time.time()}

@app.post("/reset", response_model=EmailObservation)
async def reset(body: Optional[ResetRequest]=None):
    if body is None: body = ResetRequest()
    try: return _env.reset(task_id=body.task_id)
    except Exception as e: raise HTTPException(400, str(e))

@app.post("/step")
def step(body: StepRequest):
    try:
        obs, reward, done, info = _env.step(body.action)
        reward = round(max(0.001, min(0.999, float(reward))), 4)
        return {"observation":obs.model_dump(),"reward":reward,"done":done,"info":info}
    except RuntimeError as e: raise HTTPException(400, str(e))

@app.get("/state", response_model=TriageState)
def state():
    try: return _env.state
    except RuntimeError as e: raise HTTPException(400, str(e))

@app.get("/tasks")
def tasks():
    return {"tasks":[
        {"task_id":"task_classify","name":"Email Classification","difficulty":"easy",
         "description":"Classify emails into one of 6 categories.",
         "num_emails":len(TASK_EMAILS["task_classify"]),
         "action_schema":{"action_type":"classify","required_fields":["category"],"optional_fields":[],
                          "category_options":[e.value for e in EmailCategory]}},
        {"task_id":"task_triage","name":"Email Triage","difficulty":"medium",
         "description":"Classify + priority + suggested action.",
         "num_emails":len(TASK_EMAILS["task_triage"]),
         "action_schema":{"action_type":"triage","required_fields":["category","priority","suggested_action"],
                          "optional_fields":[],"category_options":[e.value for e in EmailCategory],
                          "priority_options":[p.value for p in EmailPriority]}},
        {"task_id":"task_full_triage","name":"Full Email Triage","difficulty":"hard",
         "description":"Full 5-field triage pipeline.",
         "num_emails":len(TASK_EMAILS["task_full_triage"]),
         "action_schema":{"action_type":"full_triage",
                          "required_fields":["category","priority","suggested_action","action_items","draft_reply"],
                          "optional_fields":[]}},
    ]}

@app.post("/grader", response_model=GraderResult)
def grader(body: GraderRequest):
    if body.email_id not in EMAIL_BY_ID: raise HTTPException(404,f"email_id not found")
    if body.task_id  not in TASK_EMAILS: raise HTTPException(404,f"task_id not found")
    score, bd = do_grade(body.task_id, body.email_id, body.action)
    score = round(max(0.001, min(0.999, float(score))), 4)
    return GraderResult(task_id=body.task_id,episode_id="standalone",
                        score=score,breakdown=bd,passed=score>=0.5)

@app.get("/baseline")
def baseline():
    raw_scores, details = run_baseline_in_process()
    scores = {k: round(max(0.001,min(0.999,float(v))),4) for k,v in raw_scores.items()}
    mean   = round(max(0.001,min(0.999, sum(scores.values())/len(scores))),4)
    return {"scores":scores,"mean_score":mean,"details":details}

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1)

if __name__ == "__main__":
    main()
