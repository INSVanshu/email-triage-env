from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from models import (TriageAction, EmailObservation, TriageState, GraderResult,
                    ActionType, EmailCategory, EmailPriority)
from data import TASK_EMAILS, EMAIL_BY_ID
from graders import grade
from server.environment import EmailTriageEnvironment

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0", docs_url="/docs")
_env = EmailTriageEnvironment()

def _clamp(v):
    try: return round(max(0.001, min(0.999, float(v))), 4)
    except: return 0.001

class ResetRequest(BaseModel):
    task_id: str = "task_classify"
class StepRequest(BaseModel):
    action: TriageAction
class GraderRequest(BaseModel):
    task_id: str; email_id: str; action: TriageAction
class BaselineResponse(BaseModel):
    scores: dict; mean_score: float; details: dict

@app.get("/")
def root():
    return {"environment": "Email Triage OpenEnv", "version": "1.0.0",
            "tasks": list(TASK_EMAILS.keys()),
            "endpoints": ["/reset","/step","/state","/tasks","/grader","/baseline","/docs"]}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/reset", response_model=EmailObservation)
async def reset(body: Optional[ResetRequest] = None):
    if body is None: body = ResetRequest()
    try: return _env.reset(task_id=body.task_id)
    except Exception as e: raise HTTPException(400, str(e))

@app.post("/step")
def step(body: StepRequest):
    try:
        obs, reward, done, info = _env.step(body.action)
        return {"observation": obs.model_dump(),
                "reward": _clamp(reward),   # clamp here too
                "done": done, "info": info}
    except RuntimeError as e: raise HTTPException(400, str(e))

@app.get("/state", response_model=TriageState)
def state():
    try: return _env.state
    except RuntimeError as e: raise HTTPException(400, str(e))

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"task_id": "task_classify", "name": "Email Classification", "difficulty": "easy",
         "description": "Classify each email into one of 6 categories.",
         "num_emails": len(TASK_EMAILS["task_classify"]),
         "action_schema": {"action_type": "classify", "required_fields": ["category"],
                           "optional_fields": [], "category_options": [e.value for e in EmailCategory]}},
        {"task_id": "task_triage", "name": "Email Triage", "difficulty": "medium",
         "description": "Classify + priority + suggested action.",
         "num_emails": len(TASK_EMAILS["task_triage"]),
         "action_schema": {"action_type": "triage",
                           "required_fields": ["category","priority","suggested_action"],
                           "optional_fields": [],
                           "category_options": [e.value for e in EmailCategory],
                           "priority_options": [p.value for p in EmailPriority]}},
        {"task_id": "task_full_triage", "name": "Full Email Triage", "difficulty": "hard",
         "description": "Full 5-field triage pipeline.",
         "num_emails": len(TASK_EMAILS["task_full_triage"]),
         "action_schema": {"action_type": "full_triage",
                           "required_fields": ["category","priority","suggested_action",
                                               "action_items","draft_reply"],
                           "optional_fields": []}},
    ]}

@app.post("/grader", response_model=GraderResult)
def grader(body: GraderRequest):
    if body.email_id not in EMAIL_BY_ID:
        raise HTTPException(404, f"email_id '{body.email_id}' not found.")
    if body.task_id not in TASK_EMAILS:
        raise HTTPException(404, f"task_id '{body.task_id}' not found.")
    raw_score, breakdown = grade(body.task_id, body.email_id, body.action)
    score = _clamp(raw_score)   # clamp grader output
    return GraderResult(task_id=body.task_id, episode_id="standalone",
                        score=score, breakdown=breakdown, passed=score >= 0.5)

@app.get("/baseline")
def baseline():
    from baseline import run_baseline_in_process
    raw_scores, details = run_baseline_in_process()
    scores = {k: _clamp(v) for k, v in raw_scores.items()}   # clamp all baseline scores
    mean   = _clamp(sum(scores.values()) / len(scores))
    return BaselineResponse(scores=scores, mean_score=mean, details=details)

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1)

if __name__ == "__main__":
    main()
