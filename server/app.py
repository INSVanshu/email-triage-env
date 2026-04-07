"""
FastAPI server for the Email Triage OpenEnv environment.
Includes main() function required by openenv validate.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from models import (
    TriageAction, EmailObservation, TriageState, GraderResult,
    ActionType, EmailCategory, EmailPriority,
)
from data import TASK_EMAILS, EMAIL_BY_ID
from graders import grade
from server.environment import EmailTriageEnvironment

app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world email triage environment for AI agent training.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_env = EmailTriageEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "task_classify"

class StepRequest(BaseModel):
    action: TriageAction

class GraderRequest(BaseModel):
    task_id: str
    email_id: str
    action: TriageAction

class BaselineResponse(BaseModel):
    scores: dict
    mean_score: float
    details: dict


@app.get("/")
def root():
    return {
        "environment": "Email Triage OpenEnv",
        "version": "1.0.0",
        "tasks": list(TASK_EMAILS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/docs"],
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.post("/reset", response_model=EmailObservation)
async def reset(body: Optional[ResetRequest] = None):
    """Start a new episode. Body is fully optional."""
    if body is None:
        body = ResetRequest()
    try:
        obs = _env.reset(task_id=body.task_id)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(body: StepRequest):
    try:
        obs, reward, done, info = _env.step(body.action)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=TriageState)
def state():
    try:
        return _env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "task_classify",
                "name": "Email Classification",
                "difficulty": "easy",
                "description": "Classify each email into one of 6 categories.",
                "num_emails": len(TASK_EMAILS["task_classify"]),
                "action_schema": {
                    "action_type": "classify",
                    "required_fields": ["category"],
                    "optional_fields": [],
                    "category_options": [e.value for e in EmailCategory],
                },
            },
            {
                "task_id": "task_triage",
                "name": "Email Triage",
                "difficulty": "medium",
                "description": "Classify + set priority + suggest action.",
                "num_emails": len(TASK_EMAILS["task_triage"]),
                "action_schema": {
                    "action_type": "triage",
                    "required_fields": ["category", "priority", "suggested_action"],
                    "optional_fields": [],
                    "category_options": [e.value for e in EmailCategory],
                    "priority_options": [p.value for p in EmailPriority],
                },
            },
            {
                "task_id": "task_full_triage",
                "name": "Full Email Triage",
                "difficulty": "hard",
                "description": "Full 5-field triage: category + priority + action + items + reply.",
                "num_emails": len(TASK_EMAILS["task_full_triage"]),
                "action_schema": {
                    "action_type": "full_triage",
                    "required_fields": ["category", "priority", "suggested_action", "action_items", "draft_reply"],
                    "optional_fields": [],
                },
            },
        ]
    }

@app.post("/grader", response_model=GraderResult)
def grader(body: GraderRequest):
    if body.email_id not in EMAIL_BY_ID:
        raise HTTPException(status_code=404, detail=f"email_id '{body.email_id}' not found.")
    if body.task_id not in TASK_EMAILS:
        raise HTTPException(status_code=404, detail=f"task_id '{body.task_id}' not found.")
    score, breakdown = grade(body.task_id, body.email_id, body.action)
    return GraderResult(task_id=body.task_id, episode_id="standalone", score=score, breakdown=breakdown, passed=score >= 0.5)

@app.get("/baseline")
def baseline():
    from baseline import run_baseline_in_process
    scores, details = run_baseline_in_process()
    mean = round(sum(scores.values()) / len(scores), 4)
    return BaselineResponse(scores=scores, mean_score=mean, details=details)


def main():
    """Entry point required by openenv validate — called by [project.scripts] server = server.app:main"""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
