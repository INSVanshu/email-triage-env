"""
FastAPI server for the Email Triage OpenEnv environment.

Endpoints
─────────
POST /reset          → start new episode
POST /step           → submit action for current email
GET  /state          → current episode state
GET  /tasks          → list tasks + action schemas
POST /grader         → score a single action without advancing state
GET  /baseline       → run baseline agent on all 3 tasks (slow)
GET  /health         → liveness check
GET  /               → welcome + links
"""
from __future__ import annotations
import os
import sys
import time
import json

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import (
    TriageAction, EmailObservation, TriageState, GraderResult,
    ActionType, EmailCategory, EmailPriority,
)
from data import TASK_EMAILS, EMAIL_BY_ID
from graders import grade
from server.environment import EmailTriageEnvironment

# ─────────────────────────────────────────────
# App & global env instance
# ─────────────────────────────────────────────

app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world email triage environment for AI agent training.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# One shared environment instance per server process.
# For concurrent use, instantiate per-session (see notes in README).
_env = EmailTriageEnvironment()
_last_episode_scores: dict[str, float] = {}


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_classify"


class StepRequest(BaseModel):
    action: TriageAction


class GraderRequest(BaseModel):
    task_id: str
    email_id: str
    action: TriageAction


class BaselineResponse(BaseModel):
    scores: dict[str, float]
    mean_score: float
    details: dict


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

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
def reset(body: ResetRequest):
    """Start a new episode for the given task."""
    try:
        obs = _env.reset(task_id=body.task_id)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(body: StepRequest):
    """Submit an action for the current email."""
    try:
        obs, reward, done, info = _env.step(body.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=TriageState)
def state():
    """Return the current episode state."""
    try:
        return _env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks():
    """
    Return all available tasks with their action schema.
    The action schema describes exactly which fields an agent must populate.
    """
    return {
        "tasks": [
            {
                "task_id": "task_classify",
                "name": "Email Classification",
                "difficulty": "easy",
                "description": (
                    "Classify each email into one of the predefined categories. "
                    "The agent must correctly identify spam, work, personal, newsletter, "
                    "finance, and support emails."
                ),
                "num_emails": len(TASK_EMAILS["task_classify"]),
                "action_schema": {
                    "action_type": "classify",
                    "required_fields": ["category"],
                    "optional_fields": [],
                    "category_options": [e.value for e in EmailCategory],
                },
                "scoring": "1.0 = correct category, 0.5 = adjacent class, 0.0 = spam/non-spam confusion",
            },
            {
                "task_id": "task_triage",
                "name": "Email Triage",
                "difficulty": "medium",
                "description": (
                    "Classify the email, assign urgency priority, and suggest a one-line action. "
                    "Partial credit is awarded for each correct sub-field."
                ),
                "num_emails": len(TASK_EMAILS["task_triage"]),
                "action_schema": {
                    "action_type": "triage",
                    "required_fields": ["category", "priority", "suggested_action"],
                    "optional_fields": [],
                    "category_options": [e.value for e in EmailCategory],
                    "priority_options": [p.value for p in EmailPriority],
                },
                "scoring": "Weighted: category 30% + priority 40% + suggested_action 30%",
            },
            {
                "task_id": "task_full_triage",
                "name": "Full Email Triage",
                "difficulty": "hard",
                "description": (
                    "Full triage pipeline: classify, set priority, suggest action, "
                    "extract structured action items, and draft a short reply. "
                    "Agents are evaluated on all five dimensions."
                ),
                "num_emails": len(TASK_EMAILS["task_full_triage"]),
                "action_schema": {
                    "action_type": "full_triage",
                    "required_fields": [
                        "category", "priority", "suggested_action",
                        "action_items", "draft_reply",
                    ],
                    "optional_fields": [],
                    "field_descriptions": {
                        "action_items": "List of concrete next steps extracted from the email",
                        "draft_reply": "Short reply draft (<500 chars) to the sender",
                    },
                },
                "scoring": "Weighted: triage 40% + action_items 35% + draft_reply 25%",
            },
        ]
    }


@app.post("/grader", response_model=GraderResult)
def grader(body: GraderRequest):
    """
    Score a single (email_id, action) pair without advancing episode state.
    Useful for unit-testing your agent's output.
    """
    if body.email_id not in EMAIL_BY_ID:
        raise HTTPException(status_code=404, detail=f"email_id '{body.email_id}' not found.")
    if body.task_id not in TASK_EMAILS:
        raise HTTPException(status_code=404, detail=f"task_id '{body.task_id}' not found.")

    score, breakdown = grade(body.task_id, body.email_id, body.action)

    return GraderResult(
        task_id=body.task_id,
        episode_id="standalone",
        score=score,
        breakdown=breakdown,
        passed=score >= 0.5,
    )


@app.get("/baseline")
def baseline():
    """
    Run the rule-based baseline agent on all 3 tasks and return scores.
    This endpoint mirrors the standalone baseline.py script.
    WARNING: This resets the shared environment state.
    """
    from baseline import run_baseline_in_process
    scores, details = run_baseline_in_process()
    mean = round(sum(scores.values()) / len(scores), 4)
    return BaselineResponse(scores=scores, mean_score=mean, details=details)
