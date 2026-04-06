"""
Typed Pydantic models for the Email Triage OpenEnv environment.
Defines Action, Observation, and State following the OpenEnv spec.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class EmailCategory(str, Enum):
    SPAM        = "spam"
    WORK        = "work"
    PERSONAL    = "personal"
    NEWSLETTER  = "newsletter"
    FINANCE     = "finance"
    SUPPORT     = "support"
    UNKNOWN     = "unknown"


class EmailPriority(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class ActionType(str, Enum):
    CLASSIFY   = "classify"          # Task 1 – assign a category
    TRIAGE     = "triage"            # Task 2 – classify + set priority + suggest action
    FULL_TRIAGE = "full_triage"      # Task 3 – classify + prioritise + extract action items + draft reply


# ─────────────────────────────────────────────
# Action  (what the agent sends)
# ─────────────────────────────────────────────

class TriageAction(BaseModel):
    """The action an agent submits for the current email."""

    action_type: ActionType = Field(
        ...,
        description="Which triage operation to perform.",
    )
    # ── Task 1 fields ──
    category: EmailCategory = Field(
        default=EmailCategory.UNKNOWN,
        description="Email category label.",
    )
    # ── Task 2 fields ──
    priority: Optional[EmailPriority] = Field(
        default=None,
        description="Urgency level (required for triage / full_triage).",
    )
    suggested_action: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Brief one-line recommended action, e.g. 'Reply within 24 h'.",
    )
    # ── Task 3 fields ──
    action_items: Optional[List[str]] = Field(
        default=None,
        description="Bullet-point action items extracted from the email.",
    )
    draft_reply: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Short draft reply to send back.",
    )


# ─────────────────────────────────────────────
# Observation  (what the agent receives)
# ─────────────────────────────────────────────

class EmailObservation(BaseModel):
    """Observation returned after reset() or step()."""

    # Current email being triaged
    email_id: str                          = Field(..., description="Unique ID of this email.")
    subject: str                           = Field(..., description="Email subject line.")
    sender: str                            = Field(..., description="Sender address.")
    body: str                              = Field(..., description="Email body (plain text).")
    thread_length: int                     = Field(default=1, description="Number of messages in thread.")

    # Step feedback
    reward: float                          = Field(default=0.0, description="Step-level reward [0,1].")
    done: bool                             = Field(default=False, description="True when episode is complete.")
    success: bool                          = Field(default=False, description="Whether the last action was valid.")

    # Cumulative metrics visible to agent
    emails_remaining: int                  = Field(default=0, description="Emails left in this episode.")
    cumulative_score: float                = Field(default=0.0, description="Running average score so far.")

    # Optional feedback text
    feedback: Optional[str]                = Field(default=None, description="Human-readable feedback on last action.")


# ─────────────────────────────────────────────
# State  (internal server state, returned by state())
# ─────────────────────────────────────────────

class TriageState(BaseModel):
    """Full episode state — returned by the /state endpoint."""
    episode_id: str
    task_id: str
    step_count: int
    total_emails: int
    emails_completed: int
    cumulative_score: float
    actions_log: List[dict] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Reward model  (returned by /grader)
# ─────────────────────────────────────────────

class GraderResult(BaseModel):
    task_id: str
    episode_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Final episode score.")
    breakdown: dict = Field(default_factory=dict, description="Per-email score breakdown.")
    passed: bool
