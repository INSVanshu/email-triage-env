"""
EmailTriageEnvironment – core logic for the OpenEnv Email Triage server.

Implements the three OpenEnv lifecycle methods:
  reset()  → start a fresh episode, returns first EmailObservation
  step()   → accept a TriageAction, return (obs, reward, done, info)
  state()  → return the current TriageState

Episode flow
────────────
1. reset(task_id) selects a task and loads its email queue.
2. Each step() presents one email, grades the agent's action, and advances.
3. When the queue is exhausted, done=True is returned.
4. The episode score is the mean per-email score over the episode.
"""
from __future__ import annotations
import copy
from uuid import uuid4
from typing import Tuple

from models import (
    TriageAction, EmailObservation, TriageState,
    EmailCategory, EmailPriority, ActionType
)
from data import EMAILS, EMAIL_BY_ID, TASK_EMAILS
from graders import grade, ACTION_TYPE_MAP


# Maximum steps per episode (safety cap)
MAX_STEPS = 20


class EmailTriageEnvironment:
    """Stateful email triage environment."""

    def __init__(self) -> None:
        self._state: TriageState | None = None
        self._email_queue: list[dict] = []
        self._current_email: dict | None = None
        self._scores: list[float] = []

    # ─────────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────────

    def reset(self, task_id: str = "task_classify") -> EmailObservation:
        """Start a new episode for the given task."""
        if task_id not in TASK_EMAILS:
            task_id = "task_classify"

        email_ids = TASK_EMAILS[task_id]
        self._email_queue = [copy.deepcopy(EMAIL_BY_ID[eid]) for eid in email_ids]
        self._scores = []

        self._state = TriageState(
            episode_id=str(uuid4()),
            task_id=task_id,
            step_count=0,
            total_emails=len(self._email_queue),
            emails_completed=0,
            cumulative_score=0.0,
            actions_log=[],
        )

        return self._next_observation(reward=0.0, done=False, success=True,
                                      feedback="Episode started. Triage the email below.")

    # ─────────────────────────────────────────────
    # step
    # ─────────────────────────────────────────────

    def step(self, action: TriageAction) -> Tuple[EmailObservation, float, bool, dict]:
        """
        Process one TriageAction.

        Returns
        -------
        observation : EmailObservation
        reward      : float  [0, 1]
        done        : bool
        info        : dict
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.step_count >= MAX_STEPS:
            obs = self._next_observation(0.0, True, False, "Max steps reached.")
            return obs, 0.0, True, {"error": "max_steps_exceeded"}

        # Validate action_type matches task
        expected_type = ACTION_TYPE_MAP.get(self._state.task_id)
        if action.action_type != expected_type:
            # Mild penalty: wrong action type but still score what we can
            feedback = (
                f"⚠ Action type mismatch. Expected '{expected_type.value}' "
                f"for task '{self._state.task_id}'. Scoring anyway with penalty."
            )
            action = action.model_copy(update={"action_type": expected_type})
        else:
            feedback = None

        # Grade the action
        current_email = self._current_email
        if current_email is None:
            obs = self._next_observation(0.0, True, False, "No email loaded.")
            return obs, 0.0, True, {"error": "no_email"}

        score, breakdown = grade(self._state.task_id, current_email["email_id"], action)

        # Apply loop-penalty: same (email_id, category) repeated > 2 times
        past_actions = [
            a for a in self._state.actions_log
            if a.get("email_id") == current_email["email_id"]
        ]
        if len(past_actions) >= 2:
            score = max(0.0, score - 0.15 * (len(past_actions) - 1))
            breakdown["loop_penalty"] = True

        # Update state
        self._scores.append(score)
        self._state.step_count += 1
        self._state.emails_completed += 1
        self._state.cumulative_score = sum(self._scores) / len(self._scores)
        self._state.actions_log.append({
            "step": self._state.step_count,
            "email_id": current_email["email_id"],
            "action_type": action.action_type.value,
            "score": score,
            "breakdown": breakdown,
        })

        # Feedback message
        if feedback is None:
            if score >= 0.85:
                feedback = f"✅ Excellent triage! Score: {score:.2f}"
            elif score >= 0.60:
                feedback = f"👍 Good. Score: {score:.2f}. {_hint(breakdown)}"
            elif score >= 0.35:
                feedback = f"⚠ Partial credit. Score: {score:.2f}. {_hint(breakdown)}"
            else:
                feedback = f"❌ Low score: {score:.2f}. {_hint(breakdown)}"

        # Advance to next email
        done = len(self._email_queue) == 0
        obs = self._next_observation(
            reward=score,
            done=done,
            success=score > 0,
            feedback=feedback,
        )
        info = {"breakdown": breakdown, "episode_id": self._state.episode_id}
        return obs, score, done, info

    # ─────────────────────────────────────────────
    # state
    # ─────────────────────────────────────────────

    @property
    def state(self) -> TriageState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _next_observation(
        self, reward: float, done: bool, success: bool, feedback: str
    ) -> EmailObservation:
        """Pop next email from queue and build an observation."""
        if self._email_queue and not done:
            self._current_email = self._email_queue.pop(0)
        elif done:
            self._current_email = None

        email = self._current_email
        if email is None:
            # Terminal observation — no more emails
            return EmailObservation(
                email_id="",
                subject="[Episode Complete]",
                sender="",
                body="All emails have been triaged.",
                thread_length=0,
                reward=reward,
                done=True,
                success=success,
                emails_remaining=0,
                cumulative_score=self._state.cumulative_score if self._state else 0.0,
                feedback=feedback,
            )

        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            sender=email["sender"],
            body=email["body"],
            thread_length=email["thread_length"],
            reward=reward,
            done=done,
            success=success,
            emails_remaining=len(self._email_queue),
            cumulative_score=self._state.cumulative_score if self._state else 0.0,
            feedback=feedback,
        )

    def episode_score(self) -> float:
        """Final aggregate score for the current episode."""
        if not self._scores:
            return 0.0
        return round(sum(self._scores) / len(self._scores), 4)


# ─────────────────────────────────────────────
# Hint helper
# ─────────────────────────────────────────────

def _hint(breakdown: dict) -> str:
    """Surface the weakest sub-score as a hint."""
    hints = []

    # Category check
    cat = breakdown.get("category") or breakdown
    if isinstance(cat, dict) and cat.get("pred_category") != cat.get("gt_category"):
        hints.append(f"Category should be '{cat.get('gt_category')}'")

    # Priority check
    pri = breakdown.get("priority")
    if pri and pri.get("score", 1) < 0.5:
        hints.append(f"Priority should be '{pri.get('gt')}'")

    # Action items
    ai = breakdown.get("action_items")
    if ai and ai.get("score", 1) < 0.4:
        hints.append("Try to extract more specific action items")

    return ". ".join(hints) if hints else "Review the email details carefully."
