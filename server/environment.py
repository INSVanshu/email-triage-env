"""
EmailTriageEnvironment — clamps all rewards to strictly (0.001, 0.999).
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

MAX_STEPS = 20

def _clamp(score: float) -> float:
    """Scores must be strictly between 0 and 1 — never exactly 0.0 or 1.0."""
    return round(max(0.001, min(0.999, float(score))), 4)


class EmailTriageEnvironment:

    def __init__(self) -> None:
        self._state: TriageState | None = None
        self._email_queue: list[dict] = []
        self._current_email: dict | None = None
        self._scores: list[float] = []

    def reset(self, task_id: str = "task_classify") -> EmailObservation:
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
        return self._next_observation(reward=0.001, done=False, success=True,
                                      feedback="Episode started. Triage the email below.")

    def step(self, action: TriageAction) -> Tuple[EmailObservation, float, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.step_count >= MAX_STEPS:
            obs = self._next_observation(0.001, True, False, "Max steps reached.")
            return obs, 0.001, True, {"error": "max_steps_exceeded"}

        expected_type = ACTION_TYPE_MAP.get(self._state.task_id)
        if action.action_type != expected_type:
            feedback = f"Action type mismatch. Expected '{expected_type.value}'. Scoring anyway."
            action = action.model_copy(update={"action_type": expected_type})
        else:
            feedback = None

        current_email = self._current_email
        if current_email is None:
            obs = self._next_observation(0.001, True, False, "No email loaded.")
            return obs, 0.001, True, {"error": "no_email"}

        raw_score, breakdown = grade(self._state.task_id, current_email["email_id"], action)

        # ── CLAMP HERE — the single source of truth ───────────────
        score = _clamp(raw_score)

        # Loop penalty
        past = [a for a in self._state.actions_log
                if a.get("email_id") == current_email["email_id"]]
        if len(past) >= 2:
            score = _clamp(score - 0.15 * (len(past) - 1))
            breakdown["loop_penalty"] = True

        self._scores.append(score)
        self._state.step_count += 1
        self._state.emails_completed += 1
        self._state.cumulative_score = _clamp(sum(self._scores) / len(self._scores))
        self._state.actions_log.append({
            "step": self._state.step_count,
            "email_id": current_email["email_id"],
            "action_type": action.action_type.value,
            "score": score,
            "breakdown": breakdown,
        })

        if feedback is None:
            if score >= 0.85:
                feedback = f"Excellent! Score: {score:.3f}"
            elif score >= 0.60:
                feedback = f"Good. Score: {score:.3f}"
            elif score >= 0.35:
                feedback = f"Partial. Score: {score:.3f}"
            else:
                feedback = f"Low score: {score:.3f}"

        done = len(self._email_queue) == 0
        obs = self._next_observation(reward=score, done=done, success=score > 0.001,
                                     feedback=feedback)
        info = {"breakdown": breakdown, "episode_id": self._state.episode_id}
        return obs, score, done, info

    @property
    def state(self) -> TriageState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def _next_observation(self, reward, done, success, feedback) -> EmailObservation:
        if self._email_queue and not done:
            self._current_email = self._email_queue.pop(0)
        elif done:
            self._current_email = None

        email = self._current_email
        cum = _clamp(self._state.cumulative_score) if self._state else 0.001

        if email is None:
            return EmailObservation(
                email_id="", subject="[Episode Complete]", sender="",
                body="All emails triaged.", thread_length=0,
                reward=_clamp(reward), done=True, success=success,
                emails_remaining=0, cumulative_score=cum, feedback=feedback,
            )

        return EmailObservation(
            email_id=email["email_id"], subject=email["subject"],
            sender=email["sender"], body=email["body"],
            thread_length=email["thread_length"],
            reward=_clamp(reward), done=done, success=success,
            emails_remaining=len(self._email_queue),
            cumulative_score=cum, feedback=feedback,
        )

    def episode_score(self) -> float:
        if not self._scores:
            return 0.001
        return _clamp(sum(self._scores) / len(self._scores))
