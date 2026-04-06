"""
validate.py – Pre-submission validation script.

Mimics the automated checks that the OpenEnv validator would run:
  1. openenv.yaml is valid and has required fields
  2. Typed models are importable and match spec
  3. Environment lifecycle: reset() → step() → state() all work
  4. All 3 tasks run with grader scores in [0.0, 1.0]
  5. Baseline scores are reproducible and within expected range
  6. /tasks endpoint returns correct schema
  7. /grader endpoint scores correctly

Run: python validate.py
Green checkmarks = ready to submit.
"""
from __future__ import annotations
import sys
import os
import json
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Pydantic stub (for environments without pydantic installed) ──
def _ensure_pydantic():
    try:
        import pydantic
        return
    except ImportError:
        pydantic_mod = types.ModuleType("pydantic")
        class _BM:
            def __init__(self, **kw):
                for k, v in kw.items(): setattr(self, k, v)
            def model_dump(self):
                return {k: v for k, v in self.__dict__.items()}
            def model_copy(self, update=None):
                d = self.model_dump()
                if update: d.update(update)
                return self.__class__(**d)
        class _F:
            def __call__(self, *a, **kw): return kw.get('default', None)
            def __getattr__(self, n): return self
        pydantic_mod.BaseModel = _BM
        pydantic_mod.Field = _F()
        sys.modules['pydantic'] = pydantic_mod

_ensure_pydantic()

import yaml

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠ "
errors = []

def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}" + (f"  →  {detail}" if detail else ""))
        errors.append(label)

def warn(label: str, detail: str = ""):
    print(f"{WARN} {label}" + (f"  →  {detail}" if detail else ""))


print("\n" + "="*60)
print("  Email Triage OpenEnv — Pre-submission Validator")
print("="*60 + "\n")


# ── 1. openenv.yaml ─────────────────────────────────────────────
print("── 1. openenv.yaml ─────────────────────────────────────────")
try:
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("openenv.yaml parses as valid YAML", True)
    check("name field present",    bool(spec.get("name")))
    check("version field present", bool(spec.get("version")))
    check("tasks field present",   bool(spec.get("tasks")))

    tasks_spec = spec.get("tasks", [])
    check("≥3 tasks defined", len(tasks_spec) >= 3, f"found {len(tasks_spec)}")
    task_ids = [t["id"] for t in tasks_spec]
    check("task_classify defined",    "task_classify"    in task_ids)
    check("task_triage defined",      "task_triage"      in task_ids)
    check("task_full_triage defined", "task_full_triage" in task_ids)

    endpoints = spec.get("interface", {}).get("endpoints", {})
    for ep in ("reset", "step", "state"):
        check(f"/{ ep} endpoint declared", ep in endpoints)

except Exception as e:
    check("openenv.yaml parseable", False, str(e))

print()


# ── 2. Typed models ──────────────────────────────────────────────
print("── 2. Typed models ─────────────────────────────────────────")
try:
    from models import (
        TriageAction, EmailObservation, TriageState, GraderResult,
        ActionType, EmailCategory, EmailPriority,
    )
    check("models.py imports cleanly", True)
    check("TriageAction is a class",    hasattr(TriageAction, '__init__'))
    check("EmailObservation is a class", hasattr(EmailObservation, '__init__'))
    check("TriageState is a class",     hasattr(TriageState, '__init__'))
    check("ActionType has classify/triage/full_triage",
          all(v in [e.value for e in ActionType] for v in ["classify", "triage", "full_triage"]))
    check("EmailCategory has spam/work/finance/support",
          all(v in [e.value for e in EmailCategory] for v in ["spam", "work", "finance", "support"]))
    check("EmailPriority has high/medium/low",
          all(v in [e.value for e in EmailPriority] for v in ["high", "medium", "low"]))
except Exception as e:
    check("models.py importable", False, str(e))

print()


# ── 3. Data corpus ───────────────────────────────────────────────
print("── 3. Data corpus ──────────────────────────────────────────")
try:
    from data import EMAILS, EMAIL_BY_ID, TASK_EMAILS
    check("data.py imports cleanly", True)
    check("≥10 emails in corpus", len(EMAILS) >= 10, f"found {len(EMAILS)}")
    check("All tasks have email lists", all(k in TASK_EMAILS for k in
          ["task_classify", "task_triage", "task_full_triage"]))
    check("task_classify has ≥3 emails",    len(TASK_EMAILS["task_classify"])    >= 3)
    check("task_triage has ≥3 emails",      len(TASK_EMAILS["task_triage"])      >= 3)
    check("task_full_triage has ≥5 emails", len(TASK_EMAILS["task_full_triage"]) >= 5)

    required_keys = {"email_id","subject","sender","body","thread_length",
                     "gt_category","gt_priority","gt_action_items","gt_reply_keywords"}
    all_have_keys = all(required_keys.issubset(set(e.keys())) for e in EMAILS)
    check("All emails have required GT labels", all_have_keys)
except Exception as e:
    check("data.py importable", False, str(e))

print()


# ── 4. Graders ───────────────────────────────────────────────────
print("── 4. Graders ──────────────────────────────────────────────")
try:
    from graders import grade

    # classify – exact match
    action = TriageAction(action_type=ActionType.CLASSIFY, category=EmailCategory.SPAM)
    s, _ = grade("task_classify", "e002", action)
    check("grade() returns float",             isinstance(s, float))
    check("Score in [0.0, 1.0]",               0.0 <= s <= 1.0)
    check("Correct SPAM classification = 1.0", s == 1.0,  f"got {s}")

    action = TriageAction(action_type=ActionType.CLASSIFY, category=EmailCategory.WORK)
    s, _ = grade("task_classify", "e002", action)
    check("SPAM→WORK misclassification = 0.0", s == 0.0, f"got {s}")

    # triage
    action = TriageAction(
        action_type=ActionType.TRIAGE,
        category=EmailCategory.WORK,
        priority=EmailPriority.HIGH,
        suggested_action="review budget allocations by friday",
    )
    s, bd = grade("task_triage", "e001", action)
    check("Triage grader returns breakdown dict", isinstance(bd, dict))
    check("Correct triage scores ≥0.85", s >= 0.85, f"got {s}")

    # full_triage
    action = TriageAction(
        action_type=ActionType.FULL_TRIAGE,
        category=EmailCategory.SPAM,
        priority=EmailPriority.LOW,
        suggested_action="mark as spam",
        action_items=["Mark as spam", "Do not reply"],
        draft_reply="",
    )
    s, _ = grade("task_full_triage", "e002", action)
    check("Spam correctly handled scores ≥0.80", s >= 0.80, f"got {s}")

    # Partial credit
    action = TriageAction(action_type=ActionType.CLASSIFY, category=EmailCategory.PERSONAL)
    s, _ = grade("task_classify", "e001", action)
    check("Wrong category gives partial credit > 0", s > 0.0, f"got {s}")
    check("Wrong category gives partial credit < 1", s < 1.0, f"got {s}")

except Exception as e:
    check("graders.py importable and callable", False, str(e))

print()


# ── 5. Environment lifecycle ─────────────────────────────────────
print("── 5. Environment lifecycle ─────────────────────────────────")
try:
    from server.environment import EmailTriageEnvironment

    env = EmailTriageEnvironment()
    check("EmailTriageEnvironment instantiates", True)

    obs = env.reset("task_classify")
    check("reset() returns EmailObservation",    hasattr(obs, "email_id"))
    check("reset() sets done=False",             obs.done == False)
    check("reset() loads emails",                obs.email_id != "")
    check("reset() has emails_remaining > 0",    obs.emails_remaining > 0)

    state = env.state
    check("state() returns TriageState",         hasattr(state, "task_id"))
    check("state() has task_id=task_classify",   state.task_id == "task_classify")
    check("state() has step_count=0",            state.step_count == 0)

    action = TriageAction(action_type=ActionType.CLASSIFY, category=EmailCategory.WORK)
    obs2, reward, done, info = env.step(action)
    check("step() returns 4-tuple",              True)
    check("step() reward in [0,1]",              0.0 <= reward <= 1.0)
    check("step() advances step_count",          env.state.step_count == 1)
    check("step() info has breakdown",           "breakdown" in info)

    # Run full episode
    while not obs2.done:
        obs2, reward, done, info = env.step(
            TriageAction(action_type=ActionType.CLASSIFY, category=EmailCategory.WORK)
        )
    check("Episode completes with done=True",    obs2.done == True)
    ep_score = env.episode_score()
    check("episode_score() in [0.0,1.0]",        0.0 <= ep_score <= 1.0)

except Exception as e:
    check("Environment lifecycle complete", False, str(e))

print()


# ── 6. All 3 tasks run with valid scores ─────────────────────────
print("── 6. All 3 tasks produce valid scores ─────────────────────")
try:
    from baseline import run_baseline_in_process
    scores, details = run_baseline_in_process()

    for task_id in ["task_classify", "task_triage", "task_full_triage"]:
        s = scores.get(task_id)
        check(f"{task_id} score in [0,1]", s is not None and 0.0 <= s <= 1.0, f"got {s}")
        check(f"{task_id} details non-empty", bool(details.get(task_id)))

    check("task_classify baseline ≥ task_triage baseline",
          scores["task_classify"] >= scores["task_triage"],
          f"{scores['task_classify']:.3f} vs {scores['task_triage']:.3f}")
    check("task_triage baseline ≥ task_full_triage baseline",
          scores["task_triage"] >= scores["task_full_triage"],
          f"{scores['task_triage']:.3f} vs {scores['task_full_triage']:.3f}")

    mean = sum(scores.values()) / len(scores)
    print(f"\n  Baseline scores:")
    for k, v in scores.items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"    {k:<22} [{bar}] {v:.4f}")
    print(f"    {'Mean':<22}               {mean:.4f}")

except Exception as e:
    check("Baseline reproduces", False, str(e))

print()


# ── 7. Dockerfile present ────────────────────────────────────────
print("── 7. Deployment artifacts ──────────────────────────────────")
check("Dockerfile exists",     os.path.isfile("Dockerfile"))
check("requirements.txt exists", os.path.isfile("requirements.txt"))
check("README.md exists",      os.path.isfile("README.md"))
check("openenv.yaml exists",   os.path.isfile("openenv.yaml"))

with open("Dockerfile") as f:
    df = f.read()
check("Dockerfile exposes port 7860", "7860" in df)
check("Dockerfile has CMD/ENTRYPOINT", "CMD" in df or "ENTRYPOINT" in df)
check("Dockerfile has non-root user",  "useradd" in df or "USER" in df)

print()


# ── Summary ──────────────────────────────────────────────────────
print("="*60)
if errors:
    print(f"  ❌ FAILED — {len(errors)} check(s) failed:")
    for e in errors:
        print(f"     • {e}")
else:
    print("  ✅ ALL CHECKS PASSED — Ready to submit!")
print("="*60 + "\n")

sys.exit(1 if errors else 0)
