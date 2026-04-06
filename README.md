# 📧 Email Triage OpenEnv

> A real-world email triage environment for training AI agents — built to the [OpenEnv](https://github.com/raun/openenv-course) specification.

An AI agent receives emails one at a time and must **classify**, **prioritise**, **extract action items**, and **draft replies** — exactly the inbox management workflow that knowledge workers perform daily.

---

## Why email triage?

| Criterion | Why it qualifies |
|-----------|-----------------|
| Real-world task | Every professional manages email. The skills transfer to customer support, ticket routing, and document review. |
| Clear grading | Category, priority, action items, and reply quality are all objectively measurable. |
| Partial-progress reward | An agent that gets the category right but misjudges priority still earns partial credit. |
| Scalable difficulty | Task 1 is a single-label classification; Task 3 requires five coordinated outputs. |

---

## Quick start

```bash
# 1 – Clone
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/email-triage-env
cd email-triage-env

# 2 – Install
pip install -r requirements.txt

# 3 – Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# 4 – Run baseline (rule-based, no API key needed)
python baseline.py

# 5 – Run LLM baseline (requires OpenAI API key)
OPENAI_API_KEY=sk-... python baseline.py --llm --model gpt-4o-mini
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

# With LLM baseline
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-triage-env
```

---

## API reference

All endpoints are available at `http://localhost:7860`.
Interactive docs: `http://localhost:7860/docs`

### `POST /reset`

Start a new episode.

```bash
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task_classify"}'
```

**Response** — `EmailObservation`:

```json
{
  "email_id": "e001",
  "subject": "Q3 Budget Review – Action Required by Friday",
  "sender": "cfo@acmecorp.com",
  "body": "...",
  "thread_length": 1,
  "reward": 0.0,
  "done": false,
  "success": true,
  "emails_remaining": 4,
  "cumulative_score": 0.0,
  "feedback": "Episode started. Triage the email below."
}
```

---

### `POST /step`

Submit a triage action for the current email.

```bash
# Task 1 – classify only
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{
       "action": {
         "action_type": "classify",
         "category": "work"
       }
     }'

# Task 3 – full triage
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{
       "action": {
         "action_type": "full_triage",
         "category": "work",
         "priority": "high",
         "suggested_action": "Review budget and confirm allocations by Friday COB",
         "action_items": [
           "Review Q3 budget spreadsheet",
           "Confirm department allocations by Friday COB",
           "Submit written justification for variances >5% to finance@acmecorp.com"
         ],
         "draft_reply": "Hi Sarah, thank you for the reminder. I will review the budget allocations and confirm by Friday COB. Best regards."
       }
     }'
```

**Response**:

```json
{
  "observation": { "...": "EmailObservation fields" },
  "reward": 0.97,
  "done": false,
  "info": {
    "breakdown": { "category": {...}, "priority": {...}, "action_items": {...} },
    "episode_id": "3f8a1c..."
  }
}
```

---

### `GET /state`

Returns the current episode state.

```bash
curl http://localhost:7860/state
```

```json
{
  "episode_id": "3f8a1c...",
  "task_id": "task_classify",
  "step_count": 2,
  "total_emails": 5,
  "emails_completed": 2,
  "cumulative_score": 0.75,
  "actions_log": [...]
}
```

---

### `GET /tasks`

Lists all tasks and their required action schemas.

```bash
curl http://localhost:7860/tasks
```

---

### `POST /grader`

Score a single action without advancing episode state. Useful for unit-testing.

```bash
curl -X POST http://localhost:7860/grader \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "task_triage",
       "email_id": "e001",
       "action": {
         "action_type": "triage",
         "category": "work",
         "priority": "high",
         "suggested_action": "Confirm allocations by Friday"
       }
     }'
```

---

### `GET /baseline`

Runs the rule-based baseline agent on all 3 tasks and returns scores.

```bash
curl http://localhost:7860/baseline
```

---

## Observation space

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | `string` | Unique identifier for this email |
| `subject` | `string` | Email subject line |
| `sender` | `string` | Sender address |
| `body` | `string` | Plain-text email body |
| `thread_length` | `int` | Number of messages in thread |
| `reward` | `float [0,1]` | Step-level reward from previous action |
| `done` | `bool` | True when all emails in episode are processed |
| `success` | `bool` | True when last action scored > 0 |
| `emails_remaining` | `int` | Emails left in this episode |
| `cumulative_score` | `float [0,1]` | Running mean score over episode |
| `feedback` | `string` | Human-readable hint about the last action |

---

## Action space

| Field | Type | Required for |
|-------|------|-------------|
| `action_type` | `classify \| triage \| full_triage` | All tasks |
| `category` | `spam \| work \| personal \| newsletter \| finance \| support \| unknown` | All tasks |
| `priority` | `high \| medium \| low` | `triage`, `full_triage` |
| `suggested_action` | `string` (≤200 chars) | `triage`, `full_triage` |
| `action_items` | `list[string]` | `full_triage` |
| `draft_reply` | `string` (≤500 chars) | `full_triage` |

---

## Tasks

### Task 1 — Email Classification `task_classify` *(easy)*

**Objective:** Assign each of 5 emails to one of 6 categories.

**Grading:**

| Result | Score |
|--------|-------|
| Correct category | 1.00 |
| Adjacent class (e.g. work vs support) | 0.50 |
| Spam / non-spam confusion | 0.00 |
| Any other wrong class | 0.20 |

**Baseline score: 1.00**

---

### Task 2 — Email Triage `task_triage` *(medium)*

**Objective:** Classify + prioritise + suggest action for 6 emails.

**Grading weights:**

| Component | Weight |
|-----------|--------|
| Category accuracy | 30% |
| Priority accuracy | 40% |
| Suggested action (keyword coverage) | 30% |

**Baseline score: 0.71**

---

### Task 3 — Full Email Triage `task_full_triage` *(hard)*

**Objective:** All 5 outputs across all 10 emails.

**Grading weights:**

| Component | Weight |
|-----------|--------|
| Triage sub-score (category + priority + action) | 40% |
| Action items coverage | 35% |
| Draft reply quality (keyword coverage) | 25% |

**Special rule:** Spam and newsletter emails score 1.0 on the reply component only if the draft reply is empty.

**Baseline score: 0.49**

---

## Reward function design

```
Step reward = weighted_grader_score(action, ground_truth)
```

**Dense signal** — reward is provided after *every* email, not just at episode end.

**Penalties:**
- Repeating the same action on the same email: `-0.15` per repeat
- No `suggested_action` on a HIGH-priority email: `-0.20` on that component
- Non-empty `draft_reply` for SPAM/newsletter: scores 0 on reply component

**Episode score** = arithmetic mean of all step rewards.

---

## Baseline scores (rule-based heuristic)

Run `python baseline.py` to reproduce:

```
══════════════════════════════════════════════════════════
  Email Triage OpenEnv — Baseline Evaluation
  Agent: Rule-based heuristic
══════════════════════════════════════════════════════════

▶ Task: task_classify
  Episode score: 1.0000

▶ Task: task_triage
  Episode score: 0.7125

▶ Task: task_full_triage
  Episode score: 0.4893

──────────────────────────────────────────────────────────
  [easy  ] task_classify        [████████████████████] 1.0000
  [medium] task_triage          [██████████████░░░░░░] 0.7125
  [hard  ] task_full_triage     [█████████░░░░░░░░░░░] 0.4893
──────────────────────────────────────────────────────────
  Mean score: 0.7339
══════════════════════════════════════════════════════════
```

LLM agents (GPT-4o, Claude) typically score **0.85–0.95** on Task 1,
**0.78–0.88** on Task 2, and **0.65–0.80** on Task 3.

---

## Project structure

```
email-triage-env/
├── models.py          # Typed Pydantic models (Action, Observation, State)
├── data.py            # Synthetic email corpus with ground-truth labels
├── graders.py         # Deterministic graders for all 3 tasks
├── baseline.py        # Rule-based + LLM baseline agents
├── openenv.yaml       # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── server/
│   ├── __init__.py
│   ├── app.py         # FastAPI routes (/reset /step /state /tasks /grader /baseline)
│   └── environment.py # Core env logic (reset / step / state)
└── README.md
```

---

## Extending the environment

**Add more emails:** Edit `data.py`. Each email needs:
`email_id`, `subject`, `sender`, `body`, `thread_length`,
`gt_category`, `gt_priority`, `gt_action_items`, `gt_reply_keywords`

**Add a new task:** Add an entry to `TASK_EMAILS` in `data.py` and
a new grader function in `graders.py`.

**Multi-session support:** Instantiate `EmailTriageEnvironment()` per
session and store in a dict keyed by session token.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For LLM baseline only | OpenAI API key |
| `PORT` | No (default 7860) | Server port |

---

## License

MIT
