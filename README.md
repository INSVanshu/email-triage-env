---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - email
  - triage
app_port: 7860
---

# 📧 Email Triage OpenEnv

A real-world email triage environment for training AI agents — built to the OpenEnv specification.

An AI agent receives emails one at a time and must classify, prioritise, extract action items, and draft replies.

## Quick start
```bash
curl https://Vansh051201-email-triage-env.hf.space/health
```

## Endpoints

- `POST /reset` — start new episode
- `POST /step` — submit action  
- `GET /state` — current episode state
- `GET /tasks` — list all tasks
- `POST /grader` — score an action
- `GET /baseline` — run baseline agent
- `GET /docs` — interactive API docs
