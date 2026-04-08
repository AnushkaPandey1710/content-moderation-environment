---
title: Content Moderation Environment Server
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Content Moderation Environment

## 1. Overview

This project provides a lightweight, API-driven environment for simulating and evaluating content moderation workflows. It is designed to support experimentation with rule-based and AI-assisted moderation strategies in a controlled setting.

The environment exposes APIs that allow agents to interact step-by-step with moderation tasks, receive feedback, and improve decision-making.

---

## 2. Environment Description & Motivation

### Motivation

Content moderation is a critical component in modern platforms such as social media, forums, and enterprise tools. However, testing moderation systems in production is:

- Risky (affects real users)
- Expensive (requires large infrastructure)
- Hard to iterate (tight coupling with production systems)

This environment solves these challenges by providing:

- A safe sandbox for moderation workflows  
- Deterministic and reproducible scenarios  
- API-first design for easy agent integration  

### Environment Characteristics

- Session-based interaction (`reset → step → done`)
- Supports reinforcement learning and rule-based agents
- Designed for realistic moderation decision-making

---

## 3. Action Space

The agent can take one of the following actions:

| Action | Meaning |
|------|--------|
| 0 | ALLOW |
| 1 | FLAG |
| 2 | REMOVE |
| 3 | ESCALATE |

### Action Format

```json
{
  "action": 1,
  "confidence": 0.8
}


4. Observation Space

Each step returns structured observations:

current_message
toxicity
ambiguity
virality
reports
user_reputation
severity
step
reward
done

These signals simulate real-world moderation features.

5. Tasks & Difficulty

The environment includes three tasks with increasing difficulty:

Task	Difficulty	Description
basic_toxicity	Easy	Moderate based on toxicity score
contextual_moderation	Medium	Uses context (reports, reputation)
ambiguous_harm	Hard	Handles ambiguous harmful content
6. Reward Function

The reward function provides dense feedback:

Positive reward for correct actions
Partial reward for reasonable alternatives (e.g., escalate)
Penalty for incorrect moderation
Strong penalty for missing harmful content
Penalty increases with virality and severity

Rewards are normalized to [0, 1].

7. Setup Instructions
Prerequisites
Python 3.9+
pip
Install dependencies
pip install -r requirements.txt
8. Running the Server
python app.py

Server will run on:

http://0.0.0.0:7860
https://anushkapandey1710-content-moderation-environment.hf.space/
Running Inference
Single Task (STRICT - for evaluation)
python inference.py \
  --base_url https://anushkapandey1710-content-moderation-environment.hf.space/docs
  --task basic_toxicity

Run other tasks:

--task contextual_moderation
--task ambiguous_harm
Multi-task (Demo Mode)
python inference.py --base_url https://anushkapandey1710-content-moderation-environment.hf.space/docs

STDOUT FORMAT (STRICT)

The inference script strictly follows required output format:

[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
Example
[START] task=basic_toxicity env=content-moderation model=rule-based-agent
[STEP] step=1 action=FLAG reward=0.77 done=false error=null
[STEP] step=2 action=FLAG reward=0.82 done=false error=null
[STEP] step=3 action=FLAG reward=0.75 done=true error=null
[END] success=true steps=3 score=0.78 rewards=0.77,0.82,0.75
Rules
Only 3 line types allowed
No extra logs
Rewards formatted to 2 decimals
Boolean values must be lowercase
Score must be between 0 and 1
10. API Usage
Swagger UI

https://anushkapandey1710-content-moderation-environment.hf.space/docs

Endpoints
Reset Environment
POST /reset
Step Action
POST /step
Get State
GET /state
Get Schema
GET /schema
11. Baseline Scores
Task	Score
Easy	~0.70
Medium	~0.72
Hard	~0.74

Baseline uses a rule-based policy.

---

## 11. Pre-validation Script

To ensure the environment is fully compliant with OpenEnv specifications and ready for submission, a pre-validation script is included.

### What it checks

The script automatically validates:

- `/reset` endpoint (session creation)
- `/state` endpoint (state retrieval)
- `/step` endpoint (action execution)
- `/schema` endpoint (OpenEnv schema compliance)
- Observation structure (toxicity, ambiguity, etc.)
- Docker build success
- `openenv validate` compliance

It also includes retry logic to handle Hugging Face cold starts.

---

### How to Run

```bash
 bash validate-submission.sh https://anushkapandey1710-content-moderation-environment.hf.space



12. Future Improvements
PPO / RL-based training
LLM-based moderation
Multi-agent workflows
Real-world dataset integration
13. Conclusion

This environment provides a realistic and extensible framework for evaluating content moderation strategies. It enables safe experimentation and benchmarking of AI agents in a controlled setting.

14. Author

Anushka Pandey