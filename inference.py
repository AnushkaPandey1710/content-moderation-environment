import os
import argparse
import requests
from typing import List, Optional
from openai import OpenAI

from models import ModerationDecision
from tasks import TASKS

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_NAME = "content-moderation"

MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.6

#  REQUIRED (Phase 2 FIX)
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

USE_LLM = True  #  MUST be True for Phase 2

# -----------------------
# INIT LLM CLIENT
# -----------------------
client = None

if USE_LLM:
    if not API_BASE_URL or not API_KEY:
        raise ValueError("Missing API_BASE_URL or API_KEY (required for evaluation)")

    client = OpenAI(
        base_url=API_BASE_URL,   # 
        api_key=API_KEY          # 
    )

# -----------------------
# CLI CONFIG
# -----------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=None)
    return parser.parse_args()


args = get_args()

BASE_URL = args.base_url or os.getenv("BASE_URL") or \
    "https://anushkapandey1710-content-moderation-environment.hf.space"


# -----------------------
# LOGGING (STRICT FORMAT)
# -----------------------
def log_start(task: str):
    print(
        f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}",
        flush=True
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


# -----------------------
# API CALLS
# -----------------------
def reset_env():
    res = requests.post(f"{BASE_URL}/reset", timeout=15)
    data = res.json()

    state = data["state"]

    if "observation" in state:
        return data["session_id"], state["observation"]

    return data["session_id"], state


def step_env(session_id, action, confidence):
    res = requests.post(
        f"{BASE_URL}/step",
        json={
            "session_id": session_id,
            "action": action,
            "confidence": confidence
        },
        timeout=15
    )

    data = res.json()
    state = data["state"]

    if "observation" in state:
        return (
            state["observation"],
            state.get("reward", 0.0),
            state.get("done", False),
            state.get("info", {})
        )

    return (
        state,
        data.get("reward", 0.0),
        data.get("done", False),
        data.get("info", {})
    )


# -----------------------
# RULE POLICY (fallback)
# -----------------------
def rule_policy(obs):
    tox = obs["toxicity"]
    amb = obs["ambiguity"]
    rep = obs["reports"]
    vir = obs["virality"]
    user_rep = obs["user_reputation"]

    if tox < 0.2 and amb < 0.3:
        return 0, 0.95

    if tox > 0.85:
        return 1, 0.95

    if amb > 0.65:
        return 3, amb

    if tox > 0.6 and (rep > 50 or vir > 0.6):
        return 1, 0.8

    if tox > 0.5 and user_rep < 0.3:
        return 1, 0.75

    if 0.3 < tox < 0.6:
        return 1, 0.55

    return 0, 0.7


# -----------------------
# LLM POLICY (REQUIRED)
# -----------------------
def llm_policy(obs):
    prompt = f"""
You are a content moderation agent.

toxicity={obs['toxicity']}
ambiguity={obs['ambiguity']}
reports={obs['reports']}
virality={obs['virality']}
user_reputation={obs['user_reputation']}

Return ONLY:
0 = ALLOW
1 = FLAG
3 = ESCALATE
"""

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        action = int(res.choices[0].message.content.strip())

        if action not in [0, 1, 3]:
            action = 0

        return action, 0.9

    except Exception:
        # fallback
        return rule_policy(obs)


# -----------------------
# RUN SINGLE TASK
# -----------------------
def run_task(task_name: str):
    rewards: List[float] = []
    total_reward = 0.0
    step_count = 0
    final_info = {}

    log_start(task_name)

    session_id, observation = reset_env()
    done = False

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        try:
            if USE_LLM:
                action, confidence = llm_policy(observation)
            else:
                action, confidence = rule_policy(observation)

            observation, reward, done, info = step_env(
                session_id, action, confidence
            )

            final_info = info

            rewards.append(reward)
            total_reward += reward
            step_count = step

            log_step(
                step,
                ModerationDecision(action).name,
                reward,
                done,
                None
            )

        except Exception as e:
            log_step(step, "error", 0.0, True, str(e))
            break

    if "final_score" in final_info:
        score = final_info["final_score"]
    elif step_count > 0:
        score = total_reward / step_count
    else:
        score = 0.0

    score = max(0.0, min(1.0, score))
    success = score >= SUCCESS_THRESHOLD

    log_end(success, step_count, score, rewards)


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    for task_name in TASKS.keys():
        run_task(task_name)