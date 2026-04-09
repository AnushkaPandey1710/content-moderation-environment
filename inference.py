import os
import argparse
import requests
from typing import List, Optional
from openai import OpenAI

from models import ModerationDecision
from tasks import TASKS
from dotenv import load_dotenv

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv(".env", override=False)

# ✅ REQUIRED (STRICT)
LLM_BASE_URL = os.environ["API_BASE_URL"]
LLM_API_KEY = os.environ["API_KEY"]

USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"


ENV_BASE_URL = os.getenv("ENV_BASE_URL", LLM_BASE_URL)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_NAME = "content-moderation"

MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.6

# --------------------------------------------------
# INIT LLM CLIENT (LiteLLM proxy)
# --------------------------------------------------
client = None

if USE_LLM:
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    except Exception as e:
        print(f"[DEBUG] Failed to init LLM client: {e}", flush=True)
        USE_LLM = False

# --------------------------------------------------
# HEADERS (ENV SERVER ONLY)
# --------------------------------------------------
HEADERS = {
    "x-api-key": LLM_API_KEY,
    "Content-Type": "application/json"
}

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
def log_start(task: str):
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# --------------------------------------------------
# API CALLS (ENV SERVER)
# --------------------------------------------------
def reset_env():
    try:
        res = requests.post(f"{ENV_BASE_URL}/reset", headers=HEADERS, timeout=15)

        if res.status_code != 200:
            raise Exception(f"Reset failed: {res.text}")

        data = res.json()
        return data.get("session_id"), data.get("state", {})

    except Exception as e:
        print(f"[ERROR] reset_env: {e}", flush=True)
        return None, {}

def step_env(session_id, action, confidence):
    try:
        res = requests.post(
            f"{ENV_BASE_URL}/step",
            json={
                "session_id": session_id,
                "action": action,
                "confidence": confidence
            },
            headers=HEADERS,
            timeout=15
        )

        if res.status_code != 200:
            raise Exception(f"Step failed: {res.text}")

        data = res.json()

        return (
            data.get("state", {}),
            data.get("reward", 0.0),
            data.get("done", False),
            data.get("info", {})
        )

    except Exception:
        return {}, 0.0, True, {}

# --------------------------------------------------
# RULE POLICY
# --------------------------------------------------
def rule_policy(obs):
    try:
        tox = float(obs.get("toxicity", 0) or 0)
        amb = float(obs.get("ambiguity", 0) or 0)
        rep = float(obs.get("reports", 0) or 0)
        vir = float(obs.get("virality", 0) or 0)
        user_rep = float(obs.get("user_reputation", 0) or 0)

        if tox < 0.2 and amb < 0.3:
            return 0, 0.95
        if tox > 0.85:
            return 1, 0.95
        if amb > 0.7:
            return 3, min(1.0, amb)
        if tox > 0.6 and (rep > 50 or vir > 0.6):
            return 1, 0.85
        if tox > 0.5 and user_rep < 0.3:
            return 1, 0.8
        if 0.3 < tox <= 0.6:
            return 1, 0.6

        return 0, 0.7

    except Exception:
        return 0, 0.5

# --------------------------------------------------
# LLM POLICY (FIXED ✅)
# --------------------------------------------------
def llm_policy(obs, task_name):
    if client is None:
        return rule_policy(obs)

    prompt = f"""
You are a content moderation system.

TASK: {task_name}

Return ONLY:
0 = ALLOW
1 = FLAG
3 = ESCALATE

toxicity={obs.get('toxicity', 0)}
ambiguity={obs.get('ambiguity', 0)}
reports={obs.get('reports', 0)}
virality={obs.get('virality', 0)}
user_reputation={obs.get('user_reputation', 0)}
"""

    try:
        print("[LLM HIT]", flush=True)

        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a content moderation system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10
        )

        content = (res.choices[0].message.content or "").strip()

        action = int(content) if content.isdigit() else 1

        if action not in [0, 1, 3]:
            action = 1

        return action, 0.8

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}", flush=True)
        return 1, 0.5

# --------------------------------------------------
# RUN TASK
# --------------------------------------------------
def run_task(task_name: str):
    rewards: List[float] = []
    total_reward = 0.0
    step_count = 0
    final_info = {}

    log_start(task_name)

    session_id, observation = reset_env()

    if not session_id:
        log_end(False, 0, 0.0, [])
        return

    done = False

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        try:
            if USE_LLM:
                action, confidence = llm_policy(observation, task_name)
            else:
                action, confidence = rule_policy(observation)

            observation, reward, done, info = step_env(
                session_id, action, confidence
            )

            final_info = info or {}

            rewards.append(float(reward or 0.0))
            total_reward += float(reward or 0.0)
            step_count = step

            try:
                action_name = ModerationDecision(action).name
            except Exception:
                action_name = str(action)

            log_step(step, action_name, float(reward or 0.0), done, None)

        except Exception as e:
            log_step(step, "error", 0.0, True, str(e))
            break

    score = float(final_info.get("final_score", total_reward / max(step_count, 1)))
    score = max(0.0, min(1.0, score))
    success = score >= SUCCESS_THRESHOLD

    log_end(success, step_count, score, rewards)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    for task_name in TASKS.keys():
        run_task(task_name)