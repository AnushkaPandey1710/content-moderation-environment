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
load_dotenv(".env")

BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("OPENAI_API_KEY:", OPENAI_API_KEY)

if not BASE_URL or not API_KEY:
    raise ValueError("Missing API_BASE_URL or API_KEY")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_NAME = "content-moderation"

MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.6

USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

# --------------------------------------------------
# INIT LLM CLIENT
# --------------------------------------------------
client = None

if USE_LLM:
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY for LLM usage")

    client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# CLI CONFIG
# --------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default=None)
    
    args, unknown = parser.parse_known_args()  # 
    return args

args = get_args()

if args.base_url:
    BASE_URL = args.base_url

# --------------------------------------------------
# HEADERS
# --------------------------------------------------
HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
def log_start(task: str):
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)


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

# --------------------------------------------------
# API CALLS
# --------------------------------------------------
def reset_env():
    res = requests.post(f"{BASE_URL}/reset", headers=HEADERS, timeout=15)

    if res.status_code != 200:
        raise Exception(f"Reset failed: {res.text}")

    data = res.json()
    return data["session_id"], data.get("state", {})


def step_env(session_id, action, confidence):
    res = requests.post(
        f"{BASE_URL}/step",
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

# --------------------------------------------------
# RULE POLICY (fallback)
# --------------------------------------------------
def rule_policy(obs):
    tox = obs.get("toxicity", 0)
    amb = obs.get("ambiguity", 0)
    rep = obs.get("reports", 0)
    vir = obs.get("virality", 0)
    user_rep = obs.get("user_reputation", 0)

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

# --------------------------------------------------
# LLM POLICY
# --------------------------------------------------
def llm_policy(obs):
    if client is None:
        return rule_policy(obs)

    prompt = f"""
You are a content moderation agent.

toxicity={obs.get('toxicity', 0)}
ambiguity={obs.get('ambiguity', 0)}
reports={obs.get('reports', 0)}
virality={obs.get('virality', 0)}
user_reputation={obs.get('user_reputation', 0)}

Return ONLY a number:
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

        content = res.choices[0].message.content.strip()

        try:
            action = int(content)
        except:
            action = 0

        if action not in [0, 1, 3]:
            action = 0

        return action, 0.9

    except Exception:
        return rule_policy(obs)
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

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    for task_name in TASKS.keys():
        run_task(task_name)