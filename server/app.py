from fastapi import FastAPI, Request
from uuid import uuid4

from server.content_moderation_environment import ContentModerationEnvironment
from models import (
    ContentModerationObservation,
    ContentModerationAction
)

app = FastAPI()

# Store sessions
sessions = {}


# --------------------------------------------------
# RESET
# --------------------------------------------------
@app.post("/reset")
async def reset():
    session_id = str(uuid4())

    env = ContentModerationEnvironment()
    obs = env.reset()

    sessions[session_id] = env

    return {
        "session_id": session_id,
        "state": obs
    }


# --------------------------------------------------
# STEP
# --------------------------------------------------
@app.post("/step")
async def step(request: Request):
    data = await request.json()

    session_id = data.get("session_id")
    action = data.get("action")
    confidence = data.get("confidence", 0.5)

    if session_id not in sessions:
        return {"error": "Invalid session_id"}

    env = sessions[session_id]

    # Proper OpenEnv Action model
    action_obj = ContentModerationAction(
        action=action,
        confidence=confidence
    )

    result = env.step(action_obj)

    return {
        "session_id": session_id,
        "state": result,
        "reward": result.get("reward", 0.0),
        "done": result.get("done", False),
        "info": result.get("info", {})
    }


# --------------------------------------------------
# STATE (NEW ✅)
# --------------------------------------------------
@app.post("/state")
async def get_state(request: Request):
    data = await request.json()
    session_id = data.get("session_id")

    if session_id not in sessions:
        return {"error": "Invalid session_id"}

    env = sessions[session_id]

    # If no data yet (edge case)
    if not env.data:
        return {"session_id": session_id, "state": {}}

    # Safe index
    idx = min(env.current_idx, len(env.data) - 1)
    sample = env.data[idx]

    # 🔥 DIRECT mapping (NO builder)
    state = {
        "current_message": sample["text"],
        "context": sample["context"],
        "message_length": len(sample["text"]),
        "toxicity": sample["toxicity"],
        "virality": sample["virality"],
        "reports": sample["reports"],
        "user_reputation": sample["user_reputation"],
        "ambiguity": sample["ambiguity"],
        "severity": sample["severity"],
        "step": env._state.step_count,
        "total_steps": env.max_steps,
        "reward": 0.0,
        "done": False,
        "info": {}
    }

    return {
        "session_id": session_id,
        "state": state
    }
# --------------------------------------------------
# SCHEMA (NEW ✅)
# --------------------------------------------------
@app.get("/schema")
def schema():
    return {
        "observation_space": ContentModerationObservation.model_json_schema(),
        "action_space": ContentModerationAction.model_json_schema(),
        "reward_range": [0.0, 1.0]
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "Content Moderation Env Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# server/app.py

import uvicorn

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()