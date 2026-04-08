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
# STATE
# --------------------------------------------------
from fastapi import FastAPI, HTTPException, Query


@app.get("/state")  # Use GET for fetching state
def get_state(session_id: str = Query(..., description="Session ID for the environment")):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Invalid session_id: {session_id}")

    env = sessions[session_id]

    # Handle edge case: no data yet
    if not env.data:
        return {"session_id": session_id, "state": {}}

    idx = min(env.current_idx, len(env.data) - 1)
    sample = env.data[idx]

    state = {
        "current_message": sample.get("text", ""),
        "context": sample.get("context", {}),
        "message_length": len(sample.get("text", "")),
        "toxicity": sample.get("toxicity", 0.0),
        "virality": sample.get("virality", 0.0),
        "reports": sample.get("reports", 0),
        "user_reputation": sample.get("user_reputation", 0),
        "ambiguity": sample.get("ambiguity", 0.0),
        "severity": sample.get("severity", 0.0),
        "step": getattr(env._state, "step_count", 0),
        "total_steps": getattr(env, "max_steps", 0),
        "reward": 0.0,
        "done": False,
        "info": {}
    }

    return {
        "session_id": session_id,
        "state": state
    }

# --------------------------------------------------
# SCHEMA
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
    uvicorn.run("app:app", host="0.0.0.0", port=7860)