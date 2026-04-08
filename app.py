from fastapi import FastAPI, Request, HTTPException, Header
from uuid import uuid4
import os
from dotenv import load_dotenv

from server.content_moderation_environment import ContentModerationEnvironment
from models import (
    ContentModerationObservation,
    ContentModerationAction
)

import uvicorn

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY is required")
# --------------------------------------------------
# APP INIT
# --------------------------------------------------
app = FastAPI(title="Content Moderation Environment API")

# Store sessions
sessions = {}

# --------------------------------------------------
# AUTH HELPER
# --------------------------------------------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --------------------------------------------------
# RESET
# --------------------------------------------------
@app.post("/reset")
async def reset(x_api_key: str = Header(None)):
    verify_api_key(x_api_key)

    try:
        session_id = str(uuid4())

        env = ContentModerationEnvironment()
        obs = env.reset()

        sessions[session_id] = env

        return {
            "session_id": session_id,
            "state": obs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# --------------------------------------------------
# STEP
# --------------------------------------------------
@app.post("/step")
async def step(request: Request, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)

    try:
        data = await request.json()

        session_id = data.get("session_id")
        action = data.get("action")
        confidence = data.get("confidence", 0.5)

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Invalid session_id")

        env = sessions[session_id]

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


# --------------------------------------------------
# STATE
# --------------------------------------------------
@app.post("/state")
async def get_state(request: Request, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)

    try:
        data = await request.json()
        session_id = data.get("session_id")

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Invalid session_id")

        env = sessions[session_id]

        if not env.data:
            return {
                "session_id": session_id,
                "state": {},
                "message": "No data available yet"
            }

        idx = min(env.current_idx, len(env.data) - 1)
        sample = env.data[idx]

        state = {
            "current_message": sample.get("text", ""),
            "context": sample.get("context", ""),
            "message_length": len(sample.get("text", "")),
            "toxicity": sample.get("toxicity", 0),
            "virality": sample.get("virality", 0),
            "reports": sample.get("reports", 0),
            "user_reputation": sample.get("user_reputation", 0),
            "ambiguity": sample.get("ambiguity", 0),
            "severity": sample.get("severity", 0),
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State fetch failed: {str(e)}")


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


# --------------------------------------------------
# MAIN ENTRYPOINT
# --------------------------------------------------
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()