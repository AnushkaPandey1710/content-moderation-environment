# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Content Moderation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from content_moderation.models import (
    ContentModerationAction,
    ContentModerationObservation,
)


class ContentModerationEnv(
    EnvClient[ContentModerationAction, ContentModerationObservation, State]
):
    """
    Client for the Content Moderation Environment.
    """

    # --------------------------------------------------
    # STEP PAYLOAD
    # --------------------------------------------------
    def _step_payload(self, action: ContentModerationAction) -> Dict:
        return {
            "action": int(action.action),   # ✅ Enum → int
            "confidence": action.confidence  # ✅ REQUIRED
        }

    # --------------------------------------------------
    # PARSE STEP RESULT
    # --------------------------------------------------
    def _parse_result(self, payload: Dict) -> StepResult[ContentModerationObservation]:

        obs_data = payload.get("observation", {})

        observation = ContentModerationObservation(
            current_message=obs_data.get("current_message", ""),
            context=obs_data.get("context"),

            message_length=obs_data.get("message_length", 0),
            toxicity=obs_data.get("toxicity", 0.0),
            virality=obs_data.get("virality", 0.0),
            reports=obs_data.get("reports", 0),
            user_reputation=obs_data.get("user_reputation", 0.0),

            ambiguity=obs_data.get("ambiguity", 0.0),
            severity=obs_data.get("severity", 0.0),

            step=obs_data.get("step", 0),
            total_steps=obs_data.get("total_steps", 0),

            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),

            metadata=obs_data.get("metadata", {})
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {})  # ✅ include grader output
        )

    # --------------------------------------------------
    # PARSE STATE
    # --------------------------------------------------
    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )