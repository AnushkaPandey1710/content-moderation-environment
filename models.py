from openenv.core.env_server.types import Action, Observation
from pydantic import Field, ConfigDict
from typing import Optional, Dict, Any
from enum import Enum


class ModerationDecision(int, Enum):
    ALLOW = 0
    FLAG = 1
    REMOVE = 2
    ESCALATE = 3


class ContentModerationAction(Action):
    action: ModerationDecision
    confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(use_enum_values=True)


class ContentModerationObservation(Observation):
    current_message: str
    context: Optional[str] = None

    message_length: int
    toxicity: float
    virality: float
    reports: int
    user_reputation: float

    ambiguity: float
    severity: float

    step: int
    total_steps: int

    reward: float
    done: bool

    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

from pydantic import BaseModel

class StepRequest(BaseModel):
    session_id: str
    action: int
    confidence: float = 0.5

class StateRequest(BaseModel):
    session_id: str