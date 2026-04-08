# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Content Moderation Environment Implementation.

A realistic RL environment simulating content moderation decisions.
"""

from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from tasks import TASKS
from graders import grade_easy, grade_medium, grade_hard
from models import (
    ContentModerationAction,
    ContentModerationObservation,
    ModerationDecision
)


class ContentModerationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "basic_toxicity"):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.max_steps = 50
        self.current_idx = 0
        self.data = []

        self.escalations = 0
        self.history = []

        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        self.task_name = task_name
        self.task = TASKS[task_name]

    # --------------------------------------------------
    # DATA GENERATION
    # --------------------------------------------------
    def _generate_single_sample(self):

        texts = [
            ("I hate you idiot", 0.9),
            ("Have a nice day!", 0.1),
            ("You people ruin everything 🙃", 0.7),
            ("This game is killing me lol", 0.3),
            ("Go back to where you came from", 0.85),
            ("I disagree with your opinion", 0.2),
            ("You're the worst person ever", 0.75),
            ("LOL this is insane 😂", 0.2),
        ]

        contexts = [
            "Online gaming chat",
            "Political discussion thread",
            "Casual conversation",
            "Social media comment",
            "Workplace chat"
        ]

        context = random.choice(contexts)
        text, base_toxicity = random.choice(texts)

        toxicity = round(min(1.0, base_toxicity + random.uniform(-0.1, 0.1)), 2)
        virality = round(random.random(), 2)
        reports = random.randint(0, 100)
        user_reputation = round(random.random(), 2)

        ambiguity = (1 - abs(0.5 - toxicity) * 2) * 0.6 \
                    + (virality * 0.2) \
                    + ((1 - user_reputation) * 0.2)

        severity = toxicity

        # Label logic (kept but NOT used for reward anymore)
        if toxicity > 0.8:
            label = 2
        elif toxicity > 0.6:
            label = 3 if ambiguity > 0.4 else 1
        elif reports > 60:
            label = 1
        else:
            label = 0

        return {
            "text": text,
            "context": context,
            "toxicity": toxicity,
            "virality": virality,
            "reports": reports,
            "user_reputation": user_reputation,
            "ambiguity": round(ambiguity, 2),
            "severity": severity,
            "label": label
        }

    def _generate_data(self):
        samples = []
        attempts = 0

        while len(samples) < self.max_steps and attempts < self.max_steps * 10:
            sample = self._generate_single_sample()

            if self.task["filter_fn"](sample):
                samples.append(sample)

            attempts += 1

        while len(samples) < self.max_steps:
            sample = self._generate_single_sample()
            samples.append(sample)

        return samples

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------
    def reset(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.current_idx = 0
        self.data = self._generate_data()

        self.escalations = 0
        self.history = []

        sample = self.data[self.current_idx]

        obs = self._build_observation(sample, reward=0.0, done=False)

        return {
            **obs.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {}
        }

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------
    def step(self, action: ContentModerationAction):

        sample = self.data[self.current_idx]

        # ✅ FIXED: true label aligned with TASKS
        true_label = 1 if self.task["filter_fn"](sample) else 0

        act = int(action.action)
        confidence = action.confidence

        if act == 3:
            self.escalations += 1

        reward = self._calculate_reward(act, confidence, sample)

        # Temporal dynamics (kept as-is)
        if act == 0 and sample["toxicity"] > 0.7:
            sample["virality"] = min(1.0, sample["virality"] + 0.2)

        if act != true_label:
            sample["reports"] = min(100, sample["reports"] + 10)

        # Move forward
        self.current_idx += 1
        self._state.step_count = self.current_idx

        done = self.current_idx >= self.max_steps

        next_sample = sample if done else self.data[self.current_idx]

        # Log history
        self.history.append({
            "step": self._state.step_count,
            "action": act,
            "confidence": confidence,
            "reward": round(reward, 2),
            "true_label": true_label
        })

        # Grading
        if done:
            if self.task_name == "basic_toxicity":
                score = grade_easy(self.history)
            elif self.task_name == "contextual_moderation":
                score = grade_medium(self.history)
            else:
                score = grade_hard(self.history)

            info = {"final_score": score}
        else:
            info = {}

        return {
            **self._build_observation(
                next_sample,
                reward=round(reward, 2),
                done=done,
                true_label=true_label
            ).model_dump(),
            "reward": round(reward, 2),
            "done": done,
            "info": info
        }

    # --------------------------------------------------
    # OBSERVATION BUILDER
    # --------------------------------------------------
    def _build_observation(self, sample, reward, done, true_label=None):
        return ContentModerationObservation(
            current_message=sample["text"],
            context=sample["context"],
            message_length=len(sample["text"]),
            toxicity=sample["toxicity"],
            virality=sample["virality"],
            reports=sample["reports"],
            user_reputation=sample["user_reputation"],
            ambiguity=sample["ambiguity"],
            severity=sample["severity"],
            step=self._state.step_count,
            total_steps=self.max_steps,
            reward=reward,
            done=done,
            metadata={
                "true_label": true_label,
                "true_label_name": ModerationDecision(true_label).name if true_label is not None else None
            }
        )

    # --------------------------------------------------
    # REWARD FUNCTION (FIXED)
    # --------------------------------------------------
    def _calculate_reward(self, act: int, confidence: float, sample: dict) -> float:
        """
        Simplified reward aligned with graders
        """

        true_label = 1 if self.task["filter_fn"](sample) else 0

        reward = 0.0

        # ✅ Correct decision
        if act == true_label:
            reward += 1.0 * confidence

        # ⚠️ Escalation
        elif act == 3:
            if confidence < 0.6:
                reward += 0.2
            else:
                reward -= 0.2

        # ❌ Wrong decision
        else:
            reward -= 1.0 * confidence

        # Penalize too many escalations
        if act == 3:
            escalation_ratio = (self.escalations + 1) / (self.current_idx + 1)
            if escalation_ratio > 0.3:
                reward -= 0.3

        # clamp
        reward = max(-1.5, min(1.5, reward))

        # normalize to [0,1]
        normalized = (reward + 1.5) / 3.0

        return round(normalized, 2)

    # --------------------------------------------------
    # STATE
    # --------------------------------------------------
    @property
    def state(self) -> State:
        return self._state