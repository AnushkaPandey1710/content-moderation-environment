# tasks.py

TASKS = {
    "basic_toxicity": {
        "difficulty": "Easy",
        "description": "Moderate content based only on toxicity score.",
        "filter_fn": lambda x: x["toxicity"] > 0.7
    },

    "contextual_moderation": {
        "difficulty": "Medium",
        "description": "Moderate content using toxicity + user reputation + reports.",
        "filter_fn": lambda x: (
            (x["toxicity"] > 0.5 and x["user_reputation"] < 0.4)
            or (x["reports"] > 5 and x["virality"] > 0.6)
        )
    },

    "ambiguous_harm": {
        "difficulty": "Hard",
        "description": "Moderate ambiguous but potentially harmful content.",
        "filter_fn": lambda x: (
            x["ambiguity"] > 0.6 and x["severity"] > 0.6
        )
    }
}