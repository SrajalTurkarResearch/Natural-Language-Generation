"""
feedback_loops.py
=================
Module 5: Simulate Human Feedback Loop
Includes reward model training.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ResponsePair:
    prompt: str
    response_a: str
    response_b: str
    winner: str  # 'A' or 'B'


class RewardModel:
    """Simple reward model that learns from human preferences"""

    def __init__(self):
        self.scores = {}  # (prompt, response) → score

    def predict(self, prompt: str, response: str) -> float:
        key = (prompt, response)
        return self.scores.get(key, 0.0)

    def train_from_feedback(self, pair: ResponsePair):
        score_a = self.predict(pair.prompt, pair.response_a)
        score_b = self.predict(pair.prompt, pair.response_b)

        # Bradley-Terry update
        if pair.winner == "A":
            self.scores[(pair.prompt, pair.response_a)] = score_a + 0.1
            self.scores[(pair.prompt, pair.response_b)] = score_b - 0.1
        else:
            self.scores[(pair.prompt, pair.response_a)] = score_a - 0.1
            self.scores[(pair.prompt, pair.response_b)] = score_b + 0.1


# === SIMULATE FEEDBACK LOOP ===
def simulate_rlhf():
    print("🤖 RLHF SIMULATION\n")

    reward_model = RewardModel()

    feedback_data = [
        ResponsePair(
            prompt="Explain gravity",
            response_a="Gravity pulls things down.",
            response_b="Gravity is the curvature of spacetime caused by mass.",
            winner="B",
        ),
        ResponsePair(
            prompt="What is AI?",
            response_a="AI is robots.",
            response_b="AI is systems that mimic human intelligence.",
            winner="B",
        ),
    ]

    for i, pair in enumerate(feedback_data):
        print(f"Feedback {i+1}:")
        print(f"  A: {pair.response_a}")
        print(f"  B: {pair.response_b}")
        print(f"  Human chose: {pair.winner}\n")

        reward_model.train_from_feedback(pair)

    # Test reward
    print("🎯 REWARD MODEL TEST")
    print(
        f"Score A (simple): {reward_model.predict('Explain gravity', feedback_data[0].response_a):.2f}"
    )
    print(
        f"Score B (detailed): {reward_model.predict('Explain gravity', feedback_data[0].response_b):.2f}"
    )


if __name__ == "__main__":
    simulate_rlhf()
