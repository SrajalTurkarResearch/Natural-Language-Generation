"""
main_experiment.py
==================
Module 10: Full Adaptive NLG System
Combines online learning + feedback
"""

from online_learning import online_update
from feedback_loops import RewardModel, ResponsePair
import time


def run_adaptive_nlg_experiment():
    print("🚀 FULL ADAPTIVE NLG EXPERIMENT")
    print("=" * 50)

    # Step 1: Initial model (no update)
    print("\n1. Initial Generation (before feedback)")
    print("   Prompt: 'The future of AI is'")
    print("   [Model hasn't learned yet]\n")
    time.sleep(1)

    # Step 2: Simulate 3 rounds of feedback
    feedback_rounds = [
        (
            "The future of AI is",
            "The future of AI is bright and full of potential.",
            "The future of AI is robots taking over.",
            "A",
        ),
        (
            "Climate change is",
            "Climate change is a global challenge requiring urgent action.",
            "Climate change is fake.",
            "A",
        ),
        (
            "Science is",
            "Science is the systematic study of the natural world.",
            "Science is magic.",
            "A",
        ),
    ]

    reward_model = RewardModel()

    print("2. FEEDBACK LOOP STARTED")
    for i, (prompt, good, bad, winner) in enumerate(feedback_rounds):
        print(f"\n   Round {i+1}:")
        print(f"   Prompt: {prompt}")
        print(f"   Good: {good}")
        print(f"   Bad : {bad}")
        print(f"   Human chose: {winner}")

        # Online update with good response
        online_update(prompt, good, reward=0.9)

        # Train reward model
        pair = ResponsePair(prompt, bad, good, winner)
        reward_model.train_from_feedback(pair)

    print("\n✅ Model now adapted to high-quality, factual responses!")
    print("   Ready for research deployment.")


if __name__ == "__main__":
    run_adaptive_nlg_experiment()
