# mini_project.py
"""
Mini-project: A contextual NLG system for generating personalized recommendations.
Purpose: Demonstrate memory and context in a real-world NLG application.
Dependencies: pandas for mock dataset, random for simulation
"""

import pandas as pd
import random


class RecommendationSystem:
    def __init__(self):
        """Initialize memory and mock dataset."""
        self.memory = {}
        # Mock dataset of recommendations
        self.dataset = pd.DataFrame(
            {
                "category": ["food", "food", "travel", "travel"],
                "item": ["sushi", "pizza", "Paris", "Tokyo"],
                "description": [
                    "Fresh sushi rolls",
                    "Cheesy pizza",
                    "Eiffel Tower visit",
                    "Shibuya crossing",
                ],
            }
        )

    def store_preference(self, user_id, category, item):
        """Store user preferences in memory."""
        if user_id not in self.memory:
            self.memory[user_id] = {"preferences": {}, "history": []}
        self.memory[user_id]["preferences"][category] = item
        self.memory[user_id]["history"].append(f"Set favorite {category} to {item}")

    def generate_recommendation(self, user_id, category):
        """Generate a recommendation based on memory and context."""
        if user_id in self.memory and category in self.memory[user_id]["preferences"]:
            favorite = self.memory[user_id]["preferences"][category]
            return (
                f"Based on your favorite {category} ({favorite}), try {favorite} again!"
            )

        # Fallback: Recommend a random item from dataset
        options = self.dataset[self.dataset["category"] == category]["item"].tolist()
        if not options:
            return f"No {category} recommendations available."
        item = random.choice(options)
        self.memory[user_id]["history"].append(f"Recommended {item}")
        return f"Try this {category}: {item}!"


# Example usage
if __name__ == "__main__":
    rec_system = RecommendationSystem()
    rec_system.store_preference("user1", "food", "sushi")
    print(rec_system.generate_recommendation("user1", "food"))
    print(rec_system.generate_recommendation("user1", "travel"))
