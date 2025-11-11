"""
customer_support_chatbot.py
============================
Real-World Project: Customer Support Bot
Learns from user ratings (thumbs up/down).
"""

import json
from collections import defaultdict


class SupportBot:
    def __init__(self):
        self.response_db = defaultdict(list)
        self.ratings = []

    def respond(self, query: str) -> str:
        responses = {
            "refund": [
                "We can process your refund within 3 days.",
                "Refund initiated. Check email.",
            ],
            "delivery": ["Your order ships today!", "Expected delivery: 2-3 days."],
        }
        key = query.split()[0].lower()
        candidates = responses.get(key, ["I'll escalate this to a human agent."])
        # Pick highest-rated past response
        if self.response_db[key]:
            return max(self.response_db[key], key=lambda x: x[1])[0]
        return random.choice(candidates)

    def user_rating(self, query: str, response: str, rating: int):
        """Online feedback: thumbs up/down"""
        key = query.split()[0].lower()
        self.response_db[key].append((response, rating))
        self.ratings.append(rating)
        print(f"   User rated: {'Thumbs up' if rating > 0 else 'Thumbs down'}")


# === RUN SUPPORT BOT ===
def run_support_bot():
    print("CUSTOMER SUPPORT CHATBOT: Learns from Ratings")
    print("=" * 60)

    bot = SupportBot()
    interactions = [
        ("refund policy", 1),
        ("refund policy", -1),
        ("delivery time", 1),
        ("refund policy", 1),
    ]

    for query, rating in interactions:
        resp = bot.respond(query)
        print(f"\nUser: {query}")
        print(f" Bot: {resp}")
        bot.user_rating(query, resp, rating)

    print(f"\nFinal 'refund' response: {bot.respond('refund policy')}")
    print("Insight: Bot learns to avoid low-rated responses in <5 interactions.")


if __name__ == "__main__":
    run_support_bot()
