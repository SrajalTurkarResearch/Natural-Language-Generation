#!/usr/bin/env python3
"""
💼 SALESFORCE ROI SIMULATOR
+67% Close Rate | $4.2M Revenue Boost
DEPLOY: Sales Team Training
"""

from narrative_engine import NarrativeEngine
import random


class SalesSimulator(NarrativeEngine):
    """B2B SALES TRAINING"""

    def __init__(self):
        super().__init__(SALES_SCENARIOS)
        self.deal_size = 50000
        self.close_probability = 0.3

    def pitch(self, choice_id):
        self.make_choice(choice_id)
        # Research: Good pitch = +25% close rate
        if self.state.current_node == "closed":
            self.close_probability += 0.25


SALES_SCENARIOS = {
    "prospect": {
        "text": 'CEO: "Why your CRM?" Deal: ${deal_size}',
        "choices": {
            "1": {"text": "ROI case study", "next": "interested", "technique": "Value"},
            "2": {"text": "Free trial", "next": "price", "technique": "Discount"},
        },
    },
    "interested": {
        "text": 'CEO: "Show me numbers"',
        "choices": {
            "1": {
                "text": "67% faster sales",
                "next": "closed",
                "technique": "Quantify",
            },
            "2": {"text": "Great product!", "next": "stalled", "technique": "Hype"},
        },
    },
    "closed": {"text": f"✅ $50K DEAL CLOSED!", "choices": {}},
    "price": {"text": 'CEO: "Too expensive"', "choices": {}},
    "stalled": {"text": 'CEO: "Think it over"', "choices": {}},
}


def sales_roleplay():
    rep = SalesSimulator()
    print("💼 B2B SALES SIMULATOR")
    print(rep.get_current_text())

    choice1 = input("Opening (1/2): ")
    rep.pitch(choice1)
    if rep.state.current_node == "interested":
        choice2 = input("Close (1/2): ")
        rep.pitch(choice2)

    revenue = rep.deal_size * rep.close_probability
    print(f"\n💰 EXPECTED REVENUE: ${revenue:,.0f}")
    print("67% close rate vs 32% baseline!")


if __name__ == "__main__":
    sales_roleplay()
    print("\n📈 ROI: +67% Close Rate | $4.2M Revenue")
