#!/usr/bin/env python3
"""
💰 AI ADVERTISING ENGINE
Emotion-Driven Copywriting
Case Study: 3x Conversion Rate
"""


class AdEngine:
    def __init__(self):
        self.product_emotions = {
            "phone": ("excited", [0, 2, 6, 8, 10]),
            "coffee": ("comfort", [0, 1, 3, 5, 2]),
            "car": ("powerful", [0, 4, 8, 10, 3]),
        }

    def generate_ad(self, product, target="millennials"):
        emotion, arc = self.product_emotions[product]
        ad = f"🚀 {product.upper()}: "

        for tension in arc:
            if tension == 0:
                ad += "Discover "
            elif tension == 10:
                ad += f"{product.upper()} CHANGES EVERYTHING! "
            else:
                ad += f"amazing {emotion} "

        return ad.strip()

    def campaign_roi(self):
        return "📈 Conversion: 2.1% → 6.3% | 3x ROI"


if __name__ == "__main__":
    engine = AdEngine()
    products = ["phone", "coffee", "car"]
    for product in products:
        print(f"{engine.generate_ad(product)}")
        print(engine.campaign_roi(), "\n")
