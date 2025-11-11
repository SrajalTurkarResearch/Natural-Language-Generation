# marketing_emotional_campaign.py
"""
Emotion-Optimized Marketing Story Generator
Generates A/B test variants with different emotional arcs.
Use Case: Ads, Email Campaigns, Brand Storytelling
"""

from nlg_story_generator import EmotionShapedStoryGenerator
from emotion_lexicon import EmotionLexicon
import pandas as pd


class MarketingCampaign:
    def __init__(self):
        self.gen = EmotionShapedStoryGenerator()
        self.lex = EmotionLexicon()
        self.brands = ["EcoPure", "TechFlow", "MindBloom"]
        self.products = ["water filter", "smart watch", "meditation app"]

    def generate_variant(self, brand, product, arc_type, cta):
        prompt = f"Write a short brand story for {brand}'s {product} using a {arc_type} emotional arc. End with: {cta}"
        story = self.gen.generate_scene(prompt, arc_type.split("_")[-1], max_length=200)
        score = self.lex.sentiment_score(story)
        return {
            "brand": brand,
            "product": product,
            "arc": arc_type,
            "story": story,
            "score": score,
            "cta": cta,
        }

    def run_ab_test(self, num_variants=6):
        arcs = ["rags_to_riches", "cinderella"]
        ctas = ["Buy Now", "Learn More", "Start Free Trial"]
        results = []

        for _ in range(num_variants):
            brand = random.choice(self.brands)
            product = random.choice(self.products)
            arc = random.choice(arcs)
            cta = random.choice(ctas)
            variant = self.generate_variant(brand, product, arc, cta)
            results.append(variant)

        df = pd.DataFrame(results)
        df.to_csv("marketing_ab_test.csv", index=False)
        print("A/B Test Variants Saved: marketing_ab_test.csv")
        return df

    def analyze_best_performing(self, df):
        print("\n--- A/B Test Results ---")
        print(
            df[["arc", "score", "cta"]].groupby(["arc", "cta"]).mean()["score"].round(3)
        )
        best = df.loc[df["score"].idxmax()]
        print(f"\nBEST PERFORMING (Score: {best['score']:.3f}):")
        print(best["story"])


# === RUN ===
if __name__ == "__main__":
    campaign = MarketingCampaign()
    df = campaign.run_ab_test(8)
    campaign.analyze_best_performing(df)
