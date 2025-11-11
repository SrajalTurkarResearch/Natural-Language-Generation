"""
PROJECT: Emotion-Driven Marketing Email Generator
USE CASE: Digital Marketing (Inspired by Persado)
GOAL: Generate high-conversion email subject lines and body with targeted emotions.

Author: [Your Name] – AI Marketing Scientist
Date: October 29, 2025
"""

import random
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotion-to-copy mapping (based on marketing psychology)
EMOTION_STRATEGIES = {
    "urgency": {
        "subjects": [
            "Last Chance: {product} Ends in 3 Hours!",
            "Hurry! Only {n} {product} Left at This Price",
            "Flash Sale: {product} – 24 Hours Only!",
        ],
        "body_openers": [
            "Time is running out!",
            "This deal won’t last!",
            "Don’t miss your final opportunity!",
        ],
        "ctas": ["Shop Now Before It's Gone", "Claim Your Deal", "Act Fast"],
    },
    "exclusivity": {
        "subjects": [
            "VIP Access: {product} Just for You",
            "Exclusive Invite: Early Access to {product}",
            "You’re Invited: Limited {product} Preview",
        ],
        "body_openers": [
            "As one of our top customers...",
            "You’ve been selected for early access...",
            "This is exclusive — just for you.",
        ],
        "ctas": ["Get Early Access", "Unlock VIP Deal", "Reserve Your Spot"],
    },
    "joy": {
        "subjects": [
            "Good News! {product} is Back in Stock",
            "Smile – Your {product} is Here!",
            "Great Day to Treat Yourself: {product}",
        ],
        "body_openers": [
            "We have exciting news!",
            "You’re going to love this...",
            "Something to brighten your day!",
        ],
        "ctas": ["Shop with a Smile", "Treat Yourself", "Get Yours Now"],
    },
}

# Initialize text generator
generator = pipeline("text-generation", model="gpt2", max_length=150, truncation=True)


def generate_emotional_email(
    emotion, product, customer_name="Valued Customer", n_left=None
):
    """Generate full email with subject, body, and CTA."""
    strategy = EMOTION_STRATEGIES[emotion]

    subject = random.choice(strategy["subjects"]).format(
        product=product, n=n_left or "a few"
    )
    opener = random.choice(strategy["body_openers"])
    cta = random.choice(strategy["ctas"])

    prompt = f"""
    Write a warm, persuasive marketing email body.
    Emotion: {emotion}
    Product: {product}
    Opener: {opener}
    CTA: {cta}
    Keep under 100 words. Be human, not robotic.
    """

    body = generator(prompt, temperature=0.8, num_return_sequences=1)[0][
        "generated_text"
    ].split("\n\n")[0]

    email = f"""
Subject: {subject}

Dear {customer_name},

{opener}

{body}

→ {cta}

Best,
The Team
    """.strip()

    return {
        "emotion": emotion,
        "subject": subject,
        "body": body,
        "cta": cta,
        "full_email": email,
    }


def ab_test_simulation(emotions, product, runs=1000):
    """Simulate A/B test with click-through rates (CTR) based on emotion."""
    ctr_data = []
    base_ctr = 0.03  # 3% average

    emotion_boosts = {"urgency": 0.05, "exclusivity": 0.04, "joy": 0.02}

    for emotion in emotions:
        boost = emotion_boosts.get(emotion, 0.0)
        ctr = base_ctr + boost + random.gauss(0, 0.005)
        ctr_data.append({"Emotion": emotion.capitalize(), "CTR": ctr})

    df = pd.DataFrame(ctr_data)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["Emotion"], df["CTR"], color=["#FF5722", "#7B1FA2", "#4CAF50"])
    plt.title("Simulated A/B Test: Email CTR by Emotion")
    plt.ylabel("Click-Through Rate")
    plt.ylim(0, 0.1)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{height:.1%}",
            ha="center",
            fontsize=10,
        )
    plt.show()

    return df


# ——— MAIN ———
if __name__ == "__main__":
    product = "Wireless Earbuds Pro"
    emotions_to_test = ["urgency", "exclusivity", "joy"]

    print("Emotion-Driven Marketing Email Generator\n")
    for emotion in emotions_to_test:
        email = generate_emotional_email(
            emotion, product, n_left=12 if emotion == "urgency" else None
        )
        print(f"\n--- {emotion.upper()} EMAIL ---")
        print(email["full_email"])
        print("\n" + "=" * 60)

    print("\nRunning A/B Test Simulation...")
    results = ab_test_simulation(emotions_to_test, product)
    print("\nA/B Test Results:")
    print(results)
