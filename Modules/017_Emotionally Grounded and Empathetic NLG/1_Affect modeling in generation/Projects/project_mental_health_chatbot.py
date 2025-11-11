"""
PROJECT: Empathetic Mental Health Chatbot using Affective NLG
USE CASE: Mental Health Support (Inspired by Woebot)
GOAL: Generate emotionally supportive, CBT-informed responses based on user mood.

Author: [Your Name] – Aspiring AI Scientist
Date: October 29, 2025
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime

# Setup logging for scientific reproducibility
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download required resources
nltk.download("vader_lexicon", quiet=True)

# Initialize models
try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
    generator = pipeline(
        "text-generation", model="gpt2-medium", max_length=100, truncation=True
    )
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

# CBT-inspired response templates
CBT_TEMPLATES = {
    "anxiety": [
        "That sounds really overwhelming. Let's try a grounding exercise: name 5 things you can see right now.",
        "Anxiety can feel so heavy. Would you like to try a quick breathing exercise together?",
        "It's okay to feel anxious. Let's reframe: what’s one thing you *can* control right now?",
    ],
    "sadness": [
        "I'm really sorry you're feeling this way. It's okay to not be okay. Want to talk about what's hurting?",
        "Sadness is tough. You're not alone — I'm here. What would help you feel even 1% better today?",
        "Your feelings are valid. Even small steps count. Would you like to journal one thought?",
    ],
    "stress": [
        "Stress can pile up fast. Let's pause: inhale for 4, hold for 4, exhale for 4. Try it with me?",
        "You're carrying a lot. Let's break it down: what's the *next* small action you can take?",
        "Your body is signaling overload. How about a 2-minute stretch or water break?",
    ],
}


def detect_mood(text):
    """Detect primary negative emotion using keyword + VADER compound score."""
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]
    text_lower = text.lower()

    if compound < -0.3:
        if any(
            word in text_lower for word in ["stress", "overwhelm", "busy", "pressure"]
        ):
            return "stress", scores
        elif any(
            word in text_lower for word in ["anxious", "worry", "panic", "nervous"]
        ):
            return "anxiety", scores
        elif any(word in text_lower for word in ["sad", "down", "depressed", "lonely"]):
            return "sadness", scores
        else:
            return "general_distress", scores
    else:
        return "neutral_positive", scores


def generate_cbt_response(user_input, mood, scores):
    """Generate empathetic + CBT-guided response."""
    base_prompt = f"Respond as a compassionate mental health coach using CBT principles. User said: '{user_input}'. Mood: {mood}. Be warm, validating, and offer one small action."

    try:
        output = generator(
            base_prompt, max_length=120, num_return_sequences=1, temperature=0.7
        )[0]["generated_text"]
        return output
    except Exception as e:
        logger.warning(f"Generation failed, using template: {e}")
        return CBT_TEMPLATES.get(mood, ["I'm here for you."])[0]


def visualize_mood(scores, mood):
    """Plot sentiment breakdown."""
    labels = ["Positive", "Neutral", "Negative", "Compound"]
    values = [scores["pos"], scores["neu"], scores["neg"], scores["compound"]]
    colors = ["#66BB6A", "#42A5F5", "#EF5350", "#AB47BC"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.title(f"Detected Mood: {mood.capitalize()}", fontsize=14, pad=20)
    plt.ylabel("Sentiment Score")
    plt.ylim(-1, 1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom" if height > 0 else "top",
        )

    plt.tight_layout()
    plt.show()


# ——— MAIN CHAT LOOP ———
def mental_health_chatbot():
    print("Empathetic Mental Health Chatbot (Type 'quit' to exit)\n")
    logger.info("Chatbot session started.")

    conversation_log = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Take care. You're stronger than you know.")
            break

        mood, scores = detect_mood(user_input)
        response = generate_cbt_response(user_input, mood, scores)

        print(f"Bot: {response}\n")
        visualize_mood(scores, mood)

        # Log interaction
        conversation_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "mood": mood,
                "scores": scores,
                "response": response,
            }
        )

    # Save log
    with open("mental_health_chat_log.json", "w") as f:
        json.dump(conversation_log, f, indent=2)
    logger.info("Session ended. Log saved.")


if __name__ == "__main__":
    mental_health_chatbot()
