# emotion_detector.py
"""
Emotion Detection Module
Detects emotions in user text using transformer models.
Supports keyword fallback and probability output.
"""

from transformers import pipeline
import numpy as np


class EmotionDetector:
    def __init__(self, model_name="bhadresh-savani/distilbert-base-uncased-emotion"):
        """
        Initialize with a pre-trained emotion classification model.
        """
        self.emotion_pipeline = pipeline(
            "text-classification", model=model_name, return_all_scores=True
        )
        self.emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"]

    def detect(self, text: str) -> dict:
        """
        Detect emotion probabilities from text.
        Returns: {emotion: probability}
        """
        results = self.emotion_pipeline(text)[0]
        return {r["label"]: r["score"] for r in results}

    def get_top_emotion(self, text: str) -> str:
        """
        Return the most likely emotion.
        """
        probs = self.detect(text)
        return max(probs, key=probs.get)

    def keyword_fallback(self, text: str) -> str:
        """
        Simple keyword-based emotion detection (for debugging or low-resource).
        """
        text = text.lower()
        keywords = {
            "joy": ["happy", "great", "excited", "awesome"],
            "sadness": ["sad", "down", "depressed", "hopeless"],
            "anger": ["angry", "mad", "furious", "hate"],
            "fear": ["scared", "afraid", "worried", "anxious"],
        }
        for emotion, words in keywords.items():
            if any(word in text for word in words):
                return emotion
        return "neutral"


# === DEMO ===
if __name__ == "__main__":
    detector = EmotionDetector()
    text = "I just failed my exam and feel like giving up."
    print("Probabilities:", detector.detect(text))
    print("Top Emotion:", detector.get_top_emotion(text))
