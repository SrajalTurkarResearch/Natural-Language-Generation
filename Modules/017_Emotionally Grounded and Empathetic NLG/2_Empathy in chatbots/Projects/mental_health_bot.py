# mental_health_bot.py
"""
Woebot-Style Mental Health Support Bot
Uses CBT-inspired prompts, mood tracking, and crisis detection.
"""

from empathetic_generator import EmpatheticResponder
import json
import datetime
from typing import Dict, List
import re


class MentalHealthBot:
    def __init__(self, user_id: str):
        self.responder = EmpatheticResponder()
        self.user_id = user_id
        self.mood_log = self._load_mood_log()
        self.crisis_keywords = ["suicide", "kill myself", "end it", "not worth living"]

    def _load_mood_log(self) -> List[Dict]:
        try:
            with open(f"mood_log_{self.user_id}.json", "r") as f:
                return json.load(f)
        except:
            return []

    def _save_mood_log(self):
        with open(f"mood_log_{self.user_id}.json", "w") as f:
            json.dump(self.mood_log, f, indent=2)

    def check_crisis(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.crisis_keywords)

    def log_mood(self, emotion: str, intensity: float):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "emotion": emotion,
            "intensity": intensity,
        }
        self.mood_log.append(entry)
        self._save_mood_log()

    def respond(self, user_input: str) -> str:
        if self.check_crisis(user_input):
            return (
                "I'm really concerned about you. Please reach out to a trusted person or "
                "call a helpline immediately. In the US: 988 | Internationally: befrienders.org"
            )

        emotion = self.responder.emotion_detector.get_top_emotion(user_input)
        intensity = max(self.responder.emotion_detector.detect(user_input).values())

        self.log_mood(emotion, intensity)

        prompt = f"""
You are a compassionate mental health companion using CBT principles.
User said: "{user_input}"
Detected emotion: {emotion} (intensity: {intensity:.2f})
Recent mood trend: {self._get_mood_trend()}
Respond with:
- Validation
- Gentle reflection
- One small coping suggestion
- Open-ended question
"""
        response = self.responder.generator(prompt, max_length=120)[0]["generated_text"]
        return (
            response.split("Respond with:")[-1].strip()
            if "Respond with:" in response
            else response
        )

    def _get_mood_trend(self) -> str:
        if not self.mood_log:
            return "First check-in"
        recent = [m for m in self.mood_log[-3:] if m["emotion"] in ["sadness", "fear"]]
        if len(recent) >= 2:
            return "You've been feeling down lately"
        return "Stable"


# === RUN ===
if __name__ == "__main__":
    bot = MentalHealthBot(user_id="user123")
    print("Mental Health Bot: Hi, I'm here to listen. How are you feeling?")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        print(f"Bot: {bot.respond(msg)}\n")
