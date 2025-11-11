# mental_health_story_therapist.py
"""
AI Emotional Narrative Therapist
Generates personalized healing stories based on user mood.
Use Case: Mental Health Support, Grief Counseling, Resilience Building
"""

import json
import logging
from datetime import datetime
from emotion_lexicon import EmotionLexicon
from nlg_story_generator import EmotionShapedStoryGenerator
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NarrativeTherapist:
    def __init__(self):
        self.lex = EmotionLexicon()
        self.gen = EmotionShapedStoryGenerator(model_name="gpt2-medium")
        self.therapeutic_arcs = {
            "grief_to_acceptance": [
                (-0.8, "sadness"),
                (-0.4, "reflection"),
                (0.2, "hope"),
                (0.7, "acceptance"),
            ],
            "anxiety_to_calm": [
                (-0.6, "fear"),
                (-0.2, "breathing"),
                (0.3, "grounding"),
                (0.8, "calm"),
            ],
            "failure_to_growth": [
                (-0.7, "failure"),
                (-0.3, "effort"),
                (0.4, "learning"),
                (0.9, "growth"),
            ],
        }

    def assess_user_mood(self, user_input):
        """Estimate dominant emotion from user text"""
        score = self.lex.sentiment_score(user_input)
        dom = self.lex.dominant_emotion(user_input)
        logger.info(f"User mood: {dom} | Score: {score:.2f}")
        return dom, score

    def recommend_arc(self, emotion):
        mapping = {
            "sadness": "grief_to_acceptance",
            "fear": "anxiety_to_calm",
            "anger": "failure_to_growth",
            "disgust": "failure_to_growth",
            "joy": "failure_to_growth",  # Reinforce positivity
        }
        return mapping.get(emotion, "failure_to_growth")

    def generate_therapy_story(self, user_name, user_input, custom_character=None):
        emotion, _ = self.assess_user_mood(user_input)
        arc = self.recommend_arc(emotion)
        character = custom_character or user_name

        # Personalize prompt
        prompt = f"You are feeling {emotion}. This is a story about {character} who felt the same..."
        logger.info(f"Generating therapeutic story with arc: {arc}")

        # Use custom arc
        self.gen.arc_presets[arc] = self.therapeutic_arcs[arc]
        story = self.gen.generate_story(
            arc_name=arc, character=character, setting="a quiet journey"
        )

        # Add therapeutic closing
        closing = "\n\n**Therapist's Note:** Just like in the story, your feelings are valid. Healing takes time, but growth is possible. You're not alone."
        full_story = story + closing

        # Save session
        session = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "detected_emotion": emotion,
            "arc_used": arc,
            "story": full_story,
        }
        with open(
            f"therapy_session_{user_name}_{int(datetime.now().timestamp())}.json", "w"
        ) as f:
            json.dump(session, f, indent=2)

        return full_story


# === DEMO ===
if __name__ == "__main__":
    therapist = NarrativeTherapist()
    user_input = input("How are you feeling today? Tell me in a sentence: ")
    name = input("Your name (optional): ") or "Friend"
    story = therapist.generate_therapy_story(name, user_input)
    print("\n" + "=" * 60)
    print("YOUR HEALING STORY")
    print("=" * 60)
    print(story)
