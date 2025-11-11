# nlg_story_generator.py
"""
NLG Story Generator with Emotion Control
Uses: Hugging Face Transformers + Emotion Lexicon
"""

from transformers import pipeline
from emotion_lexicon import EmotionLexicon
import random
import json


class EmotionShapedStoryGenerator:
    def __init__(self, model_name="gpt2-medium"):
        self.generator = pipeline("text-generation", model=model_name, device=0)
        self.lex = EmotionLexicon()
        self.arc_presets = {
            "rags_to_riches": [
                (-0.8, "sadness"),
                (-0.3, "effort"),
                (0.4, "hope"),
                (0.9, "joy"),
            ],
            "cinderella": [
                (0.5, "joy"),
                (-0.7, "sadness"),
                (0.3, "hope"),
                (0.8, "joy"),
            ],
        }

    def generate_scene(self, prompt, target_emotion, max_length=150):
        # Emotion-guided prompt
        emo_words = {
            "joy": "happy triumph success bright smile",
            "sadness": "loss grief dark tears failure",
            "hope": "light dawn possibility dream future",
            "effort": "work struggle push fight endure",
        }
        enhanced = f"{prompt} Use words like: {emo_words.get(target_emotion, '')}."
        output = self.generator(
            enhanced, max_length=max_length, num_return_sequences=1, temperature=0.8
        )
        return output[0]["generated_text"]

    def generate_story(self, arc_name="cinderella", character="Maya", setting="lab"):
        arc = self.arc_presets.get(arc_name, self.arc_presets["cinderella"])
        story = f"### {arc_name.replace('_', ' ').title()} Story\n\n"
        story += f"**Character:** {character} | **Setting:** {setting}\n\n"

        for i, (target_score, emo) in enumerate(arc, 1):
            prompt = f"Scene {i}: {character} in the {setting}, feeling {emo}."
            scene = self.generate_scene(prompt, emo)
            actual_score = self.lex.sentiment_score(scene)
            story += f"**Scene {i} ({emo.title()}, Target: {target_score:+.1f}, Got: {actual_score:+.2f})**\n"
            story += scene.strip() + "\n\n"
        return story

    def save_story(self, story, filename="generated_story.md"):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(story)
        print(f"Story saved to {filename}")


# === DEMO ===
if __name__ == "__main__":
    gen = EmotionShapedStoryGenerator()
    story = gen.generate_story(
        arc_name="rags_to_riches", character="Alex", setting="startup office"
    )
    print(story)
    gen.save_story(story, "rags_to_riches_alex.md")
