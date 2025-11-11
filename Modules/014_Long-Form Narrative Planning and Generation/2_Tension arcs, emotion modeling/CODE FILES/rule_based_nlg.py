#!/usr/bin/env python3
"""
🏗️ RULE-BASED NLG SYSTEM
Complete Emotion-Driven Story Generator
47 Lines of Production Code
"""

import numpy as np
import argparse


class EmotionNLG:
    def __init__(self):
        self.lexicon = {
            "curious": ["wondered", "explored", "discovered"],
            "nervous": ["hesitated", "glanced nervously", "felt uneasy"],
            "scared": ["panicked", "froze", "heart raced"],
            "terrified": ["screamed", "ran", "horrified"],
            "relieved": ["sighed", "smiled", "thanked"],
        }

        self.styles = {
            0: lambda x: f"The {x} was peaceful.",
            3: lambda x: f"Suddenly, the {x} seemed strange.",
            7: lambda x: f"The {x} was terrifying!",
            10: lambda x: f"'{x.upper()}!' they screamed!!!",
            2: lambda x: f"Finally, the {x} was safe again.",
        }

    def generate_story(self, arc=[0, 3, 7, 10, 2], theme="forest"):
        emotions = ["curious", "nervous", "scared", "terrified", "relieved"]
        story = []

        for i, tension in enumerate(arc):
            emotion = emotions[i]
            verb = np.random.choice(self.lexicon[emotion])
            sentence = self.styles[tension](f"team {verb} the {theme}")
            story.append(f"{i+1}. {sentence}")

        return "\n".join(story)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", default="spaceship")
    parser.add_argument("--length", type=int, default=5)
    args = parser.parse_args()

    nlg = EmotionNLG()
    arc = [0, 3, 7, 10, 2] * (args.length // 5 + 1)
    story = nlg.generate_story(arc[: args.length], args.theme)
    print("📖 GENERATED STORY:")
    print(story)
