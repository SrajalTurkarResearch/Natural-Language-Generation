# educational_story_builder.py
"""
Emotion-Rich Educational Story Generator
Teaches science, history, or values through emotional narratives.
Use Case: EdTech, Curriculum Design, Moral Education
"""

from nlg_story_generator import EmotionShapedStoryGenerator
from emotion_lexicon import EmotionLexicon


class EducationalStoryBuilder:
    def __init__(self):
        self.gen = EmotionShapedStoryGenerator()
        self.lex = EmotionLexicon()
        self.topics = {
            "photosynthesis": "how plants eat sunlight",
            "gravity": "why things fall down",
            "kindness": "helping others feels good",
            "recycling": "saving the Earth",
        }

    def build_lesson(self, topic, age_group="8-12"):
        if topic not in self.topics:
            print("Topic not found!")
            return

        desc = self.topics[topic]
        if "8" in age_group:
            arc = "cinderella"  # Wonder → Confusion → Aha! → Joy
            character = "a curious child named Sam"
        else:
            arc = "rags_to_riches"
            character = "a young scientist"

        prompt = f"Explain {desc} through a story about {character}. Use simple words. Teach one key fact."
        story = self.gen.generate_story(
            arc_name=arc, character=character, setting="a magical garden"
        )

        # Add quiz
        quiz = f"\n**Quick Quiz:** What did {character} learn about {topic}?\nAnswer: ____________________"

        full_lesson = story + quiz
        filename = f"lesson_{topic}_{age_group}.md"
        with open(filename, "w") as f:
            f.write(full_lesson)
        print(f"Lesson saved: {filename}")
        return full_lesson


# === DEMO ===
if __name__ == "__main__":
    builder = EducationalStoryBuilder()
    topic = input("Choose topic (photosynthesis/gravity/kindness/recycling): ")
    age = input("Age group (8-12 or 13-16): ") or "8-12"
    lesson = builder.build_lesson(topic, age)
    print("\n" + lesson)
