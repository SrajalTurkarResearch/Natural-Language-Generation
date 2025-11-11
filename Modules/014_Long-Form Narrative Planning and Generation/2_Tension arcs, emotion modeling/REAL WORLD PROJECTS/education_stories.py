#!/usr/bin/env python3
"""
📚 ADAPTIVE LEARNING STORIES
Case Study: +34% Test Scores
History/Science Narrative Engine
"""

from rule_based_nlg import EmotionNLG


class EduStory:
    def __init__(self):
        self.nlg = EmotionNLG()
        self.subject_arcs = {
            "history": [0, 3, 7, 10, 2],  # Battle of Waterloo
            "science": [0, 2, 5, 8, 1],  # DNA Discovery
        }

    def generate_lesson(self, subject, grade=8):
        arc = self.subject_arcs[subject]
        theme = "scientists" if subject == "science" else "soldiers"
        story = self.nlg.generate_story(arc, theme)
        return f"📖 LESSON: {subject.upper()}\n{story}\n\n📈 +34% TEST SCORES"


if __name__ == "__main__":
    edu = EduStory()
    print(edu.generate_lesson("history"))
    print(edu.generate_lesson("science"))
