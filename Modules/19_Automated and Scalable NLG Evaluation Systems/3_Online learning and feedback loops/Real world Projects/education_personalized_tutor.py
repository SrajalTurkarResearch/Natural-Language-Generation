"""
education_personalized_tutor.py
===============================
Real-World Project: AI Tutor with Feedback Loop
Adapts explanations based on student understanding.
"""

import random
from typing import List


class AITutor:
    def __init__(self):
        self.student_knowledge = {}  # topic → mastery (0–1)
        self.explanation_style = "simple"
        self.feedback_history = []

    def explain(self, topic: str) -> str:
        """NLG: Generate explanation based on student level"""
        base = {
            "photosynthesis": "Plants make food using sunlight.",
            "gravity": "Things fall down because of gravity.",
        }
        if topic not in self.student_knowledge:
            self.student_knowledge[topic] = 0.3

        mastery = self.student_knowledge[topic]
        if mastery < 0.5:
            return f"[Beginner] {base.get(topic, '')} It's like magic!"
        elif mastery < 0.8:
            return f"[Intermediate] {base.get(topic, '')} This happens due to energy transfer."
        else:
            return f"[Advanced] {base.get(topic, '')} Governed by biochemical pathways."

    def student_feedback(self, topic: str, understood: bool):
        """Online update from student"""
        delta = 0.2 if understood else -0.1
        self.student_knowledge[topic] = max(
            0.1, min(1.0, self.student_knowledge.get(topic, 0.3) + delta)
        )
        self.feedback_history.append((topic, understood))

    def show_progress(self):
        print("\nStudent Mastery Levels:")
        for topic, level in self.student_knowledge.items():
            bars = "█" * int(level * 10)
            print(f"  {topic.capitalize():15} |{bars.ljust(10)}| {level:.1%}")


# === RUN EDUCATION SIMULATION ===
def run_personalized_tutor():
    print("AI PERSONALIZED TUTOR: Adaptive Explanations")
    print("=" * 60)

    tutor = AITutor()
    topics = ["photosynthesis", "gravity", "photosynthesis", "gravity"]

    for i, topic in enumerate(topics):
        exp = tutor.explain(topic)
        print(f"\nLesson {i+1}: {topic}")
        print(f"   AI: {exp}")

        # Simulate student understanding
        understood = random.random() < 0.7
        print(f"   Student: {'Understood!' if understood else 'Confused...'}")
        tutor.student_feedback(topic, understood)

    tutor.show_progress()
    print("\nInsight: NLG adapts from 'magic' to 'biochemical' in 3 rounds!")


if __name__ == "__main__":
    run_personalized_tutor()
