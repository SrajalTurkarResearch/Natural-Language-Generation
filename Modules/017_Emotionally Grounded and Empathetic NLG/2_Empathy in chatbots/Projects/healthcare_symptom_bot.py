# healthcare_symptom_bot.py
"""
Ada Health-Style Empathetic Symptom Checker
Reassures, asks follow-up, avoids alarmism.
"""

from empathetic_generator import EmpatheticResponder


class SymptomCheckerBot:
    def __init__(self):
        self.responder = EmpatheticResponder()
        self.symptoms = []
        self.severity = "mild"

    def ask_initial(self) -> str:
        return "I’m here to help. What symptoms are you experiencing?"

    def process_symptom(self, user_input: str) -> str:
        self.symptoms.append(user_input)
        emotion = self.responder.emotion_detector.get_top_emotion(user_input)

        if any(w in user_input.lower() for w in ["chest pain", "shortness", "faint"]):
            self.severity = "urgent"
            return (
                "I understand this is worrying. Chest pain can be serious. "
                "Please call emergency services now or go to ER."
            )

        reassurance = (
            "I hear you. That sounds uncomfortable."
            if emotion in ["fear", "sadness"]
            else "Thanks for sharing."
        )
        follow_up = self._next_question()
        return f"{reassurance} {follow_up}"

    def _next_question(self) -> str:
        questions = [
            "When did it start?",
            "Is it constant or comes and goes?",
            "Any other symptoms?",
        ]
        return questions[min(len(self.symptoms) - 1, len(questions) - 1)]

    def summarize(self) -> str:
        return (
            f"Summary: {', '.join(self.symptoms[:3])}\n"
            f"Based on this, I suggest seeing a doctor soon. Would you like a symptom report?"
        )


# === RUN ===
if __name__ == "__main__":
    bot = SymptomCheckerBot()
    print(bot.ask_initial())
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        print(f"Bot: {bot.process_symptom(msg)}\n")
