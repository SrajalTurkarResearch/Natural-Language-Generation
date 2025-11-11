# education_tutor_bot.py
"""
Duolingo-Style Encouraging Language Tutor
Detects frustration, celebrates progress, gives hints.
"""

from empathetic_generator import EmpatheticResponder


class LanguageTutorBot:
    def __init__(self, language="Spanish"):
        self.responder = EmpatheticResponder()
        self.language = language
        self.streak = 0
        self.hint_used = False

    def respond_to_answer(self, user_answer: str, correct: bool, question: str) -> str:
        if correct:
            self.streak += 1
            return self._celebrate()
        else:
            self.streak = 0
            return self._encourage(question, user_answer)

    def _celebrate(self) -> str:
        cheers = [
            f"¡Excelente! {self.streak} in a row!",
            "You're on fire!",
            "Perfect! Keep it up!",
            "¡Muy bien! You're improving fast.",
        ]
        return random.choice(cheers)

    def _encourage(self, question: str, user_answer: str) -> str:
        emotion = self.responder.emotion_detector.get_top_emotion(user_answer)
        if emotion == "sadness":
            return (
                "It's okay to make mistakes — that's how we learn! "
                "The answer was close. Want a hint?"
            )
        return "Not quite, but you're close! Try again?"

    def give_hint(self, word: str) -> str:
        self.hint_used = True
        return f"Hint: The word starts with '{word[0]}' and means..."


# === RUN ===
if __name__ == "__main__":
    bot = LanguageTutorBot()
    print("Tutor: Translate 'hello' to Spanish:")
    while True:
        ans = input("You: ")
        if ans.lower() == "quit":
            break
        correct = ans.strip().lower() == "hola"
        print(f"Tutor: {bot.respond_to_answer(ans, correct, 'hello')}\n")
