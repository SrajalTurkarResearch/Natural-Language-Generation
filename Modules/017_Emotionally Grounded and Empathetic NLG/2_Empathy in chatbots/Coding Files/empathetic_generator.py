# empathetic_generator.py
"""
Empathetic Response Generator
Uses prompt engineering + LLMs to generate caring, context-aware replies.
"""

from transformers import pipeline
from context_manager import ContextManager
from emotion_detector import EmotionDetector


class EmpatheticResponder:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        self.generator = pipeline("text-generation", model=model_name)
        self.emotion_detector = EmotionDetector()
        self.context_manager = ContextManager(max_history=5)

    def respond(self, user_input: str) -> str:
        """
        Generate an empathetic response.
        """
        # 1. Detect emotion
        emotion = self.emotion_detector.get_top_emotion(user_input)

        # 2. Get context
        context = self.context_manager.get_context()

        # 3. Build empathetic prompt
        prompt = f"""
You are a warm, empathetic friend.
User emotion: {emotion}
Previous context:
{context}
Current message: {user_input}

Respond with:
- Validation of feelings
- Warm, caring tone
- Optional support or question
Response:
"""

        # 4. Generate
        output = self.generator(
            prompt,
            max_length=150,
            truncation=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0]["generated_text"]

        # 5. Extract response
        response = output.split("Response:")[-1].strip()

        # 6. Update context
        self.context_manager.add_turn(user_input, response)

        return response


# === DEMO ===
if __name__ == "__main__":
    bot = EmpatheticResponder()
    print(bot.respond("I just broke up with my partner"))
    print("\n--- Next ---\n")
    print(bot.respond("I don't know how to move on"))
