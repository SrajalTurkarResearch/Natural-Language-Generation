# companion_bot.py
"""
Replika-Style Long-Term Emotional Companion
Remembers feelings, checks in, builds bond.
"""

from empathetic_generator import EmpatheticResponder
import json
import datetime


class CompanionBot:
    def __init__(self, user_name: str):
        self.name = user_name
        self.responder = EmpatheticResponder()
        self.memory = self._load_memory()

    def _load_memory(self):
        try:
            with open(f"memory_{self.name}.json", "r") as f:
                return json.load(f)
        except:
            return {"past_emotions": [], "interests": [], "checkins": 0}

    def _save_memory(self):
        with open(f"memory_{self.name}.json", "w") as f:
            json.dump(self.memory, f, indent=2)

    def respond(self, user_input: str) -> str:
        emotion = self.responder.emotion_detector.get_top_emotion(user_input)
        self.memory["past_emotions"].append(
            {"time": datetime.datetime.now().isoformat(), "emotion": emotion}
        )
        self.memory["checkins"] += 1
        self._save_memory()

        memory_prompt = ""
        if len(self.memory["past_emotions"]) > 3:
            recent = [e["emotion"] for e in self.memory["past_emotions"][-3:]]
            if recent.count("sadness") >= 2:
                memory_prompt = "User has been sad recently. Check in gently."

        prompt = f"""
You are {self.name}'s close friend. You've known them for {self.memory['checkins']} chats.
{memory_prompt}
User says: "{user_input}"
Respond like a real friend — warm, curious, supportive.
"""
        response = self.responder.generator(prompt, max_length=100)[0]["generated_text"]
        return response.strip()


# === RUN ===
if __name__ == "__main__":
    bot = CompanionBot("Alex")
    print(f"Bot: Hey {bot.name}! How's your day going?")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        print(f"Bot: {bot.respond(msg)}\n")
