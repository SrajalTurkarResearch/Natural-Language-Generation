# memory_chatbot.py
"""
A simple NLG chatbot with memory to store user preferences and conversation history.
Purpose: Demonstrate short-term and long-term memory in NLG for beginner scientists.
Usage: Run standalone or import into a Jupyter Notebook for interactive testing.
"""


class MemoryChatbot:
    def __init__(self):
        """Initialize an empty memory dictionary to store user data."""
        self.memory = {}

    def update_memory(self, user_id, user_input):
        """Store user input in memory and update preferences if applicable."""
        if user_id not in self.memory:
            self.memory[user_id] = {"history": [], "preferences": {}}
        self.memory[user_id]["history"].append(user_input)

        # Limit short-term memory to last 5 messages to simulate token window
        if len(self.memory[user_id]["history"]) > 5:
            self.memory[user_id]["history"] = self.memory[user_id]["history"][-5:]

        # Check for preference updates (e.g., favorite food)
        if "my favorite" in user_input.lower():
            parts = user_input.lower().split("my favorite")[1].split("is")
            if len(parts) > 1:
                key, value = parts[0].strip(), parts[1].strip()
                self.memory[user_id]["preferences"][key] = value
                return f"Noted! Your favorite {key} is {value}."
        return None

    def generate_response(self, user_input, user_id="user1"):
        """Generate a response based on memory and context."""
        # Update memory with new input
        preference_response = self.update_memory(user_id, user_input)
        if preference_response:
            return preference_response

        # Generate context-aware response
        if "suggest" in user_input.lower():
            for key in self.memory[user_id]["preferences"]:
                if key in user_input.lower():
                    return f"How about something related to your favorite {key}: {self.memory[user_id]['preferences'][key]}?"
            return "I can suggest something if you tell me your favorites!"

        # Check conversation history for context
        if "more" in user_input.lower() and self.memory[user_id]["history"]:
            last_topic = (
                self.memory[user_id]["history"][-2]
                if len(self.memory[user_id]["history"]) > 1
                else ""
            )
            if "weather" in last_topic.lower():
                return "More weather info: Expect sunny skies tomorrow!"

        return "Tell me more or share a favorite (e.g., 'My favorite food is sushi')!"


# Example usage
if __name__ == "__main__":
    chatbot = MemoryChatbot()
    print(chatbot.generate_response("My favorite food is sushi.", "user1"))
    print(chatbot.generate_response("Suggest a recipe.", "user1"))
    print(chatbot.generate_response("Tell me more about the weather.", "user1"))
