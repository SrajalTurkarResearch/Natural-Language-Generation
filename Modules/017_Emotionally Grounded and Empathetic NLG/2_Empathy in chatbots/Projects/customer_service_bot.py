# customer_service_bot.py
"""
KLM BlueBot-Style Customer Service Agent
Handles complaints, delays, lost luggage with empathy + action.
"""

from empathetic_generator import EmpatheticResponder
import random


class CustomerServiceBot:
    def __init__(self, airline="KLM"):
        self.responder = EmpatheticResponder()
        self.airline = airline
        self.complaint_templates = {
            "delay": "I’m so sorry your flight was delayed. That’s frustrating. Let me check your options.",
            "luggage": "I completely understand how upsetting lost luggage is. Let’s track it now.",
            "cancel": "I’m truly sorry your flight was canceled. I’ll find you the best rebooking.",
        }

    def detect_issue(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["delay", "late", "missed"]):
            return "delay"
        elif any(w in text for w in ["bag", "luggage", "lost"]):
            return "luggage"
        elif any(w in text for w in ["cancel", "canceled"]):
            return "cancel"
        return "general"

    def respond(self, user_input: str, ticket_id: str = None) -> str:
        issue = self.detect_issue(user_input)
        emotion = self.responder.emotion_detector.get_top_emotion(user_input)

        if emotion in ["anger", "sadness"]:
            base = self.complaint_templates.get(
                issue, "I’m sorry you’re going through this."
            )
            action = self._generate_action(issue, ticket_id)
            return f"{base}\n{action}"
        else:
            return f"Thank you for reaching out. How can I assist you today?"

    def _generate_action(self, issue: str, ticket_id: str) -> str:
        actions = {
            "delay": f"I'll check compensation and rebooking. Your ticket: {ticket_id or 'KL1234'}.",
            "luggage": "I’ve escalated to baggage team. Tracking link: klm.com/trackbag",
            "cancel": "You’re eligible for full refund or rebooking. Processing now.",
            "general": "Let me connect you to an agent.",
        }
        return actions.get(issue, actions["general"])


# === RUN ===
if __name__ == "__main__":
    bot = CustomerServiceBot()
    print("KLM Support: Hello! How can I help?")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        print(f"Bot: {bot.respond(msg, 'KL5678')}\n")
