# major_project.py
# Hybrid NLG System: Seamless Chit-Chat + Goal-Oriented Generation
# Author: Grok, inspired by Alan Turing, Albert Einstein, and Nikola Tesla
# Date: August 22, 2025
# Purpose: Demonstrates a hybrid conversational agent that blends open-domain chit-chat with task-oriented responses.

import random


def hybrid_nlg_bot(input_text):
    """
    Hybrid NLG Bot: Switches between chit-chat and goal-oriented responses.
    If a booking intent is detected, responds with a confirmation and a chit-chat follow-up.
    Otherwise, engages in open-domain conversation.
    """
    chit_chat_responses = [
        "That's fascinating! Tell me more.",
        "I'm all ears—what else is on your mind?",
        "Sounds interesting! Let's keep chatting.",
    ]
    if "book" in input_text.lower():
        # Simulate task completion + social follow-up
        return "Your booking is confirmed. By the way, do you enjoy traveling?"
    return random.choice(chit_chat_responses)


# Demo/Test
if __name__ == "__main__":
    print("Hybrid NLG Bot Response:", hybrid_nlg_bot("book flight"))
    print("Hybrid NLG Bot Response:", hybrid_nlg_bot("What's your favorite city?"))
