"""
PROJECT: Empathetic Government Service Chatbot
USE CASE: Public Sector Digital Assistant
GOAL: Provide clear, kind, and accessible responses to citizen queries.

Author: [Your Name] – Public AI Scientist
Date: October 29, 2025
"""

import re
import json
from datetime import datetime
from transformers import pipeline

# Load model
generator = pipeline("text-generation", model="gpt2", max_length=150, truncation=True)

# Knowledge base: Common citizen queries
FAQ = {
    "benefits": "You may be eligible for unemployment benefits if you've lost your job through no fault of your own. Apply online at gov.benefits.gov within 7 days.",
    "tax": "File your taxes by April 15. Use the online portal or mail Form 1040. First-time filers get a free guide.",
    "license": "Renew your driver’s license online or at any DMV office. Bring ID and proof of residency.",
    "voting": "Register to vote online at vote.gov. You need ID and proof of address. Deadline is 15 days before election.",
}


def detect_intent(user_input):
    """Simple regex-based intent detection."""
    text = user_input.lower()
    if any(k in text for k in ["benefit", "unemploy", "money", "aid"]):
        return "benefits"
    elif any(k in text for k in ["tax", "irs", "refund"]):
        return "tax"
    elif any(k in text for k in ["license", "driving", "dmv"]):
        return "license"
    elif any(k in text for k in ["vote", "election", "register"]):
        return "voting"
    else:
        return "general"


def generate_empathetic_response(intent, user_input):
    """Generate kind, clear, and actionable response."""
    base_info = FAQ.get(
        intent,
        "I can help you find the right department. Please hold or visit gov.info.",
    )

    prompt = f"""
    You are a kind, patient government service assistant.
    Citizen asked: "{user_input}"
    Key info: {base_info}
    Respond clearly, empathetically, and list next steps. Use simple language.
    """

    response = generator(prompt, temperature=0.6, num_return_sequences=1)[0][
        "generated_text"
    ]
    return response.strip()


def log_interaction(user, intent, response):
    """Log for audit and improvement."""
    log = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user,
        "detected_intent": intent,
        "response": response,
    }
    with open("gov_chat_log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")


# ——— MAIN ———
def gov_service_chatbot():
    print("Government Service Assistant (Type 'quit' to exit)\n")

    while True:
        user_input = input("Citizen: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Assistant: Thank you for using our service. Have a great day!")
            break

        intent = detect_intent(user_input)
        response = generate_empathetic_response(intent, user_input)

        print(f"Assistant: {response}\n")
        log_interaction(user_input, intent, response)


if __name__ == "__main__":
    gov_service_chatbot()
