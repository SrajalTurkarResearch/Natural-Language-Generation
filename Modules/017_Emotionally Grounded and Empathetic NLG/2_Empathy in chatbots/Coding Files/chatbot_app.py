# chatbot_app.py
"""
Complete Empathetic Chatbot Application
Run this to start an interactive session.
"""

from empathetic_generator import EmpatheticResponder


def main():
    print("Empathetic Chatbot v1.0")
    print("Type 'quit' to exit.\n")

    bot = EmpatheticResponder()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Take care! I'm here whenever you need.")
            break
        if not user_input:
            continue

        response = bot.respond(user_input)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
