# run_all_demos.py
"""
Launch any real-world bot with one command.
"""

import os

bots = {
    "1": ("Mental Health Bot", "python applications/mental_health_bot.py"),
    "2": ("Customer Service", "python applications/customer_service_bot.py"),
    "3": ("Language Tutor", "python applications/education_tutor_bot.py"),
    "4": ("Symptom Checker", "python applications/healthcare_symptom_bot.py"),
    "5": ("AI Companion", "python applications/companion_bot.py"),
}

print("Empathetic AI Suite - Choose a Bot:")
for k, (name, _) in bots.items():
    print(f"  {k}. {name}")

choice = input("\nEnter number: ")
if choice in bots:
    os.system(bots[choice][1])
else:
    print("Invalid choice.")
