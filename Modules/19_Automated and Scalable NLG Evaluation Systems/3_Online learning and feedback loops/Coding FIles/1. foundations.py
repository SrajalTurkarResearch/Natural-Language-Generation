"""
foundations.py
==============
Module 1: Core Concepts of NLP and NLG
For aspiring scientists — start here!
"""


def explain_nlp():
    """Natural Language Processing (Understanding Human Language)"""
    print("🌟 NLP = Natural Language Processing")
    print("   Job: Understand human text/speech")
    print("   Example: 'What's the time?' → AI knows you want clock time\n")


def explain_nlg():
    """Natural Language Generation (Creating Human-Like Text)"""
    print("🌟 NLG = Natural Language Generation")
    print("   Job: Write coherent, natural text from data")
    print("   Example: Data → 'It's 25°C and sunny today. Perfect for a walk!'\n")


def nlg_pipeline():
    """The 3-Step NLG Process"""
    steps = [
        ("1. Content Planning", "Decide WHAT to say"),
        ("2. Sentence Planning", "Decide HOW to organize"),
        ("3. Surface Realization", "Write grammatical sentences"),
    ]
    print("🛠️ NLG Pipeline:")
    for i, (name, desc) in enumerate(steps, 1):
        print(f"   {name}: {desc}")
    print()


if __name__ == "__main__":
    print("=== WELCOME TO NLG SCIENCE ===\n")
    explain_nlp()
    explain_nlg()
    nlg_pipeline()
