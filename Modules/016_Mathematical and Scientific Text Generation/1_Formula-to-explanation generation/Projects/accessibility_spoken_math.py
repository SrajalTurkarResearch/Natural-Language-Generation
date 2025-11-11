# accessibility_spoken_math.py
"""
Real-World Project: Spoken Math for Blind Users
Use Case: Convert LaTeX formulas to clear, spoken English.
"""

from transformers import pipeline
import pyttsx3  # pip install pyttsx3

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# NLG model for spoken-style explanation
explainer = pipeline("text2text-generation", model="t5-base")


def latex_to_spoken(latex):
    prompt = f"Convert this LaTeX math to spoken English: {latex}"
    spoken = explainer(prompt, max_length=100)[0]["generated_text"]
    return spoken


def speak(text):
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()


# Test formulas
formulas = [
    "$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$",
    "$E = mc^2$",
    "$\\int_a^b f(x) \\, dx$",
]

if __name__ == "__main__":
    print("Spoken Math Accessibility Tool\n")
    for f in formulas:
        spoken = latex_to_spoken(f)
        speak(spoken)
