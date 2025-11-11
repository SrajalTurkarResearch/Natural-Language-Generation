# nlg_basics.py
# Comprehensive module on NLG and Symbolic AI for aspiring scientists
# Dependencies: None (uses standard Python)

"""
Module 1: Fundamentals of Natural Language Generation (NLG) and Symbolic AI
Learning Objectives:
- Understand NLG pipeline and its purpose.
- Learn symbolic AI principles with logic examples.
- Apply concepts via simple exercises.
Inspired by Feynman's clear explanations and Turing's logical rigor.
"""

# --- Theory: Natural Language Generation (NLG) ---
"""
NLG converts data into human-readable text, like turning numbers into sentences.
Pipeline:
1. Content Determination: Choose key facts (e.g., select weather data).
2. Text Planning: Organize structure (e.g., intro, details, conclusion).
3. Surface Realization: Pick words/grammar (e.g., make it sound natural).
Additional tasks: Aggregation (combine facts), Referring Expressions (use pronouns).

Feynman Analogy: Like writing a story. Data is the plot points, planning is the outline,
realization is the polished writing.

Why It Matters: Ensures computers communicate clearly, used in chatbots, reports.
Real-World: Automated news (e.g., Associated Press earnings reports from financial data).
"""


# Example: Simple NLG simulation
def simple_nlg(data):
    # Content determination: Select key info
    content = {"city": data["city"], "temp": data["temp"]}
    # Text planning: Structure as intro + detail
    plan = ["intro", "temp_detail"]
    # Surface realization: Generate text
    text = f"In {content['city']}, "
    if content["temp"] > 20:
        text += f"it's warm with a temperature of {content['temp']}°C."
    else:
        text += f"it's cool at {content['temp']}°C."
    return text


# Test NLG
data = {"city": "Paris", "temp": 25}
print("NLG Output:", simple_nlg(data))

# --- Theory: Symbolic AI ---
"""
Symbolic AI uses rules and symbols (e.g., words, variables) to reason.
Key Components:
- Knowledge Base: Stores facts/rules (e.g., "If rain, then wet").
- Inference Engine: Applies rules to decide.

Turing's Logic: Like a machine manipulating symbols on a tape.
Pros: Transparent, traceable. Cons: Rigid, needs predefined rules.

Math Example (Newton-style):
Propositional logic: If P (rain) then Q (wet). P=1 (true), so Q=1.
Calculation: P ∧ (P→Q) => Q. (1 ∧ (1→1)) = 1 ∧ 1 = 1.
"""


# Symbolic AI Example: Rule-based weather response
def symbolic_weather(input_text):
    rules = {"weather": "Today is sunny with a high of 25°C."}
    return rules.get(input_text.lower(), "I don't understand.")


# Test symbolic AI
print("Symbolic AI Response:", symbolic_weather("weather"))

# --- Thought Experiment (Einstein) ---
"""
Imagine NLG as a light beam: Data particles are scattered, planning aligns them,
realization shapes them into a clear message. What if you changed the data 'color'?
"""

# --- Exercise ---
"""
Task: Write a rule-based function to respond to 'time' with 'It's 10 AM.'
Solution:
def exercise_time(input_text):
    rules = {'time': "It's 10 AM."}
    return rules.get(input_text.lower(), "Unknown query.")
print(exercise_time('time'))
"""

# --- Reflection Prompt ---
"""
As a scientist, how could you test if NLG outputs are more readable than raw data?
Design a small user study.
"""
