# nlg_integration.py
# Module on Symbolic Planners in NLG and Neurosymbolic Integration
# Dependencies: None

"""
Module 4: Symbolic Planners in NLG and Neurosymbolic Applications
Learning Objectives:
- Apply planning to text generation.
- Integrate neural and symbolic for NLG.
Inspired by Einstein's thought experiments and Feynman's clear structures.
"""

# --- Theory: Symbolic Planners in NLG ---
"""
In NLG, planners treat sentences as actions to structure text logically.
Example Schema: Intro → Detail → Conclusion.
Math: Optimize utility U = relevance - cost(length).
Calculation: Plan1: relevance=8, length=3, U=8-3=5. Plan2: relevance=9, length=4, U=5. Choose based on tie-breaker.

Real-World: Teriyaki framework (from searches) plans robot instructions from NL.
"""


# Simple NLG Planner
def nlg_planner(data, goal="inform"):
    plan = []
    if goal == "inform":
        plan.append({"action": "intro", "text": f"Today in {data['city']}, "})
        if data["temp"] > 20:
            plan.append(
                {
                    "action": "detail",
                    "text": f"expect warm weather at {data['temp']}°C.",
                }
            )
        else:
            plan.append({"action": "detail", "text": f"it's cool at {data['temp']}°C."})
    return [step["text"] for step in plan]


# Test
data = {"city": "Paris", "temp": 25}
print("NLG Plan:", " ".join(nlg_planner(data)))

# --- Theory: Neurosymbolic NLG ---
"""
Neural parses input (e.g., 'What's the weather?'), symbolic plans response.
From searches: NSP framework for navigation tasks.
Analogy: Neural as a creative writer, symbolic as an editor ensuring logic.
"""

# --- Exercise ---
"""
Task: Extend nlg_planner to include a 'conclusion' action (e.g., 'Plan your day!').
Solution:
def extended_nlg_planner(data, goal='inform'):
    plan = nlg_planner(data, goal)
    plan.append('Plan your day accordingly!')
    return plan
print(' '.join(extended_nlg_planner(data)))
"""

# --- Reflection Prompt ---
"""
How could neurosymbolic NLG improve medical report accuracy? Design a test case.
"""
