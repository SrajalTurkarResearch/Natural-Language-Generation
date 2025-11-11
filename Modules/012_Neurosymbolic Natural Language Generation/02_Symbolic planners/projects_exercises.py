# projects_exercises.py
# Module for Projects and Exercises
# Dependencies: torch, numpy, matplotlib, networkx

"""
Module 6: Mini/Major Projects and Exercises
Learning Objectives:
- Build practical neurosymbolic NLG systems.
- Practice scientific experimentation.
Inspired by Curie's hands-on approach and Einstein's innovative questions.
"""

import torch
import torch.nn as nn
from collections import deque
import heapq


# --- Mini Project: Weather NLG Planner ---
def weather_nlg(data):
    plan = []
    if data["temp"] > 20:
        plan.append(f"Warm day in {data['city']} at {data['temp']}°C.")
    else:
        plan.append(f"Cool day in {data['city']} at {data['temp']}°C.")
    plan.append("Plan your activities!")
    return " ".join(plan)


# Test
print("Mini Project Output:", weather_nlg({"city": "Paris", "temp": 25}))


# --- Major Project: Neurosymbolic Chatbot ---
class ChatbotNeural(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)  # Dummy intent classifier

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)


def chatbot_planner(intent, data):
    if intent == "weather":
        return weather_nlg(data)
    return "Unknown intent."


# Test
model = ChatbotNeural()
data = torch.tensor([[1.0] * 10])
intent_probs = model(data).detach().numpy()
intent = "weather" if intent_probs[0][0] > 0.5 else "unknown"
print("Chatbot Output:", chatbot_planner(intent, {"city": "Paris", "temp": 25}))


# --- Exercise: A* Planner ---
def a_star_planner(initial, goal, actions, heuristic):
    queue = [(0 + heuristic(initial), 0, initial, [])]
    visited = set()
    while queue:
        f, g, state, path = heapq.heappop(queue)
        state_tuple = tuple(sorted(state.items()))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        if state == goal:
            return path
        for action in actions:
            if action["precondition"](state):
                new_state = action["effect"](state.copy())
                new_g = g + 1
                heapq.heappush(
                    queue,
                    (
                        new_g + heuristic(new_state),
                        new_g,
                        new_state,
                        path + [action["name"]],
                    ),
                )
    return None


# Test A*
heuristic = lambda s: sum(1 for k, v in s.items() if goal.get(k) != v)
actions = [
    {
        "name": "stack A on B",
        "precondition": lambda s: s["A"] == "table" and s["B"] == "table",
        "effect": lambda s: {**s, "A": "B"},
    }
]
plan = a_star_planner(
    {"A": "table", "B": "table"}, {"A": "B", "B": "table"}, actions, heuristic
)
print("A* Plan:", plan)

# --- Reflection Prompt ---
"""
Design a project to use neurosymbolic NLG for educational tools. Hypothesize improvements.
"""
