# react_example.py
# Purpose: Simulate ReAct (Reasoning + Acting) for a factual query.
# Context: Shows how ReAct combines internal reasoning with external actions (e.g., tool calls) in NLG.
# Problem: Find the boiling point of water.
# For scientists: This mirrors iterative experimentation—hypothesize, test, refine.


def react_example(query):
    """
    Simulates ReAct loop: Reason, Act, Observe, Conclude.
    Returns a string with the reasoning process and answer.
    """
    state = {"query": query, "steps": []}
    # Thought: Identify what information is needed
    state["steps"].append("Thought: Need boiling point of water.")
    # Action: Simulate querying a tool (e.g., database or API)
    action_result = "100°C"  # In real scenarios, replace with actual API call
    state["steps"].append(f"Action: Query database → {action_result}")
    # Observe and conclude
    state["steps"].append("Thought: At standard pressure, it is 100°C.")
    return "\n".join(state["steps"]) + "\nAnswer: 100°C"


if __name__ == "__main__":
    # Example usage
    print("Running ReAct Example for Boiling Point Query")
    print(react_example("Boiling point of water"))

# Explanation for Aspiring Scientists:
# - ReAct is like conducting an experiment: you think, check data (action), and interpret results.
# - In NLG, this ensures factual accuracy in generated text, critical for scientific reports.
# - Extend this by adding real API calls (e.g., requests.get) to fetch live data.
