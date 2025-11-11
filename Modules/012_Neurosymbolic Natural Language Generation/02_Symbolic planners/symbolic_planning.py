# symbolic_planning.py
# Module on Symbolic Planning with Algorithms
# Dependencies: networkx, matplotlib

"""
Module 3: Symbolic Planning in AI
Learning Objectives:
- Implement a symbolic planner with BFS and A*.
- Understand PDDL-like structures.
- Visualize planning graphs.
Inspired by Turing's computational clarity and Curie's experimental tests.
"""

from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# --- Theory: Symbolic Planning ---
"""
Planning finds action sequences from start to goal state.
Components:
- State: World snapshot (e.g., {'A': 'table'}).
- Action: Operation with preconditions (must be true) and effects (changes).
- Goal: Desired state.

Algorithms:
- BFS: Checks all paths level by level.
- A*: Uses f = g (cost so far) + h (heuristic to goal).

PDDL Example (simplified):
Domain: Define actions like 'stack A on B'.
Problem: Set initial and goal states.

Feynman Analogy: Like planning a road trip—start at home, actions are roads, goal is destination.
"""


# BFS Planner
def bfs_planner(initial_state, goal_state, actions):
    queue = deque([(initial_state, [])])
    visited = set()
    while queue:
        state, path = queue.popleft()
        state_tuple = tuple(sorted(state.items()))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        if state == goal_state:
            return path
        for action in actions:
            if action["precondition"](state):
                new_state = action["effect"](state.copy())
                queue.append((new_state, path + [action["name"]]))
    return None


# Example: Blocksworld
initial = {"A": "table", "B": "table"}
goal = {"A": "B", "B": "table"}
actions = [
    {
        "name": "stack A on B",
        "precondition": lambda s: s["A"] == "table" and s["B"] == "table",
        "effect": lambda s: {**s, "A": "B"},
    }
]

plan = bfs_planner(initial, goal, actions)
print("BFS Plan:", plan)

# --- Visualization: Planning Graph ---
G = nx.DiGraph()
G.add_edges_from(
    [("Initial: A,B on table", "Stack A on B"), ("Stack A on B", "Goal: A on B")]
)
nx.draw(G, with_labels=True)
plt.title("Planning Graph")
plt.show()

# --- Exercise ---
"""
Task: Add a new action 'move A to table'. Test the planner.
Solution:
actions.append({
    'name': 'move A to table',
    'precondition': lambda s: s['A'] != 'table',
    'effect': lambda s: {**s, 'A': 'table'}
})
print(bfs_planner({'A': 'B', 'B': 'table'}, {'A': 'table', 'B': 'table'}, actions))
"""

# --- Reflection Prompt ---
"""
How could you optimize the planner for larger state spaces? Research heuristic design.
"""
