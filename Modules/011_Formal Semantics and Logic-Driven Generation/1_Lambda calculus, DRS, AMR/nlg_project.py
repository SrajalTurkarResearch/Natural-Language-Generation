# nlg_project.py
# Implements an NLG system combining Lambda, DRS, and AMR for physics reports
# Designed for aspiring scientists, with clear explanations and visualizations
# Requires: networkx, matplotlib
# Run: python nlg_project.py

import networkx as nx
import matplotlib.pyplot as plt


# --- DRS Class (from drs.py) ---
class DRS:
    def __init__(self, universe, conditions):
        self.universe = universe
        self.conditions = conditions

    def __str__(self):
        return f"Universe: {self.universe}\nConditions: {self.conditions}"


# --- NLG System ---
# Goal: Generate a physics report from data using Lambda, DRS, AMR
# Data: {mass: 2kg, velocity: 10m/s, acceleration: 5m/s²}
data = {"mass": 2, "velocity": 10, "acceleration": 5}

# Step 1: Lambda Calculus - Compute force
force = lambda m: lambda a: m * a
force_value = force(data["mass"])(data["acceleration"])
print(f"Lambda Calculus: Force = {force_value}N")

# Step 2: DRS - Represent context
drs = DRS(
    universe=["x", "e"],
    conditions=[
        f"object(x)",
        f'mass(x,{data["mass"]}kg)',
        f"move(e,x)",
        f'velocity(e,{data["velocity"]}m/s)',
        f'acceleration(e,{data["acceleration"]}m/s²)',
    ],
)
print("\nDRS Representation:")
print(drs)

# Step 3: AMR - Graph the meaning
G = nx.DiGraph()
G.add_nodes_from(["move-01", "object", "mass", "velocity", "acceleration"])
G.add_edges_from(
    [
        ("move-01", "object", {"label": ":ARG1"}),
        ("object", "mass", {"label": ":mass"}),
        ("move-01", "velocity", {"label": ":velocity"}),
        ("move-01", "acceleration", {"label": ":acceleration"}),
    ]
)

print("\nGenerating AMR Graph...")
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.title("AMR: Physics Motion")
plt.show()

# Step 4: Generate NLG Text
text = f"An object of {data['mass']}kg moves at {data['velocity']}m/s with {data['acceleration']}m/s² acceleration, exerting {force_value}N force."
print("\nGenerated Text:")
print(text)

# --- For Scientists ---
# This system automates physics reports, like Einstein explaining motion.
# Extend to real datasets (e.g., Kaggle physics data) for research.
# 2025 Update: Add neural lambda for smarter computations.
