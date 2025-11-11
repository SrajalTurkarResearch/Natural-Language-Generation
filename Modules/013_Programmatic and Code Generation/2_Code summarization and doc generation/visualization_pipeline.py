# visualization_pipeline.py
# Purpose: Draws a flowchart showing how NLG turns code into words.
# Why: Scientists use visuals to understand processes, like mapping experiment steps.
# Requires: matplotlib (run setup.py first).

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# Create figure
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 5)
ax.set_ylim(0, 2)

# Draw boxes for NLG steps
boxes = [
    ("Code", 0.5, 1),
    ("Parse", 1.5, 1),
    ("Encoder", 2.5, 1),
    ("Decoder", 3.5, 1),
    ("Text", 4.5, 1),
]
for label, x, y in boxes:
    ax.add_patch(Rectangle((x - 0.4, y - 0.2), 0.8, 0.4, fill=False))
    ax.text(x, y, label, ha="center", va="center")

# Draw arrows between steps
for i in range(len(boxes) - 1):
    ax.add_patch(
        FancyArrowPatch(
            (boxes[i][1] + 0.4, 1), (boxes[i + 1][1] - 0.4, 1), arrowstyle="->"
        )
    )

# Clean up and show
ax.axis("off")
plt.title("NLG Pipeline for Code Summarization")
plt.show()

# Why this matters: Shows the flow from code to words, like a lab process diagram.
# For science: Helps explain complex AI processes in papers or talks.
# Try it: Add a box (e.g., 'Refine') and redraw.
