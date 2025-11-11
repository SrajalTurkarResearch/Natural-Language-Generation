# Visualizations for NLG Concepts
# This script includes plots for attention mechanisms and a text-based diagram.

import matplotlib.pyplot as plt
import numpy as np

# Sample Attention Matrix Visualization
attention = np.random.rand(5, 5)  # Generate random attention weights
plt.imshow(attention, cmap="hot")  # Display as heatmap
plt.title("Sample Attention Matrix")
plt.colorbar()  # Add color scale
plt.show()

# Text-Based Feedback Loop Diagram
print("User Input → NLG Generation → Output → Feedback → Refine → Repeat")
