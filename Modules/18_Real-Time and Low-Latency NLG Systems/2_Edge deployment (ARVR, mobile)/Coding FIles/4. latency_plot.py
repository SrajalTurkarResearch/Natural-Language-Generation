# latency_plot.py
# Purpose: Plot latency comparison for edge deployment rationale.
# Theory: Latency critical in AR/VR; edge saves round-trip time. Example calc: T_transmit = data_size / bandwidth (e.g., 1MB / 100Mbps = 80ms).
# Logic: Use matplotlib bar for simple viz; data from theoretical models.
# As researcher: Collect real data from device benchmarks.

import matplotlib.pyplot as plt

# Data: ms latencies (edge: process only; cloud: transmit + process + return)
categories = ["Edge Computing", "Cloud Computing"]
latencies = [30, 130]  # Example values; adjust based on experiments

# Create bar plot
plt.bar(categories, latencies, color=["blue", "orange"])
plt.ylabel("Latency (milliseconds)")
plt.title("Edge vs. Cloud Latency Comparison for NLG")
plt.ylim(0, max(latencies) * 1.2)  # Add headroom
plt.show()

# Save for documentation
plt.savefig("latency_comparison.png")
print("Plot saved as latency_comparison.png")

# Advanced: Add error bars for variability (e.g., std dev from multiple runs)
