# Case Studies: Natural Language to Code (NL → Code) in Scientific Research

Hey, future scientist! Welcome to this detailed case study collection, showing how NL → Code (like Codex and AlphaCode) powers real-world research. These cases connect to the NL → Code tutorial and `.py` files you’ve got, helping you see how tools like those in `codex_mini_project.py` or `iris_analysis_project.py` apply to science. Each case includes a problem, a prompt, the generated code, its impact, and research insights to spark your curiosity. Think of these as lab reports from other scientists—use them to inspire your own experiments! Copy key points into your lab notebook and ask: “How can I adapt this to my field (e.g., chemistry, astronomy)?”

---

## Case Study 1: Bioinformatics - Analyzing DNA Sequences

### Problem

A biologist needs to analyze a DNA sequence to count nucleotides (A, C, G, T) and identify patterns, crucial for understanding genetic mutations or designing drugs.

### Prompt

“Write Python code to count the frequency of each nucleotide (A, C, G, T) in a DNA sequence and display the results.”

### Generated Code

```python
def count_nucleotides(dna):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for base in dna:
        counts[base] += 1
    return counts

# Example DNA sequence
dna = "AGCTTAGCCATG"
result = count_nucleotides(dna)
print("Nucleotide Counts:", result)

# Output: {'A': 3, 'C': 3, 'G': 2, 'T': 4}
```

### How It Works

- The function loops through the DNA string, counting each nucleotide.
- It’s simple but scalable, like the rule-based approach in `codex_mini_project.py`.
- **Analogy** : Like counting ingredients in a recipe to ensure the right balance.

### Impact

- **Research** : Speeds up genetic analysis by automating repetitive tasks, letting biologists focus on interpreting results.
- **Real-World** : Helps identify mutations for drug discovery, e.g., targeting cancer genes.
- **Connection to Tutorial** : Ties to Section 8 (Codex) and `codex_mini_project.py`, showing how simple prompts generate functional code.

### Research Insights

- **Challenge** : Models like Codex struggle with rare genomic formats (e.g., non-standard bases like U in RNA).
- **Opportunity** : Train models on diverse genomic datasets or develop new tokenization methods for biology.
- **Your Role** : Test NL → Code on your own DNA data and publish accuracy results.

  **Notebook Tip** : Write: “NL → Code automates DNA analysis.” Ask: “Could I use this for my biology project? What if I added visualization?”

---

## Case Study 2: Physics - Simulating Pendulum Motion

### Problem

A physicist wants to simulate a pendulum’s motion to study its dynamics or teach students about harmonic motion, requiring code to model and visualize the system.

### Prompt

“Write Python code to simulate a simple pendulum’s motion over 10 seconds and plot the angle versus time.”

### Generated Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Pendulum parameters
theta = 0.1  # Initial angle (radians)
omega = 0    # Initial angular velocity
g = 9.81     # Gravity (m/s^2)
L = 1        # Pendulum length (m)
t = np.linspace(0, 10, 1000)  # Time array
dt = t[1] - t[0]  # Time step
angles = []

# Numerical simulation (Euler method)
for _ in t:
    omega += (-g/L * np.sin(theta)) * dt  # Update velocity
    theta += omega * dt                   # Update angle
    angles.append(theta)

# Plot
plt.plot(t, angles)
plt.title('Pendulum Motion')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid(True)
plt.show()

# Output: A plot showing oscillatory motion
```

### How It Works

- Uses the Euler method to approximate pendulum motion (dθ/dt = ω, dω/dt = -g/L\*sin(θ)).
- Plots angle over time, similar to `iris_analysis_project.py`’s visualization.
- **Analogy** : Like tracking a swinging pendulum in your lab and graphing its path.

### Impact

- **Research** : Helps physicists test theories or visualize complex systems (e.g., coupled pendulums).
- **Real-World** : Used in education or engineering (e.g., designing stable structures).
- **Connection to Tutorial** : Ties to Section 15 (Applications) and `iris_analysis_project.py`, showing NL → Code’s power for simulations.

### Research Insights

- **Challenge** : Current models don’t handle multi-modal inputs well (e.g., generating code from a pendulum sketch).
- **Opportunity** : Develop models that combine images and text for physics simulations.
- **Your Role** : Experiment with prompts for other systems (e.g., springs) and compare accuracy.

  **Notebook Tip** : Write: “NL → Code simulates physics models.” Ask: “How can I simulate my experiment (e.g., orbits)?”

---

## Case Study 3: Climate Science - Temperature Trend Analysis

### Problem

A climate scientist needs to analyze historical temperature data to identify trends, requiring code to compute yearly averages and visualize changes.

### Prompt

“Write Python code to calculate average temperature by year from a dataset and plot the results.”

### Generated Code

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample temperature data (replace with real dataset)
data = pd.DataFrame({
    'year': [2020, 2020, 2021, 2021, 2022, 2022],
    'temp': [20.5, 21.0, 22.3, 22.8, 23.1, 23.5]
})

# Calculate average temperature by year
avg_temp = data.groupby('year')['temp'].mean()

# Print results
print("Average Temperature by Year:")
print(avg_temp)

# Plot
plt.plot(avg_temp.index, avg_temp.values, marker='o')
plt.title('Average Temperature by Year')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# Output: Series with averages and a line plot
```

### How It Works

- Groups data by year, computes mean temperature, and plots a trend line.
- Similar to `iris_analysis_project.py` but for time-series data.
- **Analogy** : Like summarizing weather notes to spot patterns in your lab log.

### Impact

- **Research** : Accelerates climate model validation by automating data analysis.
- **Real-World** : Helps predict warming trends for policy decisions.
- **Connection to Tutorial** : Ties to Section 11 (Metrics) and `iris_analysis_project.py`, showing data-driven NL → Code.

### Research Insights

- **Challenge** : Models struggle with noisy or incomplete datasets (e.g., missing years).
- **Opportunity** : Research robust models for messy data or new metrics for accuracy.
- **Your Role** : Test NL → Code on your climate data and publish on robustness.

  **Notebook Tip** : Write: “NL → Code analyzes climate data.” Ask: “What dataset could I analyze (e.g., rainfall)?”

---

### How to Use These Case Studies

- **Run the Code** : Copy each code block into a `.py` file or Jupyter cell. Install `numpy`, `pandas`, and `matplotlib` (`pip install numpy pandas matplotlib`).
- **Connect to `.py` Files** : Case 1 is like `codex_mini_project.py` (simple function). Cases 2 and 3 resemble `iris_analysis_project.py` (data analysis with plots).
- **Lab Notebook** : Summarize each case: Problem, Code, Impact. Ask: “How can I adapt this to my field?”
- **Research Ideas** : Use insights to design experiments, e.g., test NL → Code on rare data formats or multi-modal prompts.
- **Visual Idea** : Each case produces a plot (Case 2: oscillatory curve, Case 3: trend line). Sketch these in your notebook to understand outputs.

These cases show NL → Code’s power for science. Like Ada Lovelace envisioning computers beyond math, dream up new ways to apply this in your research!
