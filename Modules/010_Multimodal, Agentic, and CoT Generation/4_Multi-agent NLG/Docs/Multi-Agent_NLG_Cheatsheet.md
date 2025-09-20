# Multi-Agent NLG Cheatsheet for Aspiring Scientists

This cheatsheet summarizes key concepts, code snippets, visualizations, and tips from the Multi-Agent NLG tutorial. Designed for quick reference, it’s a concise guide for your scientific journey, inspired by Turing’s logic, Einstein’s clarity, and Tesla’s innovation. Use it alongside the detailed tutorial and case studies to reinforce learning.

---

## 1. Core Concepts

### What is NLG?

- **Definition** : AI process to convert data (e.g., numbers, lists) into human-readable text.
- **Analogy** : Like a chef turning ingredients (data) into a dish (text).
- **Key Stages** :
- Content Selection: Pick key facts.
- Text Planning: Organize the story.
- Sentence Aggregation: Combine ideas.
- Lexical Choice: Select words.
- Referring Expressions: Ensure clear naming.
- Surface Realization: Polish grammar and style.

### What is Multi-Agent NLG?

- **Definition** : Multiple agents (small programs) collaborate to generate text, each with a specific role.
- **Analogy** : A team of friends on a group project, each handling one task.
- **Why Use It?** : Splits complex tasks, improves quality, scales for big jobs.

### Agent Types

- **Reactive** : Quick responses (e.g., fix spelling).
- **Deliberative** : Plan ahead (e.g., structure a report).
- **Hybrid** : Combine quick and planned actions.
- **Learning** : Improve with feedback (e.g., adjust tone).

### Collaboration Methods

- **Message Passing** : Agents send notes (e.g., “Here’s data”).
- **Shared Notebook** : Central data storage for all agents.
- **Bidding** : Agents volunteer for tasks.
- **Negotiation** : Resolve disagreements (e.g., formal vs. casual tone).
- **Swarm** : Simple rules lead to smart outcomes.

### Architectures

- **Centralized** : One leader assigns tasks.
- **Decentralized** : Peer-to-peer communication.
- **Hybrid** : Leader plus group collaboration.
- **Layered** : Levels for details and strategy.
- **Emergent** : Self-organizing patterns.

---

## 2. Key Code Snippets

### Basic NLG System

```python
def data_agent(city):
    return {'city': city, 'temp': 25, 'condition': 'clear'}

def planning_agent(data):
    return ['intro', 'current']

def generation_agent(data, plan):
    return f"Weather in {data['city']}: {data['condition']} at {data['temp']}°C."

city = 'Paris'
data = data_agent(city)
plan = planning_agent(data)
print(generation_agent(data, plan))
```

### Learning Agent

```python
def learning_agent(feedback, tone):
    return max(0, min(1, tone + 0.1 * (feedback - tone)))

feedback = 0.8
tone = 0.5
print(f"New tone: {learning_agent(feedback, tone):.2f}")
```

### Visualization (Agent Architecture)

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from([('Data Agent', 'Planning Agent'), ('Planning Agent', 'Generation Agent')])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
plt.show()
```

---

## 3. Visualizations

### Agent Flow Diagram

```
[Data Agent] → [Planning Agent] → [Generation Agent] → [Refinement Agent]
(Draw arrows connecting boxes for each agent.)
```

### Learning Curve

```
Plot: Utility Score vs. Iterations
(Y-axis: 0.5 to 1.0, X-axis: 0 to 10, rising line)
```

---

## 4. Real-World Applications

- **Journalism** : Automated sports reports (e.g., Heliograf).
- **Healthcare** : Patient-friendly medical summaries.
- **Education** : Personalized lesson plans.
- **E-Commerce** : Dynamic chatbot responses.

---

## 5. Math Essentials

### Utility Function

[ U = 0.6 \cdot \text{fluency} + 0.4 \cdot \text{relevance} ]

- Example: Fluency = 0.9, Relevance = 0.8
  ( U = (0.6 \cdot 0.9) + (0.4 \cdot 0.8) = 0.54 + 0.32 = 0.86 )

### Negotiation Payoff Matrix

| Agent 1 \ Agent 2 | Formal | Casual |
| ----------------- | ------ | ------ |
| **Formal**        | (5, 5) | (2, 3) |
| **Casual**        | (3, 2) | (4, 4) |

- Optimal: Both choose Formal (5, 5).

### Probabilistic Selection

[ P(\text{Fact}) = \frac{P(\text{Data} | \text{Fact}) \cdot P(\text{Fact})}{P(\text{Data})} ]

- Example: ( P(\text{rain} | \text{humidity}=80%) = (0.9 \cdot 0.7) \div 0.8 = 0.7875 ).

---

## 6. Exercises & Projects

### Quick Exercise

- Add a bias-check agent to `basic_nlg_system.py`:
  ```python
  def bias_check_agent(text):
      return 'No bias' if 'fair' in text.lower() else 'Check bias'
  ```

### Mini Project

- Build a chatbot for weather queries using agents for data, response, and refinement.

### Major Project

- Create a news generator with a real dataset (e.g., Kaggle sports data).

---

## 7. Research Tips

- **Read** : ACL Anthology papers on NLG.
- **Experiment** : Use Hugging Face Transformers for advanced agents.
- **Innovate** : Explore ethical NLG or quantum-inspired systems.

---

## 8. What’s Missing in Other Tutorials

- **Ethics** : Bias detection and mitigation strategies.
- **Scalability** : Cloud-based agent deployment.
- **Interdisciplinary Links** : Game theory, neuroscience integration.
- **Rare Insight** : Agents can evolve via genetic algorithms for optimal performance.

---

## 9. Quick Tips for Scientists

- **Note-Taking** : Summarize each agent’s role.
- **Experiment** : Modify code inputs (e.g., city, data).
- **Research** : Propose new agents (e.g., sentiment analysis).
- **Network** : Join NLP communities on X or GitHub.

---

**Cheatsheet Note** : Keep this as a quick reference while studying the full tutorial. Run the `.py` files, review case studies, and use these snippets to build your own NLG systems!
