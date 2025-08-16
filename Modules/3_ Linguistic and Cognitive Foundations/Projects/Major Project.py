# Linguistic and Cognitive Foundations: Major Project - Coherence-Aware Text Generator
# Theory:
# - Build a rule-based text generator ensuring coherence via Centering principles.
# - Objective: Generate text with consistent entity focus.
# - Applications: Dialogue systems, text generation.
# - Research Direction: Integrate neural models (e.g., GPT) for advanced generation.
# - Rare Insight: Coherence-aware generation can mimic human-like dialogue flow.

import networkx as nx
import matplotlib.pyplot as plt


# Generate Coherent Text
def generate_coherent_text(entities, actions):
    print("\nMajor Project: Coherence-Aware Text Generator")
    sentences = []
    cb = entities[0]  # Initial focus
    sentences.append(f"{cb} {actions[0]}.")
    sentences.append(f"It {actions[1]} at {entities[1]}.")
    return sentences


# Centering Analysis
def centering_analysis(sentences):
    print("\nCentering Analysis of Generated Text:")
    for i, sent in enumerate(sentences):
        entities = sent.split()
        cf = [e for e in entities if e in entities]
        cb = None if i == 0 else sentences[i - 1].split()[0]
        transition = "None" if i == 0 else ("Continue" if cb in cf else "Shift")
        print(f"Sentence {i+1}: {sent}, Cf: {cf}, Cb: {cb}, Transition: {transition}")


# Visualize Coherence
def visualize_coherence(sentences):
    G = nx.DiGraph()
    G.add_nodes_from(sentences)
    G.add_edge(sentences[0], sentences[1], relation="Continue")
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_color="lightgreen", node_size=2000, font_size=10
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    )
    plt.title("Coherence Graph for Generated Text")
    plt.savefig("major_project_coherence_graph.png")
    plt.close()
    print("Graph saved as 'major_project_coherence_graph.png'.")


# Example
entities = ["John", "dog"]
actions = ["saw a dog", "barked"]
text = generate_coherent_text(entities, actions)
print("Generated Text:", " ".join(text))
centering_analysis(text)
visualize_coherence(text)

# Notes for Researchers:
# - Extend with neural models (e.g., Hugging Face Transformers) for dynamic generation.
# - Test on real datasets (e.g., news articles) for coherence evaluation.
# - Explore emotional context in coherence-aware generation.
