import matplotlib.pyplot as plt


def draw_emotion_graph():
    """
    Visualize emotions in a valence-arousal 2D space as a scatter plot.
    """
    # Sample sentences with valence (happy/sad) and arousal (energy)
    sentences = [
        ("I’m so excited!", (0.8, 0.7)),  # Happy, high energy
        ("I’m really sad.", (-0.6, -0.2)),  # Sad, low energy
        ("I need help.", (0.0, 0.0)),  # Neutral
    ]

    valence = [v for _, (v, _) in sentences]
    arousal = [a for _, (_, a) in sentences]
    labels = [s for s, _ in sentences]

    plt.figure(figsize=(8, 8))
    plt.scatter(valence, arousal, color="blue")
    for i, label in enumerate(labels):
        plt.annotate(label, (valence[i], arousal[i]))

    plt.xlabel("Valence (Sad to Happy)")
    plt.ylabel("Arousal (Calm to Excited)")
    plt.title("Where Emotions Sit on the Map")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    draw_emotion_graph()
