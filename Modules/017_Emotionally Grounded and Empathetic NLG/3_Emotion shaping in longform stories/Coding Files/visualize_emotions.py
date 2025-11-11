# visualize_emotions.py
"""
Advanced Visualization Suite for Emotion Shaping
Features: Interactive arcs, word clouds, sentiment heatmaps
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import numpy as np
from emotion_arc_models import cinderella, man_in_hole
from emotion_lexicon import EmotionLexicon


def plot_interactive_arc(arc_obj, data=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=arc_obj.x, y=arc_obj.y, mode="lines", name="Arc Model"))
    if data:
        fig.add_trace(
            go.Scatter(x=data[0], y=data[1], mode="markers", name="Data Points")
        )
    fig.update_layout(
        title="Interactive Emotional Arc",
        xaxis_title="Progress",
        yaxis_title="Intensity",
    )
    fig.show()


def generate_wordcloud(text, title="Emotion Word Cloud"):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


def sentiment_heatmap(story_text):
    lex = EmotionLexicon()
    sentences = [s.strip() for s in story_text.split(".") if s.strip()]
    data = []
    for sent in sentences:
        vec = lex.get_emotion_vector(sent)
        vec["sentence"] = sent[:50] + "..." if len(sent) > 50 else sent
        data.append(vec)
    df = pd.DataFrame(data).set_index("sentence")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, annot=True, cmap="RdYlGn", center=0)
    plt.title("Emotion Heatmap Across Sentences")
    plt.show()


# === DEMO ===
if __name__ == "__main__":
    arc = cinderella()
    plot_interactive_arc(arc)

    sample_text = (
        "She failed. But she tried again. Hope grew. Finally, she succeeded with joy."
    )
    generate_wordcloud(sample_text, "Sample Story Word Cloud")
    sentiment_heatmap(sample_text)
