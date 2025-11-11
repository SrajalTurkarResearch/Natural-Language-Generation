# emotion_lexicon.py
"""
Emotion Lexicon & Sentiment Tools for NLG
Supports: NRC, VADER, TextBlob
"""

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import os
from collections import defaultdict

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)


class EmotionLexicon:
    def __init__(self, path=None):
        self.analyzer = SentimentIntensityAnalyzer()
        self.nrc = self.load_nrc(path)

    def load_nrc(self, path=None):
        if path is None:
            # Auto-download NRC (public domain)
            url = "https://raw.githubusercontent.com/niharsawant/Emotion-Lexicon/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
            import urllib.request

            path = "NRC-Emotion-Lexicon.txt"
            if not os.path.exists(path):
                print("Downloading NRC Lexicon...")
                urllib.request.urlretrieve(url, path)
        data = pd.read_csv(
            path, sep="\t", header=None, names=["word", "emotion", "value"]
        )
        lexicon = defaultdict(dict)
        for _, row in data.iterrows():
            if row["value"] == 1:
                lexicon[row["word"]][row["emotion"]] = 1
        return lexicon

    def get_emotion_vector(self, text):
        words = nltk.word_tokenize(text.lower())
        vec = {
            emo: 0
            for emo in [
                "anger",
                "anticipation",
                "disgust",
                "fear",
                "joy",
                "sadness",
                "surprise",
                "trust",
            ]
        }
        for word in words:
            if word in self.nrc:
                for emo in vec:
                    vec[emo] += self.nrc[word].get(emo, 0)
        total = sum(vec.values()) + 1e-8
        return {k: v / total for k, v in vec.items()}

    def sentiment_score(self, text):
        return self.analyzer.polarity_scores(text)["compound"]

    def dominant_emotion(self, text):
        vec = self.get_emotion_vector(text)
        return max(vec, key=vec.get)


# === DEMO ===
if __name__ == "__main__":
    lex = EmotionLexicon()
    text = "She smiled as the sun rose over the peaceful valley."
    print("Text:", text)
    print("Sentiment:", lex.sentiment_score(text))
    print("Emotions:", lex.get_emotion_vector(text))
    print("Dominant:", lex.dominant_emotion(text))
