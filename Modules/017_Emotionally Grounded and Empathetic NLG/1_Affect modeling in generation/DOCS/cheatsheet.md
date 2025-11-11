# Affect Modeling in NLG – Cheatsheet

> **One-Page Reference for Scientists & Researchers** > _Print | Pin | Master_

---

## 1. Core Concepts

| Term          | Definition                                         | Example                          |
| ------------- | -------------------------------------------------- | -------------------------------- |
| **Affect**    | Emotional tone in text                             | “I’m thrilled!” → joy            |
| **NLG**       | Natural Language Generation                        | Chatbot replies, stories         |
| **Valence**   | Positive ↔ Negative                                | +0.8 (happy) → -0.7 (sad)        |
| **Arousal**   | Calm ↔ Excited                                     | Low: “content”, High: “panicked” |
| **Ekman’s 6** | happiness, sadness, anger, fear, surprise, disgust | —                                |

---

## 2. Emotion Models

```text
Categorical: [happy, sad, angry, ...]
Dimensional: Valence-Arousal Plane
  Excited ◉       Angry
     ┌────────────┐
Calm ◉│            │◉ Sad
     └────────────┘
     Negative   Positive
```

---

## 3. Key Math

### Cosine Similarity (Word Embeddings)

```python
sim = (A · B) / (||A|| × ||B||)
```

> `happy` ↔ `joyful` → **0.99**

### Logistic Regression (Sentiment)

```python
P(positive) = 1 / (1 + e^-(β₀ + β₁x₁ + ...))
```

### Cross-Entropy Loss (Fine-tuning)

```python
L = -Σ y log(ŷ)
```

---

## 4. Tools & Libraries

| Task          | Library                 | Command                        |
| ------------- | ----------------------- | ------------------------------ |
| Sentiment     | `nltk.sentiment.vader`  | `SentimentIntensityAnalyzer()` |
| Generation    | `transformers`          | `pipeline('text-generation')`  |
| Fine-tuning   | `Trainer`               | `trainer.train()`              |
| Visualization | `matplotlib`,`seaborn`  | `plt.scatter()`                |
| Embeddings    | `sentence-transformers` | `model.encode()`               |

---

## 5. Datasets

| Name            | Link                                                         | Use                      |
| --------------- | ------------------------------------------------------------ | ------------------------ |
| **EmoBank**     | [Hugging Face](https://huggingface.co/datasets/emobank)      | Valence-Arousal labels   |
| **DailyDialog** | [Link](http://yanran.li/dailydialog)                         | Emotion-tagged dialogues |
| **GoEmotions**  | [Hugging Face](https://huggingface.co/datasets/go_emotions)  | 27 emotions              |
| **ISEAR**       | [Link](https://www.affective-sciences.org/researchmaterial/) | Cross-cultural emotions  |

---

## 6. Code Snippets

### Detect Sentiment

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores("I’m so happy!")['compound']
```

### Generate Affective Text

```python
prompt = "Respond empathetically: I failed my exam."
generator(prompt, max_length=50)
```

### Visualize Valence-Arousal

```python
plt.scatter(valence, arousal)
plt.xlabel("Valence"); plt.ylabel("Arousal")
```

---

## 7. Research Keywords

```
"affect modeling" + NLG
"emotion-aware language model"
"empathetic response generation"
"valence arousal prediction"
"cultural affect adaptation"
```

---

## 8. Next Steps (Your Roadmap)

| Level        | Action                                   |
| ------------ | ---------------------------------------- |
| Beginner     | Run `affective_nlg.py`→ tweak prompts    |
| Intermediate | Fine-tune GPT-2 on EmoBank               |
| Advanced     | Build multimodal affect (text + voice)   |
| Scientist    | Publish: “Cross-Cultural Empathy in NLG” |

---

> **Print this. Stick it on your wall.**
> You now have the **complete toolkit** to become a leader in affective AI.
