# Practical Exercise: BLEU Score and Probabilistic Bot
# Evaluates NLG output and extends chit-chat with probability

from nltk.translate.bleu_score import sentence_bleu
import random

# BLEU Score Evaluation
ref = [["hello", "world"]]
cand = ["hello", "universe"]
print("BLEU Score:", sentence_bleu(ref, cand))  # ~0.367

# Probabilistic Chit-Chat Bot
responses = {"hello": "Hi! How are you?"}


def prob_bot(input_text):
    if random.uniform(0, 1) > 0.2:
        return responses.get(input_text.lower(), "Let’s chat more!")
    return "Random chit-chat!"


# Test
print("Probabilistic Response:", prob_bot("hello"))
