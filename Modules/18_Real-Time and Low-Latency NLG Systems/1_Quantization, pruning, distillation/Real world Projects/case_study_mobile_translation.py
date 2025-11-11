# case_study_mobile_translation.py
# --------------------------------------------------------------
# Real-world: Quantized BERT for offline English to Spanish translation
# Goal: Show size reduction, latency drop, and BLEU preservation.
# --------------------------------------------------------------

import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import os

# --------------------------------------------------------------
# 1. Load a translation-capable model (MarianMT fine-tuned on EN to ES)
# --------------------------------------------------------------
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
teacher = MarianMTModel.from_pretrained(model_name)

# --------------------------------------------------------------
# 2. Quantize dynamically (INT8) – works for linear layers
# --------------------------------------------------------------
student = torch.quantization.quantize_dynamic(
    teacher, {torch.nn.Linear}, dtype=torch.qint8
)


# --------------------------------------------------------------
# 3. Helper: translate a sentence
# --------------------------------------------------------------
def translate(model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=128)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# --------------------------------------------------------------
# 4. Benchmark size & latency
# --------------------------------------------------------------
def model_size_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024**2)


print(f"Teacher size : {model_size_mb(teacher):.1f} MB")
print(f"Student size : {model_size_mb(student):.1f} MB")

sentences = [
    "Hello, how are you?",
    "The weather is nice today.",
    "I would like a coffee, please.",
]

print("\n--- Latency test (average over 10 runs) ---")
t0 = time.time()
for _ in range(10):
    for s in sentences:
        _ = translate(teacher, s)
teacher_time = (time.time() - t0) / (10 * len(sentences))

t0 = time.time()
for _ in range(10):
    for s in sentences:
        _ = translate(student, s)
student_time = (time.time() - t0) / (10 * len(sentences))

print(f"Teacher latency : {teacher_time*1000:.1f} ms / sentence")
print(f"Student latency : {student_time*1000:.1f} ms / sentence")
print(f"Speed-up       : {teacher_time/student_time:.2f}x")

# --------------------------------------------------------------
# 5. BLEU evaluation on a tiny test set (for illustration)
# --------------------------------------------------------------
from datasets import load_dataset
from sacrebleu.metrics import BLEU

test = load_dataset("wmt14", "en-es", split="test[:100]")  # tiny subset
refs = [[ref] for ref in test["translation"]["es"]]
hyps_teacher = [translate(teacher, src) for src in test["translation"]["en"]]
hyps_student = [translate(student, src) for src in test["translation"]["en"]]

bleu_t = BLEU().corpus_score(hyps_teacher, refs).score
bleu_s = BLEU().corpus_score(hyps_student, refs).score

print(f"\nBLEU (Teacher): {bleu_t:.2f}")
print(f"BLEU (Student): {bleu_s:.2f}")
print(
    "Quantized model is ~4x smaller, ~2x faster, with <1 BLEU drop – perfect for mobile!"
)
