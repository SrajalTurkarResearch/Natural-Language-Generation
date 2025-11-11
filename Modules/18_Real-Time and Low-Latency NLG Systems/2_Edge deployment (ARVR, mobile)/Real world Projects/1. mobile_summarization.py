# mobile_summarization.py
# Project: Simulate real-world mobile NLG for offline text summarization (e.g., emails, articles).
# Theory: In 2025, models like Llama 3.2 run on-device via quantization and NPUs (e.g., Snapdragon), ensuring privacy (no cloud send) and low latency (<300ms TTFT).
#         NLG pipeline: Content determination (extract key info), microplanning (lexical choice), realization (generate text).
#         Math: Autoregressive prob P(summary|text) = ∏ P(word_t | words_<t, text); use beam search for better quality (explore k paths).
# Logic: Load quantized model; prompt with text to summarize; output short version.
# As researcher: Measure battery impact (simulate with time/energy proxies); extend to multilingual via fine-tuning.
# Real-world: Apps like Android's on-device summarizer; benefits: Works in airplanes, protects sensitive data.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time  # For latency measurement

# Load small edge model (proxy for Llama 3.2; in prod, use 'meta-llama/Llama-3.2-1B')
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Quantize for mobile efficiency: int8 dynamic, targets linear layers (attention/FFN)
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Sample input: Simulate long email/text
input_text = """
Dear Team,
We had a productive meeting today discussing the Q4 sales strategy. Key points include increasing marketing budget by 15%, targeting new regions in Asia, and launching the premium product line by November. Challenges noted: Supply chain delays and competitor pricing. Action items: John to prepare budget report, Sarah to research Asia markets.
Best,
Manager
"""

# Prompt engineering: Frame for summarization task
prompt = f"Summarize the following text concisely: {input_text}\nSummary:"

# Generate summary
start_time = time.perf_counter()
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model_quantized.generate(
    inputs["input_ids"], max_length=100, num_beams=3, early_stopping=True
)  # Beam search: k=3 for quality
summary = (
    tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()
)
end_time = time.perf_counter()

print("Original Text:", input_text)
print("Generated Summary:", summary)
print(f"Latency: {end_time - start_time:.4f} seconds")  # Target: <0.5s on mobile

# Research extension: Evaluate with ROUGE (recall-oriented); implement via rouge-score lib
# from rouge_score import rouge_scorer
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
# reference = "Meeting on Q4 strategy: Increase budget 15%, target Asia, launch premium by Nov. Challenges: Delays, pricing. Actions: John budget, Sarah research."
# scores = scorer.score(reference, summary)
# print("ROUGE Scores:", scores)
# Experiment: Vary beam width; plot latency vs. quality trade-off
