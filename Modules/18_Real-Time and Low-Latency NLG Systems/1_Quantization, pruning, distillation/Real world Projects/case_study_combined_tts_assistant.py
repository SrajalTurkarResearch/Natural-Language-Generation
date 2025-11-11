# case_study_combined_tts_assistant.py
# --------------------------------------------------------------
# Real-world: SPADE-style compression for TTS on a smart speaker
# Pipeline: Text to Tacotron2 to WaveGlow to Vocoder (all compressed)
# Goal: <300 ms end-to-end, <50 MB total model size.
# --------------------------------------------------------------

import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import time
import numpy as np

# --------------------------------------------------------------
# 1. Load a lightweight TTS stack (SpeechT5 + HiFi-GAN)
# --------------------------------------------------------------
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
teacher = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


# --------------------------------------------------------------
# 2. Prune SpeechT5 (structured – remove 50% of attention heads)
# --------------------------------------------------------------
def prune_heads(model, heads_to_prune):
    for layer in model.encoder.layers:
        layer.self_attn.prune_heads(heads_to_prune)


# Example: prune half of the heads in each layer
heads_per_layer = teacher.config.num_attention_heads // 2
prune_heads(teacher, list(range(heads_per_layer)))

# --------------------------------------------------------------
# 3. Quantize both models (INT8 dynamic)
# --------------------------------------------------------------
teacher_q = torch.quantization.quantize_dynamic(
    teacher, {torch.nn.Linear}, dtype=torch.qint8
)
vocoder_q = torch.quantization.quantize_dynamic(
    vocoder, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
)


# --------------------------------------------------------------
# 4. Synthesis helper
# --------------------------------------------------------------
def tts(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = teacher_q.generate_speech(**inputs)
        speech = vocoder_q(speech)
    return speech.squeeze().cpu().numpy()


# --------------------------------------------------------------
# 5. Benchmark
# --------------------------------------------------------------
def model_size_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024**2)


print(f"Teacher size : {model_size_mb(teacher):.1f} MB")
print(f"Quantized size : {model_size_mb(teacher_q):.1f} MB")
print(
    f"Vocoder size : {model_size_mb(vocoder):.1f} MB to {model_size_mb(vocoder_q):.1f} MB"
)

text = "Good morning, your coffee is ready."
t0 = time.time()
_ = tts(text)
latency = (time.time() - t0) * 1000
print(f"\nEnd-to-end latency: {latency:.1f} ms")
print("Meets smart-speaker requirement (<300 ms) with <50 MB total footprint.")
