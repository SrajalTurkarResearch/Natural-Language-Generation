# Cheat Sheet: Image-to-Text and Caption Storytelling in NLG

This cheat sheet summarizes the key concepts, terms, math, code snippets, and research tips for Image-to-Text and Caption Storytelling in NLG. It’s designed as a quick reference for aspiring scientists to reinforce learning and guide research, inspired by the clarity of Turing, the vision of Einstein, and the innovation of Tesla. Use this alongside the `.py` files and case studies to master the topic.

## Key Concepts

- **Image-to-Text (Image Captioning)** : Computer generates a description from a picture, e.g., "A cat sleeps on a couch."
- **Object Detection** : Finds things like "cat."
- **Scene Understanding** : Knows the setting, like "room."
- **Attribute Extraction** : Adds details, like "fluffy."
- **Caption Storytelling** : Turns descriptions into stories, e.g., "In a sunny room, a fluffy cat naps, dreaming of treats."
- **Narrative Structure** : Start, middle, end.
- **Emotional Inference** : Guesses feelings, like "happy."
- **Creative Augmentation** : Adds fun details, like "dreaming."
- **Computer Vision (CV)** : Helps computers see pictures using CNNs (Convolutional Neural Networks) or ViTs (Vision Transformers).
- **Natural Language Processing (NLP)** : Helps computers write words using Transformers or LSTMs.
- **Encoder-Decoder** : Encoder (sees picture) makes numbers; Decoder (writes words) makes sentences.
- **Attention** : Focuses on picture parts for each word, like looking at the cat for "cat."
- **Zero-Shot Learning** : Describes new pictures without extra training.

## Key Terms

- **Feature Vector** : Numbers summarizing a picture, e.g., f = [0.7, 0.3].
- **Embedding** : Numbers for words/pictures, e.g., "cat" = [0.1, 0.5, -0.2].
- **Beam Search** : Tries multiple sentence options, picks the best.
- **Pre-training** : Learns general skills from big data (e.g., web images).
- **Fine-tuning** : Adjusts for specific tasks (e.g., medical images).
- **Hyperparameters** : Settings like batch size or epochs (learning rounds).

## Mathematical Foundations

- **Image to Features** :
  [
  f = \text{CNN}(I) = \sum W_k \cdot I + b_k
  ]
  (W_k: weights, I: image, b_k: bias)
- **Attention** :
  [
  \text{Attention}(Q, K, V) = \softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) V
  ]
  (Q: query, K: key, V: value, d_k: vector size)
- **Sentence Probability** :
  [
  P(\text{sentence} | f) = \prod P(\text{word}_t | \text{previous words}, f)
  ]
- **Loss** :
  [
  L = -\sum \log P(\text{correct word})
  ]
- **Example** :
  For "A cat sleeps," f = [0.7, 0.3]:
- P("A"|f) = 0.75, P("cat"|"A", f) = 0.8, P("sleeps"|"A cat", f) = 0.9
- Total P = 0.75 × 0.8 × 0.9 = 0.54
- Loss = - (log 0.75 + log 0.8 + log 0.9) ≈ 0.616

## Key Code Snippets

### Basic Captioning (BLIP)

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
url = "https://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Caption:", caption)
```

### Storytelling (GPT-2)

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
prompt = "Create a story based on: A cat sleeps on a couch. Story:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
story = tokenizer.decode(outputs[0])
print("Story:", story)
```

### Evaluation (BLEU)

```python
from nltk.translate.bleu_score import sentence_bleu

reference = [["A", "cat", "sleeps", "on", "a", "couch"]]
candidate = ["A", "cat", "is", "resting"]
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print("BLEU-1 Score:", score)
```

## Visualizations

- **Attention Heatmap** : Shows where the model looks (hot colors = more focus).

```python
  import matplotlib.pyplot as plt
  import numpy as np
  attention = np.random.rand(224, 224)
  plt.imshow(attention, cmap="hot")
  plt.title("Attention Heatmap")
  plt.colorbar()
  plt.show()
```

- **Model Architecture** : Encoder (CNN/ViT) → Attention → Decoder (Transformer).

## Real-World Applications

- **Healthcare** : X-ray reports, e.g., "Abnormal shadow in lung" (RefCap, Nature 2023).
- **Accessibility** : Seeing AI narrates for blind users, e.g., "Person smiles in park."
- **Autonomous Vehicles** : Tesla describes scenes, e.g., "Pedestrian crossing ahead."
- **Retail** : Amazon auto-generates product descriptions, e.g., "Blue T-shirt."
- **Education** : Captions for diagrams, e.g., "Water molecule bonding."

## Evaluation Metrics

- **BLEU** : Counts matching words (BLEU-1 to BLEU-4).
- **METEOR** : Allows similar words (e.g., "run" and "dash").
- **CIDEr** : Weights important words.
- **SPICE** : Checks meaning (objects, relations).
- **Example** : Reference: "A cat sleeps." Candidate: "A cat rests." BLEU-1 ≈ 0.67.

## Research Directions

- **Zero-Shot Learning** : Caption new images without training (TPCap, arXiv 2025).
- **Video Storytelling** : Use 3D CNNs for moving images.
- **Multilingual** : Support non-English captions.
- **Ethics** : Reduce bias (e.g., gender stereotypes in datasets).
- **New Metrics** : Combine BLEU and SPICE for better meaning evaluation.

## Exercises

1. **Basic** : Run BLIP code on 3 images. Write if captions are correct.
2. **Intermediate** : Change GPT-2 max_length to 150. Compare story quality.
3. **Advanced** : Calculate BLEU for "A dog runs" vs. "Dog chases ball."
4. **Project** : Build a captioner on 10 personal photos with custom captions.

## What’s Missing in Standard Tutorials

- **Ethics** : Addressing bias in datasets (e.g., cultural underrepresentation).
- **Scalability** : Distributed training for large datasets.
- **Multimodal** : Combining images with audio/video.
- **Rare Insight** : Reinforcement learning with CIDEr rewards for diverse captions.

## Future Steps

- **Read** : "Show and Tell" (2015), TPCap (2025) on arXiv.
- **Experiment** : Fine-tune BLIP on a small dataset (e.g., Flickr30k).
- **Join** : Hugging Face community, follow arXiv for latest papers.
- **Innovate** : Propose a new metric or bias-reduction technique.
