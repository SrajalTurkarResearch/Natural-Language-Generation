```python
# nlg_comprehensive.py: World-Class NLG Toolkit for Soft Prompts and Embedding Control
# Run: python nlg_comprehensive.py
# Requires: transformers, peft, torch, sentence-transformers, matplotlib, numpy, datasets
# Purpose: Comprehensive, evergreen toolkit for beginners to master NLG, covering theory, code, visualizations, projects, exercises, research directions, and more.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

class NLGComprehensive:
    """
    Comprehensive NLG class for soft prompts and embedding control.
    
    Theory:
    - NLG generates text from data/prompts, automating scientific communication (e.g., reports).
    - Embeddings: Vectors in high-dimensional space capturing meaning (e.g., 'cat' ~ [0.1, -0.2, 0.5]).
    - Soft Prompts: Trainable vectors for PEFT, unlike fixed hard prompts.
    - Embedding Control: Manipulate vectors for style/tone (e.g., add 'happy' vector).
    
    Analogy: NLG = Chef (model) turning ingredients (data) into dishes (text). Soft prompts = Digital recipe that learns.
    
    Rare Insights (2023-2025):
    - Interpretable Prompts (arXiv 2025): Map to readable text, use CPT to limit overfitting.
    - Input-Dependent Prompts (arXiv 2025): Self-attention for dynamic prompts.
    - Mixture of Soft Prompts (MSP) (EMNLP 2023): Multi-attribute text.
    - Vulnerabilities (OpenReview 2024): Prompt injection risks. Use hybrid prompts.
    - Ethics: Subtract bias vectors for fairness.
    
    Applications:
    - Healthcare: Medical reports.
    - Education: Personalized lessons.
    - Science: Hypothesis generation.
    - Ethics: Bias mitigation.
    
    Research Directions:
    - Multimodal NLG (text+image).
    - Interpretable prompts.
    - Dynamic prompts.
    - Ethical controls.
    
    Future Trends:
    - Hybrid prompts.
    - Auto-optimized prompts (RL).
    - Quantum-inspired embeddings.
    
    What Was Missing:
    - Ethics: Bias, hallucination control.
    - Scalability: Distributed training.
    - Metrics: BLEU, ROUGE.
    - Ablations: Baseline without prompts.
    - Interdisciplinary: Physics, biology.
    - Security: Prompt injection.
    - Data Efficiency: Few-shot learning.
    """
    
    def __init__(self, model_name='gpt2', embed_model_name='all-MiniLM-L6-v2'):
        """Initialize models."""
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_model = SentenceTransformer(embed_model_name)
        self.peft_model = None

    def setup_soft_prompt(self, num_tokens=5, init_text="Generate positive text:"):
        """Configure soft prompt."""
        config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=num_tokens,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text=init_text
        )
        self.peft_model = get_peft_model(self.model, config)
        return self.peft_model

    def generate_text(self, prompt, max_length=50):
        """Generate text with soft prompt."""
        if not self.peft_model:
            raise ValueError("Setup soft prompt first.")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.peft_model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0])

    def control_embedding(self, text, target_text, neutral_text, lambda_=0.5):
        """Shift embedding for tone/style."""
        e_text = self.embed_model.encode(text)
        e_target = self.embed_model.encode(target_text)
        e_neutral = self.embed_model.encode(neutral_text)
        return e_text + lambda_ * (e_target - e_neutral)

    def visualize_embeddings(self, texts, labels, title="Embedding Space"):
        """Visualize embeddings in 2D."""
        embeddings = [self.embed_model.encode(t) for t in texts]
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        plt.scatter(reduced[:,0], reduced[:,1])
        for i, label in enumerate(labels):
            plt.text(reduced[i,0], reduced[i,1], label)
        plt.title(title)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.show()

    def train_soft_prompt(self, dataset_name='imdb', epochs=3, batch_size=8, lr=1e-3):
        """Train soft prompt on dataset."""
        dataset = load_dataset(dataset_name)['train']
        optimizer = AdamW(self.peft_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                inputs = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                outputs = self.peft_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f'Batch {i//batch_size+1}, Loss: {loss.item():.4f}')

    def run_mini_project(self):
        """Mini Project: Train soft prompt for positive reviews."""
        print("Running Mini Project: Sentiment-Tuned NLG")
        self.setup_soft_prompt()
        # self.train_soft_prompt()  # Uncomment with dataset
        print("Generated:", self.generate_text("The movie was"))

    def run_exercise_1(self):
        """Exercise 1: Vary lambda in embedding control."""
        print("Running Exercise 1: Varying λ")
        text = "The results are good."
        formal = "The outcomes are satisfactory."
        neutral = "The results are okay."
        lambdas = [0.1, 0.5, 1.0]
        controlled = [self.control_embedding(text, formal, neutral, l) for l in lambdas]
        texts = [text, formal, neutral] + [text] * len(lambdas)
        labels = ['Input', 'Formal', 'Neutral'] + [f'Controlled λ={l}' for l in lambdas]
        self.visualize_embeddings(texts, labels, "Exercise 1: Varying λ")

    def run_exercise_3(self):
        """Exercise 3: Mixture of Soft Prompts (MSP) - Simplified."""
        print("Running Exercise 3: MSP")
        self.setup_soft_prompt(init_text="Generate positive scientific text:")
        print("MSP Output:", self.generate_text("The experiment was"))

if __name__ == "__main__":
    """
    Tutorial: Use this file as a standalone NLG toolkit.
    - Run mini project: nlg.run_mini_project()
    - Run exercises: nlg.run_exercise_1()
    - Extend for major project (e.g., Streamlit app in nlg_app.py)
    - Next Steps: Experiment with datasets, publish on arXiv.
    - Tips: Start small, validate ethics, attend NeurIPS.
    """
    nlg = NLGComprehensive()
    nlg.setup_soft_prompt()
    print("Generated:", nlg.generate_text("The movie was"))
    nlg.run_exercise_1()
    nlg.run_mini_project()
```