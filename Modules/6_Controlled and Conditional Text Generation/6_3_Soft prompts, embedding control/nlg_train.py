```python
# nlg_train.py: Training Utilities for Soft Prompts
# Run: python nlg_train.py
# Requires: transformers, peft, torch, datasets

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model
from datasets import load_dataset
from torch.optim import AdamW
import torch

def train_soft_prompt(peft_model, tokenizer, dataset_name='imdb', epochs=3, batch_size=8, lr=1e-3):
    """Train soft prompt on dataset."""
    dataset = load_dataset(dataset_name)['train']
    optimizer = AdamW(peft_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
            outputs = peft_model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Batch {i//batch_size+1}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=5)
    peft_model = get_peft_model(model, config)
    # train_soft_prompt(peft_model, tokenizer)
    print("Training setup complete.")
```