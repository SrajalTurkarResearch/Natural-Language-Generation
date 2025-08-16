```python
# nlg_core.py: Core NLG Functions for Soft Prompts and Embedding Control
# Run: python nlg_core.py
# Requires: transformers, peft, torch, sentence-transformers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model
from sentence_transformers import SentenceTransformer

class NLGToolbox:
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

if __name__ == "__main__":
    toolbox = NLGToolbox()
    toolbox.setup_soft_prompt()
    print("Generated:", toolbox.generate_text("The movie was"))
```