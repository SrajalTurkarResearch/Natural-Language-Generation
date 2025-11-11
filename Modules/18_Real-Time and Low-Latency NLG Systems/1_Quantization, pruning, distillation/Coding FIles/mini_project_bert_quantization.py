# mini_project_bert_quantization.py
# Mini-project: Quantize BERT and run inference.
# Theory: Post-training dynamic quantization for NLG tasks.
# Requires transformers and torch; downloads models on first run.
# As a researcher, extend to measure accuracy on GLUE subsets.

from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Apply dynamic quantization (targets linear layers to INT8)
model_q = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear: torch.qint8}, dtype=torch.qint8
)

# Example NLG-related inference (e.g., embedding generation)
input_text = "Hello world"  # Replace with NLG prompt
inputs = tokenizer(input_text, return_tensors="pt")
output = model_q(**inputs)

print("Output shape:", output.last_hidden_state.shape)
print("Quantized model ready for deployment.")

# Research extension: Evaluate on SST-2 dataset for accuracy drop.
