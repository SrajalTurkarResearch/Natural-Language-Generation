from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    BartForConditionalGeneration,
)
import faiss
import numpy as np

# Initialize models
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-multiset-base"
)
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-multiset-base"
)
generator = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Index passages
passages = ["Paris is the capital of France.", ...]  # Preprocessed corpus
passage_embeddings = context_encoder.encode(passages)
index = faiss.IndexFlatIP(passage_embeddings.shape[1])
index.add(passage_embeddings)

# Query processing
query = "What is the capital of France?"
query_embedding = question_encoder.encode(query)
scores, indices = index.search(query_embedding, k=5)
retrieved_passages = [passages[i] for i in indices]

# Generate answer
input_text = f"Question: {query} Context: {' '.join(retrieved_passages)}"
answer = generator.generate(input_text)
print(answer)
