# RAG Cheat Sheet: Quick Reference for Aspiring Scientists

Dear Scholar,

This cheat sheet, crafted in the spirit of Turing’s clarity, Einstein’s insight, and Tesla’s precision, summarizes the Retrieval-Augmented Generation (RAG) tutorial. Use it to revise, experiment, and spark research ideas. Keep it in your notebook for quick access.

## 1. Core Concepts

- **NLG** : Generates human-like text from data. _Analogy_ : Chef (AI) cooking text from raw ingredients (data).
- **LLMs** : Predict next words using transformers. _Formula_ : $P(w_1, ..., w_n) = \prod P(w_i | w_1, ..., w_{i-1})$.
- **RAG** : Combines retrieval (external data) with generation (LLM). _Pipeline_ :

1. Embed query → Vector $e_q$.
2. Retrieve top-k docs (cosine similarity: $\frac{e_q \cdot e_d}{||e_q|| \cdot ||e_d||}$).
3. Augment prompt → LLM generates answer.

- **Why RAG?** Reduces hallucinations, updates knowledge, customizes via databases.

## 2. Key Components

- **Knowledge Base** : Documents in vector DB (e.g., FAISS). _Think_ : Library index.
- **Retriever** : Finds relevant docs via embeddings. _Math_ : Cosine similarity.
- **Generator** : LLM (e.g., GPT-2, Llama) creates text.
- **Embedder** : Converts text to vectors (e.g., Sentence-BERT).

## 3. Essential Code Snippets

```python
# Setup Embeddings (rag_setup.py)
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Knowledge Base (rag_knowledge_base.py)
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(documents, embeddings)

# Run RAG (rag_retrieval_generation.py)
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
llm = HuggingFaceHub(repo_id="gpt2")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(k=2))
response = qa_chain.run("What is RAG?")
```

## 4. Visualizations

- **Embedding Plot** : Use PCA to reduce vectors to 2D. _Code_ :

```python
  from sklearn.decomposition import PCA
  import matplotlib.pyplot as plt
  pca = PCA(n_components=2)
  reduced = pca.fit_transform(embeddings)
  plt.scatter(reduced[:,0], reduced[:,1])
```

- **Pipeline Diagram** : Sketch: Query → Embed → Retrieve → Augment → Generate.
- _Tip_ : Closer points in 2D = higher similarity.

## 5. Key Formulas

- **Softmax for Word Prediction** : $P(w_i) = \frac{e^{s_i}}{\sum e^{s_j}}$, where $s_i$ is score for word $i$.
- **Cosine Similarity** : $\text{cos}(\theta) = \frac{e_q \cdot e_d}{||e_q|| \cdot ||e_d||}$.
- **ROUGE-1 (Evaluation)** : $F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$, where Precision = $\frac{\text{Overlap}}{\text{Generated}}$, Recall = $\frac{\text{Overlap}}{\text{Reference}}$.

## 6. Applications

- **Healthcare** : Retrieve patient records → Generate diagnostics.
- **Legal** : Retrieve case laws → Draft arguments.
- **Research** : Retrieve arXiv papers → Summarize findings.

## 7. Research Tips

- **Experiment** : Vary k in retrieval (try `k=1` vs. `k=5` in `rag_retrieval_generation.py`).
- **Evaluate** : Use ROUGE, BLEU, or human scores (see `rag_exercises.py`).
- **Innovate** : Test adaptive RAG (dynamic k) or multi-modal RAG (text + images).
- **Ethics** : Check for bias in retrieved data (e.g., skewed news sources).
- _Einstein’s Advice_ : Hypothesize: How can RAG solve a problem in your field?

## 8. Common Pitfalls

- **Hallucinations** : Pure LLMs invent facts. Solution: RAG grounds in data.
- **Noisy Retrieval** : Irrelevant docs. Fix: Hybrid retrieval or reranking.
- **Scalability** : Large DBs slow retrieval. Optimize: Quantize models, use FAISS.

## 9. Quick Exercises

1. **Basic** : Run `rag_exercises.py` (Exercise 1) with `k=1`. Note response focus.
2. **Intermediate** : Compute cosine similarity (Exercise 2). _Expect_ : 0.7-0.9 for similar texts.
3. **Advanced** : Implement reranking (Exercise 3). _Goal_ : Improve precision.

## 10. Next Steps

- Read: Lewis et al. (2020) RAG paper.
- Try: LlamaIndex or Haystack for advanced RAG.
- Research: Apply RAG to your domain (e.g., physics simulations).
- _Tesla’s Vision_ : Build a RAG prototype for a real dataset (use `rag_major_project.py`).

## 11. Missing in Standard Tutorials

- **Uncertainty** : Add Bayesian priors to retrieval scores.
- **Ablation** : Test RAG without retrieval to quantify impact.
- **Reproducibility** : Use `torch.manual_seed(42)` for consistent results.

This cheat sheet is your compass. Review it, run the `.py` files, and question like Einstein: _What if RAG unlocks new scientific frontiers?_
