# Vector Stores (FAISS & BM25) in NLG: Cheatsheet for Aspiring Scientists

Dear Future Researcher,

This cheatsheet distills the essentials of vector stores (FAISS and BM25) for NLG, inspired by Turing’s clarity, Einstein’s insight, and Tesla’s innovation. Use it to reinforce your learning, take notes, and spark research ideas. It’s structured for quick reference: **Concepts** , **Formulas** , **Code Snippets** , **Visuals** , **Tips** , and **Research Notes** . Assume Python 3.12+; install: `pip install faiss-cpu rank_bm25 sentence-transformers numpy matplotlib torch scikit-learn`.

## 1. Core Concepts

- **Vector Store:** Database for similarity search (vectors = numeric text representations).
  - **Analogy:** Library where books (vectors) are found by similarity, not exact titles.
- **Sparse Vectors (BM25):** Mostly zeros, keyword-based (e.g., TF-IDF).
- **Dense Vectors (FAISS):** Full numbers, semantic (e.g., BERT embeddings).
- **NLG Role:** Retrieval-Augmented Generation (RAG): Retrieve → Generate text.
- **BM25:** Probabilistic ranking for keywords; fast, lexical.
- **FAISS:** Approximate nearest neighbor (ANN) for dense vectors; semantic, scalable.

## 2. Key Formulas

- **BM25 Score (Document (D), Query (Q)):**
  [
  \text{BM25}(D, Q) = \sum_i \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
  ]
  - (\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5})
  - (f(q_i, D)): Term frequency; (N): Total docs; (n(q_i)): Docs with (q_i).
  - (k_1 = 1.2), (b = 0.75), (\text{avgdl}): Average doc length.
- **Cosine Similarity (FAISS):**
  [
  \cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{||\vec{a}|| \cdot ||\vec{b}||}
  ]
- **Euclidean Distance (FAISS):**
  [
  d = \sqrt{\sum (a_i - b_i)^2}
  ]

## 3. Code Snippets

- **BM25 (Manual, from `bm25_implementation.py`):**

```python
  import re, numpy as np
  corpus = ["The quick fox", "Fox jumps"]
  query = "fox"
  def preprocess(text): return re.findall(r'\w+', text.lower())
  def compute_idf(corpus, term):
      N = len(corpus)
      n_term = sum(1 for doc in corpus if term in doc)
      return np.log((N - n_term + 0.5) / (n_term + 0.5)) if n_term else 0
  corpus_pre = [preprocess(doc) for doc in corpus]
  scores = [sum(compute_idf(corpus_pre, qt) for qt in preprocess(query)) for doc in corpus_pre]
  print(scores)
```

- **FAISS (from `faiss_implementation.py`):**

```python
  import faiss, numpy as np
  embeddings = np.random.rand(4, 384).astype('float32')  # Demo
  query_emb = np.random.rand(1, 384).astype('float32')
  index = faiss.IndexFlatIP(384)
  faiss.normalize_L2(embeddings)
  index.add(embeddings)
  scores, indices = index.search(query_emb, k=2)
  print(indices)
```

- **RAG Mock (from `mini_project_rag.py`):**

```python
  top_idx = np.argmax(scores)  # From BM25 or hybrid
  print(f"NLG: Based on '{corpus[top_idx]}', response: Generated text.")
```

## 4. Visualizations

- **Vector Space Plot:** Scatter plot, docs as blue dots, query as red star. Close = similar.
  ```python
  import matplotlib.pyplot as plt
  vec2d = np.array([[0.1, 0.2], [0.15, 0.25], [0.8, 0.1]])
  query2d = np.array([[0.12, 0.22]])
  plt.scatter(vec2d[:,0], vec2d[:,1], c='blue', label='Docs')
  plt.scatter(query2d[0,0], query2d[0,1], c='red', marker='*', label='Query')
  plt.legend(); plt.grid(); plt.show()
  ```
- **BM25 Bar Plot:** Docs on x-axis, scores on y-axis. Tall bars = relevant.
- **Sketch Tip:** Draw vectors as arrows from origin (like Tesla’s electric fields).

## 5. Practical Tips

- **BM25 Tuning:** Adjust `k1` (1.2–2.0, term saturation), `b` (0.5–0.8, length penalty).
- **FAISS Speed:** Use `IndexIVFFlat` (10–100 clusters); train on subset.
- **Dataset:** Start with 20 Newsgroups (`sklearn.datasets.fetch_20newsgroups`).
- **Debugging:** Print intermediate IDF, TF for BM25; check embedding norms for FAISS.
- **Research Hack:** Log metrics (precision@K, recall@K) in a lab notebook.

## 6. Research Notes for Scientists

- **Why It Matters:** Vector stores are the backbone of RAG, reducing NLG hallucinations (e.g., 40% in healthcare, per Case Study 1).
- **Experiment Ideas:**
  - Compare BM25 vs. FAISS on arXiv dataset (recall@10).
  - Test hybrid retrieval (e.g., 0.4 BM25 + 0.6 FAISS) on legal texts.
- **Rare Insight:** Embeddings may encode biases (e.g., over-representing certain topics). Research debiasing (like Einstein questioning assumptions).
- **Future Path:** Explore quantum-inspired ANN (Turing’s computation frontier) or multimodal retrieval (text+images, Tesla’s visionary leap).
- **Publish:** Benchmark retrieval metrics; submit to ACL/NeurIPS.

## 7. Quick Reference Table

| Aspect       | BM25 (Sparse)                | FAISS (Dense)                    |
| ------------ | ---------------------------- | -------------------------------- |
| Use Case     | Keyword search (e.g., legal) | Semantic search (e.g., chatbots) |
| Strength     | Fast, exact matches          | Captures meaning                 |
| Weakness     | Misses synonyms              | Computation-heavy                |
| Code Library | `rank_bm25`                  | `faiss`                          |
| NLG Role     | Quick retrieval              | Contextual augmentation          |

**Turing’s Advice:** Master the logic behind each line of code. **Einstein’s Wisdom:** Visualize high-dimensional spaces as 2D for intuition. **Tesla’s Vision:** Innovate by combining sparse and dense for ultimate NLG power.

**Note-Taking Tip:** Copy this cheatsheet into your research journal. Highlight formulas and code. Reflect: How can I apply this to my domain (e.g., bioinformatics)?
