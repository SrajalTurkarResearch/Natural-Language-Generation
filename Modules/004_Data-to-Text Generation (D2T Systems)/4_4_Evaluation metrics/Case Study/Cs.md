# Case Studies: Applying NLG Evaluation Metrics

This document presents two detailed case studies showing how NLG evaluation metrics are used in real-world scenarios. These examples complement the Jupyter Notebook tutorial and offer practical insights for your scientific career.

---

## Case Study 1: Evaluating a Customer Service Chatbot

**Scenario:**  
A retail company developed a chatbot to answer customer queries such as “What’s the return policy?” They needed to evaluate its responses for both accuracy and user satisfaction.

**Setup:**

- **Task:** Generate responses to customer queries.
- **Reference Response:**  
  “You can return products within 30 days with a receipt for a full refund.”
- **Generated Response:**  
  “Returns are accepted within 30 days if you have a receipt.”
- **Metrics Used:**
  - **BLEU:** Measures n-gram overlap.
  - **ROUGE-1:** Measures recall of words.
  - **Human Evaluation:** Scores for clarity and tone (1–5 scale).

**Process:**

1. Collected 100 query-response pairs.
2. Computed BLEU and ROUGE using Python libraries.
3. 50 users rated responses for clarity and tone.

**Results:**

- **BLEU:** 0.85 (high overlap due to similar wording)
- **ROUGE-1:** 0.90 (most reference words captured)
- **Human Evaluation:**
  - Clarity: 4.5/5
  - Tone: 4/5 (slightly less formal than desired)
- **Visualization:**  
  A bar plot showed BLEU and ROUGE scores, with human scores overlaid as annotations.

**Outcome:**  
High BLEU and ROUGE scores confirmed the chatbot’s accuracy, but human feedback highlighted the need for friendlier language. The company fine-tuned the model to use more conversational phrases, improving user satisfaction.

**Lessons:**

- Combining automatic and human metrics provides a complete picture.
- Human feedback is critical for tone and user experience.

---

## Case Study 2: News Article Summarization

**Scenario:**  
A news agency used an NLG system (T5 model) to summarize articles for quick reader updates. They evaluated summaries to ensure key information was retained.

**Setup:**

- **Task:** Summarize a 500-word article on a hurricane.
- **Reference Summary:**  
  “Hurricane causes widespread flooding in coastal areas, displacing thousands.”
- **Generated Summary:**  
  “Coastal regions face severe flooding from hurricane, displacing many residents.”
- **Metrics Used:**
  - **ROUGE-L:** Measures longest common subsequence.
  - **BERTScore:** Measures semantic similarity.
  - **Human Evaluation:** Scores for informativeness and conciseness.

**Process:**

1. Used the CNN/DailyMail dataset for 50 articles.
2. Generated summaries with T5.
3. Computed ROUGE-L and BERTScore.
4. 10 journalists rated summaries.

**Results:**

- **ROUGE-L:** 0.82 (captured key sequences like “flooding” and “displacing”)
- **BERTScore:** 0.92 (high semantic similarity despite different wording)
- **Human Evaluation:**
  - Informativeness: 4.8/5
  - Conciseness: 4.6/5
- **Visualization:**  
  A scatter plot compared ROUGE-L vs. BERTScore, showing strong correlation.

**Outcome:**  
BERTScore’s high value confirmed the summaries captured meaning, even with paraphrasing. ROUGE-L ensured key content was retained. Human feedback validated the results, leading to deployment in the agency’s app.

**Lessons:**

- Embedding-based metrics like BERTScore are powerful for paraphrased text.
- Domain-specific human evaluation ensures practical usability.
