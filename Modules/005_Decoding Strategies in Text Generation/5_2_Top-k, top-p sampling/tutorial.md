Mastering Top-k and Top-p Sampling in Natural Language Generation: A Beginner's Tutorial for Aspiring Scientists and Researchers
Section 1: Introduction to Natural Language Generation (NLG)
1.1 What is NLG?
Natural Language Generation (NLG) is a subfield of artificial intelligence where computers generate human-readable text. It powers applications like chatbots, automated report generators, and creative writing tools. For a researcher, NLG is akin to synthesizing coherent language from statistical patterns, much like deriving a physical law from experimental data.

Core Logic: NLG models predict the next word (or token, a unit like a word or subword) based on patterns learned from vast text corpora. Each prediction builds a sequence, forming sentences.

1.2 Why Sampling Techniques Like Top-k and Top-p?
NLG models generate text by selecting tokens from a probability distribution. Choosing the highest-probability token every time (greedy decoding) produces repetitive, predictable text. Sampling introduces controlled randomness to enhance creativity, akin to an engineer iterating designs to optimize innovation.

Analogy: Writing a story by always picking the most obvious word is dull (e.g., "The dog chased the... cat"). Sampling is like brainstorming creative plot twists by choosing from a curated set of good options.

Section 2: Basics of How Language Models Work
2.1 Language Models and Probabilities
A language model (LM) predicts the next token in a sequence by assigning probabilities to all possible tokens in its vocabulary (e.g., 50,000 tokens in modern models). These probabilities come from logits, raw scores output by the model, transformed via the softmax function.

Softmax Function:For logits $$ z*1, z_2, \dots, z_V $$ (where $$ V $$ is vocabulary size), the probability $$ p_i $$ for token $$ i $$ is:$$p_i = \frac{e^{z_i}}{\sum*{j=1}^V e^{z_j}}$$
Logic: Exponentiation ($$ e^{z_i} $$) amplifies higher logits; normalization ensures probabilities sum to 1.
Analogy: Think of softmax as a voting system where logits are candidate scores, and votes (probabilities) reflect relative popularity.

2.2 Example: A Tiny Vocabulary
Consider a vocabulary of five tokens with logits: "apple" (3), "banana" (2), "cat" (1), "dog" (0), "elephant" (-1).

Calculation:

Exponentials: $$ e^3 \approx 20.09 $$, $$ e^2 \approx 7.39 $$, $$ e^1 \approx 2.72 $$, $$ e^0 = 1 $$, $$ e^{-1} \approx 0.37 $$.
Sum: $$ 20.09 + 7.39 + 2.72 + 1 + 0.37 \approx 31.57 $$.
Probabilities: apple ≈ 0.636, banana ≈ 0.234, cat ≈ 0.086, dog ≈ 0.032, elephant ≈ 0.012.

Visualization:
Probability Distribution:
apple: |||||||||||||||||||||| (0.64)
banana: |||||||||| (0.23)
cat: |||| (0.09)
dog: || (0.03)
elephant: | (0.01)

Research Note: Real models have skewed distributions—high probabilities for few tokens, low for most. This shapes sampling strategies.

Section 3: Decoding Strategies in NLG
3.1 What is Decoding?
Decoding is the process of selecting tokens from a probability distribution to form text. Common methods include:

Greedy Decoding: Always pick the highest-probability token (deterministic, often repetitive).
Beam Search: Track multiple high-probability sequences (complex, less creative).

3.2 Why Sampling?
Pure random sampling from all probabilities risks selecting low-probability tokens, leading to incoherent text. Top-k and Top-p sampling introduce controlled randomness to balance coherence and creativity.

Analogy: Sampling is like choosing ingredients for a dish—not random spices, but a curated selection of the best ones to ensure a tasty result.

Application: In scientific writing tools, sampling generates diverse hypotheses or explanations, avoiding repetitive phrases.

Section 4: Top-k Sampling – Theory and Details
4.1 What is Top-k Sampling?
Top-k sampling selects the top $$ k $$ highest-probability tokens, renormalizes their probabilities to sum to 1, and samples from this subset.

Theory: By focusing on the $$ k $$ most likely tokens, it avoids low-probability "junk" tokens. The parameter $$ k $$ controls diversity: small $$ k $$ is conservative, large $$ k $$ is diverse.

Math:

Sort probabilities in descending order: $$ p*{(1)} \geq p*{(2)} \geq \dots \geq p*{(V)} $$.
Take top $$ k $$: Sum $$ S = \sum*{i=1}^k p*{(i)} $$.
Renormalize: $$ q_i = \frac{p*{(i)}}{S} $$ for $$ i = 1 $$ to $$ k $$.
Sample a token with probability $$ q_i $$.

Analogy: Like choosing from the top $$ k $$ bestselling books in a bookstore—reliable but may miss niche gems.

4.2 Example with Calculation
Using our example: apple=0.636, banana=0.234, cat=0.086, dog=0.032, elephant=0.012. Set $$ k=3 $$.

Steps:

Top 3: apple (0.636), banana (0.234), cat (0.086).
Sum: $$ S = 0.636 + 0.234 + 0.086 = 0.956 $$.
Renormalized: apple ≈ $$ \frac{0.636}{0.956} \approx 0.665 $$, banana ≈ $$ \frac{0.234}{0.956} \approx 0.245 $$, cat ≈ $$ \frac{0.086}{0.956} \approx 0.090 $$.
Sample: Random draw might select "banana" (probability 0.245).

Context Example: For prompt "The fruit is...", top-k=3 likely selects apple or banana, avoiding irrelevant tokens like "cat".

4.3 Real-World Cases

Chatbots: Top-k=50 ensures responses are coherent yet varied, as in AI assistants like myself.

Creative Writing: Used in story generation to produce diverse narratives without implausible words.

Scientific Application: In drug discovery, top-k sampling generates chemical descriptions, ensuring terms are chemically valid.

Pros/Cons:

Pros: Simple, predictable diversity.
Cons: Fixed $$ k $$ may exclude good tokens if probabilities are flat or include too many if overly large.

4.4 Visualization
Sketch a bar chart:
Original Probabilities:
apple: |||||||||||||||||||||| (0.64)
banana: |||||||||| (0.23)
cat: |||| (0.09)
dog: || (0.03)
elephant: | (0.01)

Top-k=3 (Renormalized):
apple: |||||||||||||||||||||||| (0.67)
banana: ||||||||| (0.24)
cat: ||| (0.09)

Section 5: Top-p Sampling (Nucleus Sampling) – Theory and Details
5.1 What is Top-p Sampling?
Top-p sampling (or nucleus sampling) selects the smallest set of tokens whose cumulative probability exceeds $$ p $$, renormalizes their probabilities, and samples from this "nucleus."

Theory: Unlike fixed-size top-k, top-p adapts to the probability distribution. If probabilities are concentrated, the nucleus is small; if spread, it’s larger. Introduced by Holtzman et al. (2019).

Math:

Sort probabilities: $$ p*{(1)} \geq \dots \geq p*{(V)} $$.
Find smallest $$ m $$ where cumulative sum $$ c*m = \sum*{i=1}^m p*{(i)} \geq p $$.
Renormalize: $$ q_i = \frac{p*{(i)}}{c_m} $$ for $$ i = 1 $$ to $$ m $$.
Sample from $$ q_i $$.

Analogy: Like fishing in a pond, catching only from the area containing 90% of the fish ($$ p=0.9 $$), ignoring sparse regions.

5.2 Example with Calculation
Using our example: apple=0.636, banana=0.234, cat=0.086, dog=0.032, elephant=0.012. Set $$ p=0.9 $$.

Steps:

Sorted: apple (0.636), banana (0.234), cat (0.086).
Cumulative: apple=0.636 (<0.9), apple+banana=0.870 (<0.9), apple+banana+cat=0.956 (≥0.9).
Nucleus: $$ m=3 $$, sum = 0.956.
Renormalized: apple ≈ $$ \frac{0.636}{0.956} \approx 0.665 $$, banana ≈ $$ \frac{0.234}{0.956} \approx 0.245 $$, cat ≈ $$ \frac{0.086}{0.956} \approx 0.090 $$.
Sample: Might select "cat".

Alternate Case: If probs are concentrated, e.g., apple=0.8, banana=0.1, others=0.1 total:

Cumulative: apple=0.8 (<0.9), apple+banana=0.9 (≥0.9).
Nucleus: $$ m=2 $$, sum = 0.9.
Renormalized: apple ≈ $$ \frac{0.8}{0.9} \approx 0.889 $$, banana ≈ $$ \frac{0.1}{0.9} \approx 0.111 $$.

5.3 Real-World Cases

Machine Translation: Top-p ensures fluent translations by adapting to context-specific vocabularies.

Code Generation: Used in tools like GitHub Copilot with $$ p=0.95 $$ for diverse, syntactically correct code.

Scientific Application: In bioinformatics, top-p generates plausible protein sequences for drug design, adapting to distribution confidence.

Pros/Cons:

Pros: Adaptive, better for skewed distributions common in LMs.
Cons: May include more tokens if probabilities are flat, risking incoherence.

5.4 Visualization
Sketch a cumulative probability curve:
Cumulative Probability vs. Tokens (sorted):
Token 1 (apple): 0.64
Token 2 (banana): 0.87
Token 3 (cat): 0.96 ----> Cut at p=0.9
Token 4 (dog): 0.99
Token 5 (elephant): 1.0

Section 6: Comparing Top-k and Top-p
6.1 Key Differences

Top-k: Fixed number of tokens ($$ k $$).
Top-p: Dynamic number based on cumulative probability ($$ p $$).

Aspect
Top-k
Top-p

Selection
Top $$ k $$ tokens
Tokens until cumulative $$ \geq p $$

Adaptivity
Fixed size
Adapts to distribution

Best Use
Uniform probabilities
Skewed probabilities

Risk
Misses good tokens if $$ k $$ too small
Includes junk if probs flat

Analogy: Top-k is like inviting a fixed number of top students to a seminar; top-p invites enough to cover 90% of expertise, adjusting dynamically.

6.2 Math Comparison
For our probs, $$ k=3 $$ and $$ p=0.9 $$ both select apple, banana, cat. But for concentrated probs (apple=0.9, others=0.1 spread):

Top-k=3: Includes apple + two low-prob tokens.
Top-p=0.9: Includes only apple (or apple+banana), more conservative.

Section 7: Advanced Topics for Researchers
7.1 Temperature Parameter
Both methods often use a temperature parameter $$ T $$, scaling logits before softmax: new logits = $$ \frac{z_i}{T} $$.

$$ T > 1 $$: Flatter distribution (more random).

$$ T < 1 $$: Sharper distribution (less random).

Research Tip: Experiment with $$ T=1.2 $$ and top-p for creative scientific writing, or $$ T=0.7 $$ with top-k for precise technical reports.

7.2 Pros, Cons, and When to Choose

Pros of Sampling: Enhances diversity, avoids repetitive loops.
Cons: Non-deterministic, may produce incoherent text if parameters are poorly tuned.
Choosing:
Use top-k for simplicity and controlled diversity.
Use top-p for adaptive quality, especially in modern NLG systems.

7.3 Research Impact

Case Study: Holtzman et al. (2019) introduced top-p in "The Curious Case of Neural Text Degeneration," improving creative NLG. Cite this in your papers.
Your Research: Implement sampling in Python, test on datasets like WikiText, and publish novel findings.

Pseudocode for Top-p:
def top_p_sampling(probs, p):
sorted_probs = sort_descending(probs)
cumul = 0
nucleus = []
for prob in sorted_probs:
nucleus.append(prob)
cumul += prob
if cumul >= p:
break
renorm = [pr / cumul for pr in nucleus]
return sample_from(renorm)

Section 8: Exercises and Next Steps

Exercise: For probs A=0.5, B=0.3, C=0.1, D=0.1, calculate top-k=2 and top-p=0.8.
Answer: Top-k=2: A=$$ \frac{0.5}{0.8}=0.625 $$, B=$$ \frac{0.3}{0.8}=0.375 $$. Top-p=0.8: A=0.5 (<0.8), A+B=0.8 (≥0.8), same renormalization.

Think: How could top-p improve climate report generation (e.g., adapting to uncertain data)?
Experiment: Use Hugging Face’s Transformers library to implement top-k and top-p, test on a small dataset, and analyze coherence vs. diversity.

Visual Summary:

Probability bars for original and renormalized distributions.
Cumulative probability curve for top-p.
Comparison table for top-k vs. top-p.

Conclusion
You've decoded the mechanics of top-k and top-p sampling, much like Turing cracking Enigma. These tools are your instruments for crafting intelligent, creative NLG systems. Experiment, question, and apply them to scientific challenges—your journey as a researcher has taken a leap forward!
