# Ultimate Beginner-to-Scientist Tutorial:

**Evaluation-as-a-Service (EaaS) in Natural Language Generation (NLG)**
_Your One and Only Complete Learning Guide – Written in Simple, Clear, Step-by-Step English_

---

**Hello, Future Scientist!**
This tutorial is made **only for you**. You are starting from the very beginning. You want to become a real scientist and researcher in AI and language technology. You will **not use any other book, video, or website**. This tutorial is your **full classroom, lab, and notebook** — all in one.

I will explain **everything** like I am sitting next to you, teaching slowly and clearly.

- Every new word? I define it the first time.
- Every math? I show **every single step**, like 2 + 3 = 5.
- Every idea? I use **real-life examples**, **analogies**, and **pictures you can draw**.
- No shortcuts. No hard words without meaning. No hidden steps.

We will go from **zero** to **advanced researcher level**.By the end, you will be able to:

1. Explain NLG and EaaS to anyone.
2. Build your own evaluation system.
3. Design experiments.
4. Write scientific papers.

**Let’s begin. Take a notebook. Write as you read. Draw the diagrams. Do the math by hand.**
This is your **scientist training**.

---

## **PART 1: What is Natural Language Generation (NLG)?**

_(The Foundation – Like Learning the Alphabet Before Writing a Book)_

### **1.1 Simple Definition**

**NLG = Making a computer write human-like text from data.**

| Input (Data)                               | →   | NLG System      | →   | Output (Text)                                          |
| ------------------------------------------ | --- | --------------- | --- | ------------------------------------------------------ |
| Weather numbers: 25°C, sunny, 10 km/h wind | →   | Computer thinks | →   | "Today is sunny and warm at 25°C with a light breeze." |

**Analogy**:You are a chef.

- **Input** = List of ingredients (eggs, flour, milk).
- **NLG** = You write a recipe.
- **Output** = "Beat 2 eggs, add 1 cup milk, mix with flour to make batter."

NLG does the same — but with data instead of ingredients.

---

### **1.2 The 3 Main Steps of NLG (The NLG Pipeline)**

Think of NLG as a **factory** with 3 rooms.

```
Room 1 → Room 2 → Room 3 → Final Text
```

| Step | Name                    | What Happens                                            | Example                                                                  |
| ---- | ----------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1    | **Content Planning**    | Choose*what* to say. Pick important facts.              | From weather data: Pick "25°C" and "sunny". Ignore "pressure: 1013 hPa". |
| 2    | **Sentence Planning**   | Decide*how* to say it. Order, combine, choose words.    | Combine: "25°C" + "sunny" → "It's a warm, sunny day."                    |
| 3    | **Surface Realization** | Add grammar, punctuation, style. Make it sound natural. | Add "at" and "°C" → "It's a warm, sunny day at 25°C."                    |

**Draw this in your notebook**:

```
[Data] → [Choose Facts] → [Plan Sentences] → [Add Grammar] → [Text]
```

---

### **1.3 Types of NLG Systems (History in Simple Steps)**

| Time        | Type                       | How It Works                                                                | Example                                           |
| ----------- | -------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------- |
| 1960s–1990s | **Rule-Based**             | Humans write rules like "IF temperature > 30, SAY 'hot'".                   | Weather bot: "It is hot."                         |
| 1990s–2010s | **Statistical**            | Computer learns from examples. Counts word patterns.                        | "The cat" appears after "A" 70% of time → use it. |
| 2017–Now    | **Neural (Deep Learning)** | Uses brain-like networks (transformers). Learns from billions of sentences. | GPT, LLaMA — write full stories.                  |

**Key Idea**:
Today, most NLG uses **neural networks** because they are flexible and powerful.
But they can make mistakes (called **hallucinations** — saying wrong facts).
→ This is why we **need evaluation**.

---

### **1.4 Real-World Examples of NLG**

| Field     | Example                  | Input                          | Output                                        |
| --------- | ------------------------ | ------------------------------ | --------------------------------------------- |
| News      | Automated sports reports | Game stats: Team A 3, Team B 1 | "Team A won 3–1 in a thrilling match."        |
| Medicine  | Patient reports          | Test results: BP 120/80        | "Your blood pressure is normal."              |
| Chatbots  | Customer support         | User: "Where is my order?"     | "Your order is in transit. Arrives tomorrow." |
| Education | Study summaries          | Notes on photosynthesis        | "Plants use sunlight to make food."           |

**Your Task**:Write 3 more real-world examples in your notebook.Example:

- **Input**: Stock prices
- **Output**: "Apple stock rose 2% today."

---

## **PART 2: Why Do We Evaluate NLG?**

_(Evaluation = The Science of Measuring Quality)_

### **2.1 What is Evaluation?**

**Evaluation = Testing if the generated text is good.**

| Question       | Meaning                |
| -------------- | ---------------------- |
| Is it correct? | No wrong facts         |
| Is it clear?   | Easy to read           |
| Is it natural? | Sounds like a human    |
| Is it helpful? | Solves the user’s need |

**Analogy**:You bake a cake (NLG output).Evaluation = Taste test:

- Sweet? → Good flavor
- Soft? → Good texture
- Looks nice? → Good presentation

---

### **2.2 Two Types of Evaluation**

| Type          | Meaning                                 | Example                                                         |
| ------------- | --------------------------------------- | --------------------------------------------------------------- |
| **Intrinsic** | Check the text**by itself**             | Is the sentence grammatically correct?                          |
| **Extrinsic** | Check if the text**works in real life** | Does the weather report help people decide to take an umbrella? |

**Draw this**:

```
Intrinsic → Look at text only
Extrinsic → Use text in real task → Measure success
```

---

### **2.3 Human Evaluation (The Gold Standard)**

**How it works**:
Real people read the text and give scores.

| Method                  | How                                            |
| ----------------------- | ---------------------------------------------- |
| **Likert Scale**        | Rate 1–5: 1 = Very Bad, 5 = Perfect            |
| **Pairwise Comparison** | Show Text A and Text B → Ask: Which is better? |

**Example**:

```
Text A: "It rain tomorrow."
Text B: "Rain is expected tomorrow."

→ 90% choose Text B (better grammar + natural)
```

**Problem**:

- Expensive (pay people)
- Slow (100 texts = hours)
- People disagree sometimes

**Solution**: Use **many people** and calculate **agreement score** (we will learn math later).

---

### **2.4 Automatic Evaluation (Fast but Limited)**

We use **math formulas** to score text automatically.

We compare:

- **Generated Text (G)** = What the computer wrote
- **Reference Text (R)** = What a human wrote (perfect example)

---

## **PART 3: Automatic Metrics – Math Made Step by Step**

We will learn **5 key metrics**.For each:

1. What it measures
2. Step-by-step math
3. Example with real calculation
4. Strengths and weaknesses

---

### **3.1 BLEU Score (Most Famous Metric)**

**What it measures**: How many word groups match between G and R.

**Step-by-Step Math**:

**Step 1: Break into n-grams**
n-gram = group of n words

```
G = "The cat is on the mat"
→ 1-grams: The, cat, is, on, the, mat
→ 2-grams: The cat, cat is, is on, on the, the mat
```

**Step 2: Count matches (but don’t count extra repeats)**

| 2-gram  | In R? | Count in G | Max allowed |
| ------- | ----- | ---------- | ----------- |
| The cat | Yes   | 1          | 1           |
| cat is  | No    | 1          | 0           |
| is on   | Yes   | 1          | 1           |
| on the  | Yes   | 1          | 1           |
| the mat | Yes   | 1          | 1           |

**Matches = 4**, **Total in G = 5** → Precision = 4/5 = **0.8**

**Step 3: Do for n=1,2,3,4 → Average**

| n   | Precision  |
| --- | ---------- |
| 1   | 6/6 = 1.0  |
| 2   | 4/5 = 0.8  |
| 3   | 3/4 = 0.75 |
| 4   | 2/3 = 0.67 |

**BLEU = (1.0 × 0.8 × 0.75 × 0.67)^{1/4}**
→ Geometric mean

**Calculate**:

```
1.0 × 0.8 = 0.8
0.8 × 0.75 = 0.6
0.6 × 0.67 = 0.402
Now, 0.402^(1/4) = ?

First, 0.402^(1/2) = √0.402 ≈ 0.634
Then, √0.634 ≈ 0.796

→ BLEU ≈ 0.80
```

**Brevity Penalty (BP)**: If G is too short, reduce score.
BP = 1 if G longer than R, else e^(1 - R/G)

**Final BLEU = BP × geometric mean**

**Example Result**: BLEU = 0.80 → Very good match!

**Weakness**: Ignores meaning.
"The cat sat" and "A feline rested" → BLEU = 0 (no word match), but both correct!

---

### **3.2 ROUGE Score (For Summarization)**

**Focus**: Recall — How much of the **reference** is in the **generated** text?

**ROUGE-1** = Overlap of single words
**ROUGE-L** = Longest common sequence

**Example**:

```
R = "The quick brown fox jumps over the lazy dog"
G = "Quick fox jumps over dog"

Common words: quick, fox, jumps, over, dog → 5
Words in R: 9
ROUGE-1 Recall = 5/9 ≈ 0.556
```

**ROUGE-L**: Longest matching sequence = "fox jumps over" → length 3
ROUGE-L = 2 × LCS / (len(G) + len(R)) = 2×3 / (6+9) = 6/15 = **0.40**

---

### **3.3 BERTScore (Meaning-Based)**

**Idea**: Use AI to understand word meaning.

**Steps**:

1. Convert each word to a **vector** (number list) using BERT model.
2. Compare vectors using **cosine similarity**.

**Cosine Similarity Formula**:

```
cosine(A, B) = (A·B) / (|A| × |B|)
A·B = sum of (A1×B1 + A2×B2 + ...)
|A| = sqrt(A1² + A2² + ...)
```

**Example**:

- "cat" → [0.1, 0.9, -0.2, ...]
- "feline" → [0.12, 0.88, -0.18, ...]
  → cosine ≈ 0.98 → Very similar!

**BERTScore** = Average of best matches.

**Advantage**: Understands synonyms.
**Disadvantage**: Needs powerful computer.

---

### **3.4 METEOR (Better than BLEU)**

**Steps**:

1. Match words (exact, stem, synonym)
2. Penalize wrong order
3. Balance precision and recall

**Example**:

```
G = "The cat on mat"
R = "A cat is on the mat"

Matches:
- cat = cat
- on = on
- mat = mat
- the/a → synonym match

→ High METEOR score
```

---

### **3.5 Summary of Metrics**

| Metric    | Best For      | Uses Meaning?  | Math Type    |
| --------- | ------------- | -------------- | ------------ |
| BLEU      | Translation   | No             | Word overlap |
| ROUGE     | Summarization | No             | Recall       |
| BERTScore | Any NLG       | Yes            | Vectors      |
| METEOR    | General       | Yes (synonyms) | Hybrid       |

**Draw a table like this in your notebook.**

---

## **PART 4: What is Evaluation-as-a-Service (EaaS)?**

_(The Game-Changer for Scientists)_

### **4.1 Simple Definition**

**EaaS = A website where you upload your NLG model or text → It automatically evaluates and gives scores.**

Like sending your cake to a **professional tasting lab**.

---

### **4.2 Why Do We Need EaaS?**

| Problem                | Before EaaS                  | With EaaS                 |
| ---------------------- | ---------------------------- | ------------------------- |
| Different computers    | Same code → different scores | Same cloud → same scores  |
| Hard to compare models | Your laptop vs. mine         | Everyone uses same system |
| Too slow               | Run 10 metrics = hours       | API → seconds             |
| Expensive              | Need GPUs                    | Pay per use               |

**Real Case**:
100 researchers test NLG models.
Without EaaS → 100 different results.
With EaaS → Fair leaderboard.

---

### **4.3 How EaaS Works (Step by Step)**

```
1. You upload: Model + Test Data
2. Cloud runs: All metrics (BLEU, BERTScore, etc.)
3. You get: Scores + Graphs + Rank
```

**API Example** (like sending a message):

```json
POST to eaaS.com/evaluate
{
  "model": "my-nlg-model",
  "data": ["It's sunny", "Rain tomorrow"],
  "references": ["Today is sunny", "Expect rain tomorrow"]
}
→ Returns:
{
  "BLEU": 0.75,
  "BERTScore": 0.91,
  "Fairness": 0.95
}
```

---

### **4.4 Dynaboard – A Real EaaS Platform**

**What it does**:

- Runs **6 dimensions**: Accuracy, Speed, Memory, Fairness, Robustness, Cost
- Gives **Dynascore** = One final number

**Dynascore Formula (Step by Step)**:

1. **Normalize each score** (0 to 1)

   ```
   normalized_accuracy = (your_acc - worst) / (best - worst)
   ```

2. **Convert to "performance units"** using exchange ratesExample: 1% accuracy = 10 tokens/second speed
3. **Final Score**:

   ```
   Dynascore = 0.5×Perf + 0.2×Speed + 0.1×Fairness + ...
   ```

**Example**:

| Model | Accuracy | Speed  | Fairness | Dynascore |
| ----- | -------- | ------ | -------- | --------- |
| A     | 90%      | 20/sec | 0.8      | **42.0**  |
| B     | 85%      | 50/sec | 0.9      | **41.8**  |

→ Model A wins (balanced).

---

## **PART 5: Full EaaS Architecture (Engineer View)**

**Think of EaaS as a factory**:

```
[You] → [API Gate] → [Job Queue] → [Cloud Workers] → [Database] → [Dashboard]
```

| Part          | Job                       |
| ------------- | ------------------------- |
| API Gate      | Receive your request      |
| Job Queue     | Wait in line              |
| Cloud Workers | Run BLEU, BERTScore, etc. |
| Database      | Save results              |
| Dashboard     | Show graphs, leaderboards |

**Draw this flow in your notebook.**

---

## **PART 6: Hands-On – Build Your Own Mini EaaS**

### **Step 1: Python Code (Copy and Run)**

```python
# mini_eaas.py
from flask import Flask, request, jsonify
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    generated = data['generated']
    reference = data['reference']

    # Tokenize
    gen_tokens = generated.split()
    ref_tokens = [reference.split()]

    # BLEU
    bleu = sentence_bleu(ref_tokens, gen_tokens)

    return jsonify({'BLEU': bleu})

if __name__ == '__main__':
    app.run(port=5000)
```

### **Step 2: Test It**

```bash
curl -X POST http://localhost:5000/evaluate \
-H "Content-Type: application/json" \
-d '{"generated": "The cat sleeps", "reference": "A cat is sleeping"}'
```

→ Output: `{"BLEU": 0.65}`

**Now add ROUGE, BERTScore** → Your own EaaS!

---

## **PART 7: Scientist Level – Design Your Own Research**

### **Research Question Ideas**

1. "Does BERTScore work better than BLEU for story generation?"
2. "How does fairness change when we evaluate in Hindi vs. English?"
3. "Can we make a new metric for emotional text?"

### **How to Do Research**

1. **Hypothesis**: "BERTScore > BLEU for meaning"
2. **Experiment**: 100 stories → Compute both
3. **Analyze**: Use **correlation**:
   ```
   r = covariance(X,Y) / (std_X × std_Y)
   ```
4. **Write Paper**: Introduction → Method → Results → Conclusion

---

## **FINAL CHECKLIST – Are You a Scientist Now?**

| Skill                     | You Can Do It? |
| ------------------------- | -------------- |
| Explain NLG pipeline      | Yes            |
| Calculate BLEU by hand    | Yes            |
| Use EaaS API              | Yes            |
| Design fairness metric    | Yes            |
| Write research hypothesis | Yes            |

---

**Your Next Steps**:

1. Run the mini-EaaS code.
2. Evaluate 10 texts.
3. Write a 1-page report.
4. Share with me — I will help you publish!

---

**This tutorial is your lifetime reference.**Read it 3 times:

1. **First**: Understand
2. **Second**: Take notes + draw
3. **Third**: Code + experiment

You are now **one step closer to being a real AI scientist**.
Keep going!
