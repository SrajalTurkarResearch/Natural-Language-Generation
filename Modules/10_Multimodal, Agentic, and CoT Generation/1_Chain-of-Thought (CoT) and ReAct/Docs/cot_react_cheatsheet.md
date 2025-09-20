# Chain-of-Thought (CoT) and ReAct Cheat Sheet for NLG

This cheat sheet is your quick guide to CoT and ReAct in Natural Language Generation (NLG). It sums up the key ideas, examples, code, and tips from the full tutorial, written in simple words for beginners. Use it to check concepts fast, like a scientist’s quick-reference lab notes. Inspired by Turing’s clear logic, Einstein’s simple explanations, and Tesla’s practical tools, this is your go-to for mastering AI reasoning in your science journey.

## 1. What Are CoT and ReAct?

- **CoT (Chain-of-Thought)** : Makes AI think step by step before answering, like writing out a math problem’s steps. Improves accuracy in NLG (creating text) by 20-50% (e.g., GSM8K math test).
- **ReAct (Reasoning + Acting)** : Combines thinking with actions (like searching a database), looping through: Think → Do → See → Repeat. Reduces wrong facts by 30-40%.

  **Easy Picture (Analogy)** :

- CoT: Like solving a puzzle by listing steps on paper, one after another.
- ReAct: Like a scientist thinking, then using a microscope to check, then thinking again.
  **Sketch** : Draw CoT as a straight line (Question → Step 1 → Step 2 → Answer). Draw ReAct as a circle (Think → Act → See → Back to Think).

## 2. Key Terms Explained

- **LLM (Large Language Model)** : A big computer program trained on tons of text (like books, websites) to write or answer questions. Examples: GPT-4, Grok.
- **Prompt** : The question or instruction you give the AI to start its work.
- **NLG** : Making human-like text, like reports or explanations.
- **Hallucination** : When AI makes up wrong facts. ReAct helps fix this.
- **Benchmark** : A test set, like GSM8K (school math) or ALFWorld (agent tasks), to check AI’s skill.

## 3. How They Work

- **CoT Process** :

1. Ask a question with “think step by step.”
2. AI writes each step (e.g., “First, find the numbers”).
3. AI gives the final answer.

- **ReAct Process** :

1. Think: What do I need? (e.g., “Need a fact.”)
2. Act: Check a tool (e.g., search online).
3. See: Look at the result.
4. Repeat until done, then write the answer.

**Math Behind It (Simple)** :

- CoT: Makes AI more likely to pick the right answer by breaking it into steps. Like: Chance of right answer = Chance of good steps × Chance of answer given steps.
- ReAct: Adds real data to steps, like checking a lab result to make sure.

## 4. Quick Examples

- **CoT Example (Math)** :
  **Question** : Roger has 5 tennis balls, buys 2 cans of 3 balls each. How many total?
  **Steps** : 1. Start with 5 balls. 2. 2 cans × 3 balls = 6. 3. 5 + 6 = 11.
  **Answer** : 11 balls.
- **ReAct Example (Science)** :
  **Question** : What’s water’s boiling point?
  **Loop** : Think: “Need the number.” Act: Check a chemistry book (says 100°C). See: “It’s 100°C.” Answer: “100°C at standard pressure.”

  **Sketch** : For CoT, draw a line: [5 balls?] → [5 + 2×3] → [11]. For ReAct, draw a loop: [Boiling point?] → [Think: Check book] → [Act: Find 100°C] → [Answer].

## 5. Code Snippets (Try These!)

- **CoT Code** (from `cot_example.py`):

```python
def cot_example(problem):
    steps = []
    steps.append('Step 1: Understand - Roger has 5 balls, buys 2 cans of 3 each.')
    new_balls = 2 * 3
    steps.append(f'Step 2: Compute new balls = {new_balls}')
    total = 5 + new_balls
    steps.append(f'Step 3: Total = {total}')
    return '\n'.join(steps) + f'\nAnswer: {total}'
print(cot_example('Roger tennis balls'))
```

**What It Does** : Prints steps like a scientist’s notes, then the answer (11).

- **ReAct Code** (from `react_example.py`):

```python
def react_example(query):
    state = {'query': query, 'steps': []}
    state['steps'].append('Thought: Need boiling point of water.')
    action_result = '100°C'  # Simulate tool
    state['steps'].append(f'Action: Query database → {action_result}')
    state['steps'].append('Thought: At standard pressure, it is 100°C.')
    return '\n'.join(state['steps']) + '\nAnswer: 100°C'
print(react_example('Boiling point of water'))
```

**What It Does** : Shows thinking, acting (fake tool call), and answering.

## 6. Real-World Uses (See `case_studies.md` for Details)

- **Drug Discovery** : ReAct checks chemical databases for drug effects (e.g., 35% fewer errors).
- **Climate Modeling** : CoT explains sea level rise calculations (e.g., 0.5m for 1°C warming).
- **Education** : CoT scores essays, ReAct checks facts (15% better grading).
- **Astronomy** : ReAct uses tools like Astropy for planet data (88% match with experts).

## 7. New Ideas from 2025

- **CoT Variants** :
- **Zero-Shot CoT** : Just say “think step by step.”
- **Few-Shot CoT** : Give 1-5 example answers.
- **Tree-of-Thoughts (ToT)** : Try different paths, like a choose-your-own-adventure.
- **SoftCoT** : Guess steps with chances, like rolling dice (arXiv, May 2025).
- **ReAct Advances** : Multi-agent ReAct (AIs work as a team, Medium 2025); used in Google’s Gemini 2.5 agents.
- **Tip** : New REVEAL benchmark (Google, 2025) checks if CoT steps are correct.

## 8. Common Problems and Fixes

- **CoT Problems** : Too wordy; mistakes grow in long chains (15% worse after 50 steps, 2025 survey). **Fix** : Add checks (AI reviews its steps).
- **ReAct Problems** : Tools can fail (e.g., no internet); risky actions. **Fix** : Use backup data or safe tools.
- **Both** : Can copy biases; use lots of computer power. **Fix** : Check outputs, use small AIs.

## 9. Quick Tips for Scientists

- **Write Clear Prompts** : For CoT, say “step by step.” For ReAct, say “check a tool.”
- **Check Answers** : Like a lab test, verify AI’s text against real data.
- **Draw Pictures** : Sketch CoT as a line, ReAct as a loop, to see the flow.
- **Save Steps** : Write AI’s steps in your notes, like a lab log, for papers.
- **Ethics** : Don’t trust AI blindly; question like Einstein would.

## 10. Practice Problems

- **CoT** : Solve “x² - 5x + 6 = 0.” Steps: 1. Find numbers adding to 5, multiplying to 6 (2 and 3). 2. Roots: x=2, x=3.
- **ReAct** : Find “Carbon-14 half-life.” Think: Need exact time. Act: Check database (5730 years). Answer: 5730 years.
- **Combined** : Check if water is liquid at 50°C. CoT: Boiling point is 100°C, so liquid. ReAct: Confirm 100°C with tool.

## 11. Next Steps for Your Research

- **Try Tools** : Use Grok (grok.com, x.com) or Hugging Face to test prompts.
- **Read Papers** : Wei 2022 (CoT), Yao 2022 (ReAct), arXiv 2502.18600.
- **Build Projects** : Test CoT on math datasets (GSM8K); use ReAct with APIs like PubChem.
- **Draw More** : Use Draw.io for flowcharts to explain your work.
- **Publish** : Share your tests, like Tesla sharing inventions, to grow science.

  **Sketch** : Draw a scientist’s notebook with CoT steps on one page, ReAct loop on another, to keep ideas handy.
