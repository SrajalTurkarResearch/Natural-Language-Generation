# Comprehensive Learning Tutorial: Symbolic Planners in Neurosymbolic Natural Language Generation

As a beginner starting your journey to become a scientist and researcher, think of this tutorial as your own personal science notebook. It is written in the spirit of great thinkers like Albert Einstein, who used simple thought experiments to picture big ideas; Richard Feynman, who explained hard topics with everyday stories; Marie Curie, who showed us how to test ideas through careful experiments; Alan Turing, who built clear step-by-step logic for computers; and Isaac Newton, who based everything on basic rules we can see and understand. Since you are depending only on this tutorial for your learning, I have made it very detailed. I explain every single idea, word, and term in simple, clear words—no hidden meanings, no complicated jumps. Each part builds on the last, like stacking blocks one by one. The language is easy, like talking to a friend, but it keeps a professional tone, as if I am your professor guiding you in a lab. I use short sentences where possible, define every new term right away, and repeat key points gently to make sure they stick. After reading this, you will fully understand the concepts and not need to look back or search elsewhere. Pause after each part, as Einstein would, and ask yourself: "What if I change this part?" Or, like Curie, think: "How can I try this out myself?" This will help you think like a true scientist.

## 1. Introduction: Getting Ready for Your Learning Journey

### 1.1 Why This Topic Is Important for You as a Future Scientist

The topic is symbolic planners inside neurosymbolic natural language generation. Let me break that down: "Symbolic planners" are tools that use clear rules to plan steps, like a to-do list. "Neurosymbolic" means mixing brain-like learning with rule-based thinking. "Natural language generation" is when computers make text that sounds like human words. This whole area is a new and exciting part of artificial intelligence, or AI, which is the field where we make machines smart.

As someone who wants to be a scientist, learning this will help you create new ideas in AI. For example, it can make AI more trustworthy by fixing mistakes in how machines talk or think. Think of it like Newton's laws of motion meeting Darwin's idea of evolution: the laws give strict rules (that's symbolic), and evolution adds flexible changes (that's neural, or brain-like). This mix solves problems in pure AI systems, like when they make up wrong facts, called "hallucinations."

- **The Reason Behind Focusing on This**: Pure brain-like AI is good at spotting patterns but bad at following strict logic. Rule-based AI is good at logic but not flexible. The mix, neurosymbolic, gives both. This is key for safe AI in areas like medicine or robots.

### 1.2 How This Tutorial Is Set Up and How to Learn from It

We will start from the very beginning and go step by step to advanced ideas, like Feynman teaching physics from simple atoms to big theories. For each idea, I include:

- Theory: What it is and why it works, explained in plain words.
- Analogies: Everyday examples, like comparing to cooking or driving.
- Examples: From easy ones to harder ones, with full details.
- Math: If it fits, with every step shown in calculations.
- Visuals: Pictures or diagrams to help you see the idea.
- Real-world cases: How it's used in life, with references.
- Exercises: Things you can do yourself to practice, like a scientist in a lab.

I also add notes from history to inspire you, like how Turing's ideas from the 1950s led to today's AI. Read slowly, take notes, and try the exercises—this will make you a researcher who can invent new things.

## 2. Basic Building Blocks: Key Ideas in NLG and AI Ways of Thinking

### 2.1 Natural Language Generation (NLG) Explained Fully

Natural language generation, or NLG, is when a computer takes information—like numbers or facts—and turns it into words or sentences that people can read easily, like a story or report. It's not just making words; it's making them make sense and flow well.

- **Full Theory**: NLG works in steps. First, decide what information to include (content determination). Second, organize it into a structure (text planning). Third, pick the right words and grammar (surface realization). There's also aggregation, which means combining similar facts, like saying "apples and oranges" instead of repeating. And referring expressions, like using "it" instead of repeating "the car." The reason for this: Computers need clear steps because they don't naturally understand language like humans do. This ensures the output is accurate and easy to follow.
- **Everyday Analogy**: It's like being a storyteller. You have facts (like ingredients), you plan the story order (recipe steps), and then you tell it in nice words (cooking and serving).
- **Detailed Example**: Suppose you have data about a phone: name is iPhone, cost is $999, features are good camera and long battery. NLG turns this into: "The iPhone costs $999. It has an advanced camera for clear photos and a battery that lasts all day." See how it adds words to make it sound natural?
- **Real-World Use**: In news, companies like the Associated Press use NLG to write quick reports on company earnings from financial numbers. This saves time and reduces errors.

### 2.2 Symbolic AI: Using Clear Rules to Represent and Solve Problems

Symbolic AI is a way of making computers think by using symbols—things like words, numbers, or signs—and following exact rules, much like math equations or logic puzzles.

- **History to Understand Better**: It started in the 1950s with people like Allen Newell and Herbert Simon, who made a program called Logic Theorist to prove math theorems using rules. Later, in the 1970s, systems like MYCIN used rules to diagnose diseases. Alan Turing's idea of a machine that follows symbol rules influenced all this.
- **Full Reason Why It Works**: It stores knowledge as facts and rules in a "knowledge base," like a library. Then, an "inference engine" applies rules to make decisions. This is clear because you can follow each step, like checking a math problem. Good points: Easy to explain and fix. Bad points: It doesn't handle surprises well, and collecting all rules is hard.
- **Math Example with Steps**: Use simple logic called propositional logic. A proposition is a statement like P: "It is raining." A rule: If P, then Q (Q: "Ground is wet"). In numbers, true is 1, false is 0. If P=1, then Q=1. Calculation: Start with P=1, apply rule P→Q, result Q=1. No hidden steps.
- **Everyday Analogy**: Like playing chess. Symbols are pieces, rules are how they move. You plan based on rules, no guessing.
- **Detailed Example**: In a simple chatbot, rule: If user inputs "What is the weather?", then get data from a weather source and say "Today is sunny." The symbol is the word "weather," matched to a rule.
- **Visual Help**: Picture a flowchart: Start box → If condition → Action box → End. This shows the rule flow.

### 2.3 Neural AI: Learning Patterns from Lots of Examples

Neural AI uses networks that copy how the human brain works, with connected parts called neurons that learn from data.

- **Deeper Explanation**: A neural network has layers: input (data in), hidden (processing), output (answer). It learns by adjusting connections, called weights, using a method called backpropagation. This means calculating errors and fixing them step by step.
- **Math with Full Calculation**: Activation function example: Sigmoid, which squishes numbers to 0-1. Formula: σ(x) = 1 / (1 + e^{-x}), where e is about 2.718. For x=0: e^0=1, 1+1=2, 1/2=0.5. In word prediction: Use softmax to turn scores into probabilities. Scores for "cat" =2, "dog"=1. e^2≈7.39, e^1≈2.72, total≈10.11. P(cat)=7.39/10.11≈0.73, P(dog)=2.72/10.11≈0.27. Pick the highest.
- **Everyday Analogy**: Like a child learning to talk by listening a lot. No rules given, just patterns from examples. Feynman would say: It's like water flowing downhill— it finds the easy path by trial and error.
- **Good and Bad Points Expanded**: Good: Deals with messy data, like blurry images. Bad: Hard to see inside (black box), needs huge data, can learn wrong patterns if data is bad.

### 2.4 Neurosymbolic AI: Combining the Two for Better Results

Neurosymbolic AI joins neural (pattern learning) and symbolic (rule following) to get the strengths of both.

- **Types of Setup**: Loose coupling: Neural does one part, symbolic another, like passing a baton. Tight coupling: Symbols inside neural networks.
- **Reason It Works**: Neural handles real-world mess, like understanding speech. Symbolic adds checks, like a safety net. This makes AI smarter and safer.
- **Thought Experiment Like Einstein**: Imagine you are tiny, riding on data waves (neural sees patterns), but you use a ruler to measure exactly (symbolic rules).
- **Visual**: Diagram shows neural layer feeding symbols to rule layer.

## 3. Symbolic Planning: The Main Tool for Logical Steps

### 3.1 What Planning Means in AI

Planning is finding a list of actions to go from a starting point (initial state) to a desired end (goal state). Symbolic planning uses symbols for states and rules for actions.

- **Different Kinds**: Classical: Everything is certain. Probabilistic: Some chance involved. Temporal: Includes time.
- **Reason Behind It**: It models the world simply, searches for best path, like a map app finding routes.

### 3.2 Main Parts and How to Solve Plans

- **Parts**: State: Current world facts, like "door is closed." Action: What to do, with needs (preconditions) and results (effects). Goal: What you want.
- **Solving Methods**: Breadth-first search (BFS): Checks all options level by level. A\*: Smarter, uses f = g + h, where g is steps so far, h is guess to goal.
  - Full Math Calculation: In blocks game, h = number of wrong-placed blocks. Start: 2 wrong, g=0, f=2. After one move: 1 wrong, g=1, f=2. Keep going until f leads to goal.
- **PDDL Language**: A standard way to write plans. Domain: General rules. Problem: Specific start and goal.
  - Example Code Explained:
    ```
    (define (domain blocks)  ; This sets the world rules
      (:predicates (on ?x ?y) (clear ?x))  ; Facts like x on y, x is clear
      (:action stack :parameters (?x ?y)  ; Action to stack x on y
        :precondition (and (clear ?y) (holding ?x))  ; Must be true first
        :effect (and (on ?x ?y) (clear ?x) (not (holding ?x)) (not (clear ?y))))  ; Changes after
    ```
- **Visual**: Graph with circles (states) connected by arrows (actions).

### 3.3 Full Examples to Learn From

- **Robot Moving**: Use neurosymbolic: Brain-like part understands "go to kitchen," rule part plans steps like "turn left, walk 5 steps."
- **Practice Exercise**: Plan making tea. Start: Kettle empty. Actions: Fill water (need: Tap works), boil (need: Water in). Goal: Hot tea. Write your own list.

## 4. Symbolic Planners Inside NLG: Making Text Follow Logic

### 4.1 Planning for Text Structure

In NLG, symbolic planners treat text parts as actions. They plan how ideas connect, called discourse.

- **Tools Like Schemas**: Use patterns, like Rhetorical Structure Theory (RST): Relations such as "explain more" or "contrast."
- **Math for Best Plan**: Utility = total usefulness - cost. Example: Plan 1: Usefulness 10, cost 3 (length), total 7. Plan 2: 12 - 5 = 7. Pick based on extra rules if tie.

### 4.2 How It Fits in Neurosymbolic NLG

- **Systems**: Like Teriyaki for robot guides: Neural reads human words, symbolic plans the response steps.
- **Real Use**: In complex paths, like NSPS making code from words.

## 5. Advanced Ideas: Problems, Different Types, and What's Next

### 5.1 Common Problems to Know

- **Size Issues**: Too many states make search slow.
- **Connecting Symbols to Real Life**: Called grounding, like linking word "apple" to actual fruit picture.
- **Mixing Challenges**: Neural and symbolic don't always match easily.

### 5.2 Other Types and Add-Ons

- **With Numbers**: Plans that track amounts, like resources. Minimize cost formula.
- **With Senses**: Mix with images or sound.

### 5.3 Ideas for the Future

- Better ways to explain AI decisions, combine more senses. As a future scientist, think: How to use this for fair AI?

## 6. Hands-On Practice: Exercises and Code

### 6.1 Making a Simple Planner in Code

Here is fake code (pseudocode) like Python:

```python
def a_star(start, goal, actions, heuristic):  # Function to find plan
    queue = [(0 + heuristic(start), 0, start, [])]  # List to hold options: f, g, state, path
    while queue:  # Keep going until empty
        f, g, state, path = heapq.heappop(queue)  # Get best option
        if state == goal:  # Found it?
            return path  # Give the steps
        for action in actions:  # Try each action
            if applicable(action, state):  # Can do it?
                new_state = apply(action, state)  # New world
                new_g = g + cost(action)  # Add step cost
                heapq.heappush(queue, (new_g + heuristic(new_state), new_g, new_state, path + [action]))  # Add to list
```

- **Exercise**: Use this for blocks. Start: A on table, B on A. Goal: B on table, A on B. Run mentally or on paper.

### 6.2 Ideas for Your Research

- **Project Suggestion**: Make neurosymbolic NLG for doctor reports. Guess: It could make them 20% more correct. Test it like Curie—with experiments.

## 7. Wrapping Up: Your Next Steps as a Scientist

This tutorial has explained every part in simple words, with all details, so you understand fully. Go back only if you want to review exercises. Like Newton building on past work, use this to create your own ideas. You are now ready to advance in your science career!
