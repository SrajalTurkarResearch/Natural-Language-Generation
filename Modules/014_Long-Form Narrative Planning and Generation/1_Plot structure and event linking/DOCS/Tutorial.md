Below is a comprehensive, beginner-friendly tutorial on **Plot Structure and Event Linking in Natural Language Generation (NLG)**, crafted in simple language to ensure you can easily understand every concept without any complex terms or hidden meanings. Since you’re relying solely on this tutorial to advance your journey as a scientist and researcher, I’ve kept all the details, depth, and length from the previous version, but made the language clear and accessible. The tutorial includes theory, examples, analogies, real-world cases, mathematical explanations, visualizations, practical exercises, and tips for your scientific career. It’s structured for note-taking, with every concept explained step-by-step to build your skills in NLG.

---

# Tutorial: Plot Structure and Event Linking in Natural Language Generation (NLG)

## Introduction

Natural Language Generation (NLG) is when a computer writes text that sounds like a human wrote it, using data like facts, numbers, or events. Imagine a computer creating a story, a news article, or a science report—that’s NLG! This tutorial focuses on two key ideas: **plot structure** and **event linking**, which help make computer-written stories or reports clear, exciting, and meaningful.

- **Plot Structure**: The plan or framework that organizes a story, like a map that shows the path from start to finish.
- **Event Linking**: How events in a story connect, like snapping LEGO pieces together to build something complete.

**Why is this important for a scientist?** As a future scientist, you might use NLG to write research summaries, explain experiments, or create stories for simulations (like studying how people act in different situations). Learning plot structure and event linking will help you make computer-written text clear and interesting, which is key for sharing your research or building smart AI systems.

**Analogy**: Plot structure is like a recipe for your favorite smoothie. It tells you the steps (add fruit, blend, pour) to make it perfect. Event linking is like choosing the right ingredients and mixing them in the right order so the smoothie tastes great, not like a random mix of stuff.

**What You’ll Learn**:

1. How plot structure organizes a story.
2. How to connect events to make a story flow.
3. How to use math to plan stories.
4. How to build a simple NLG system with code.
5. Real-world examples and tips to grow as a scientist.

---

## Section 1: Understanding Plot Structure

### 1.1 What is Plot Structure?

Plot structure is the way a story is put together, like the skeleton that holds it up. It arranges events so the story has a clear beginning, middle, and end, making it fun and easy to follow. In NLG, plot structure helps computers write stories or reports that feel natural, not like a random list of facts.

**The Classic Way: Freytag’s Pyramid**
A writer named Gustav Freytag created a popular way to organize stories, called Freytag’s Pyramid. It has five parts:

1. **Exposition**: The start, where you meet the characters, learn where the story happens, and get some background.
   - Example: “Sam, a curious kid, lived in a small town.”
2. **Rising Action**: Things get exciting as problems or events pile up.
   - Example: “Sam heard about a hidden cave with treasure.”
3. **Climax**: The most thrilling part, where something big happens.
   - Example: “Sam found the treasure but faced a scary bear!”
4. **Falling Action**: Things calm down as the big moment gets sorted out.
   - Example: “Sam tricked the bear and took the treasure.”
5. **Resolution**: The end, where everything is wrapped up.
   - Example: “Sam brought the treasure home, and the town celebrated.”

**Other Ways to Organize Stories**:

- **Three-Act Structure**: Like a movie with three parts—Setup (start), Confrontation (middle), Resolution (end). Think of _The Lion King_.
- **Hero’s Journey**: The hero leaves home, faces challenges, and comes back changed. Think of _Harry Potter_.
- **Non-Linear Stories**: Events are told out of order, like a puzzle you figure out, used in movies like _Pulp Fiction_.

**Why Learn Different Ways?** As a scientist, you might build NLG systems for different things. A news report might need a simple structure, while a video game might use a Hero’s Journey. Knowing all these options makes you a versatile researcher.

**Analogy**: Plot structure is like building a sandcastle. The base (exposition) sets it up, the towers and walls (rising action and climax) make it exciting, and the final touches (falling action and resolution) finish it perfectly.

### 1.2 Why Plot Structure Matters in NLG

Without a good plot structure, a computer’s story might be confusing or boring. For example:

- **Bad Story**: “Lily went to the park. A spaceship landed. She ate lunch.” (No connection, so it’s confusing!)
- **Good Story**: “Lily, an adventurous girl, went to the park (exposition). She saw strange lights in the sky (rising action). A spaceship landed with friendly aliens (climax). She helped them fix their ship (falling action). The aliens thanked her, and she went home happy (resolution).”

In NLG, plot structure is like the plan the computer makes before writing the story, so it feels like something a human would write.

### 1.3 Real-World Example: News Reports

An AI called **Heliograf**, used by The Washington Post, writes sports news using plot structure:

- **Exposition**: “The Patriots played the Chiefs on Sunday.”
- **Rising Action**: “The Chiefs scored early, but the Patriots fought back with two touchdowns.”
- **Climax**: “In the final seconds, Brady threw a winning touchdown pass.”
- **Falling Action**: “The Chiefs tried a last play but couldn’t score.”
- **Resolution**: “The Patriots won 34-31, moving up in the playoffs.”

**Picture This**:
Imagine a hill showing how exciting the story gets:

```
Excitement
  ^         Climax (top of the hill)
  |         /\
  |        /  \
  |       /    \
  |      /      \
  |     /        \
  |    /          \
  |   /            \
  |  /              \
  | /                \
  |/                  \
Exposition    Falling Action   Resolution
---------------------------------> Time
```

**For Your Notes**:

- Draw Freytag’s Pyramid and write the five parts.
- Pick a movie or book you like. Write one sentence for each part of the pyramid to describe its story.
- List the other structures (Three-Act, Hero’s Journey, Non-Linear) and one example for each (like a movie or book).

---

## Section 2: All About Event Linking

### 2.1 What is Event Linking?

Event linking is how you connect the events in a story so they make sense together, like linking train cars to form a complete train. In NLG, this means making sure each event leads to the next in a logical way.

**Four Ways to Connect Events**:

1. **Causal Linking**: One event causes the next.
   - Example: “It rained (event 1), so the river flooded (event 2), and people moved to higher ground (event 3).”
2. **Temporal Linking**: Events happen in the order of time.
   - Example: “Jack woke up (event 1), ate breakfast (event 2), then rode his bike to school (event 3).”
3. **Thematic Linking**: Events share the same idea or theme.
   - Example: In a story about bravery, every event shows the character being brave.
4. **Spatial Linking**: Events happen in the same or related places.
   - Example: “Emma explored a forest (event 1), then found a hidden lake in the forest (event 2).”

**Analogy**: Event linking is like a chain of paper clips. Each clip (event) hooks onto the next, making a strong chain (story). If the clips don’t connect, the chain falls apart.

### 2.2 Why Event Linking Matters in NLG

If events aren’t connected, the story feels like a pile of random pieces. Event linking makes sure the computer’s story is clear and fun to read. For a scientist, this helps you:

- Write reports that make sense.
- Create exciting stories for games or simulations.
- Build AI that tells stories like a human would.

**Example**:

- **Disconnected Events**: “Tom studied math. A fire started. He went swimming.”
- **Connected Events**: “Tom studied math late at night (event 1). He left a candle burning, which started a fire (event 2, caused by event 1). He escaped and ran to the lake (event 3, caused by event 2). He went swimming to calm down (event 4, caused by event 3).”

### 2.3 Advanced Ways to Link Events

As a scientist, you can use these cool ideas:

- **Knowledge Graphs**: Like a map where events are dots, and lines show how they connect (e.g., “causes” or “happens next”).
- **Chances (Probabilities)**: Use numbers to decide which event comes next (e.g., there’s a 90% chance finding a map leads to exploring a cave).
- **Language Rules**: Use words like “because” or “then” to make events sound natural when written.

**Real-World Example: Doctor’s Reports**
In a hospital, an NLG system might write a patient report:

- **Causal**: “The patient had a cough (event 1), so doctors did a lung test (event 2), which found a cold (event 3).”
- **Temporal**: “The patient came to the hospital on Monday (event 1), got a test on Tuesday (event 2), and started medicine on Wednesday (event 3).”
- **Thematic**: All events focus on the patient getting better.

**Picture This**:

```
Event 1 (Cough) --> Event 2 (Lung Test) --> Event 3 (Cold Found)
   (Causal)           (Causal)
```

**For Your Notes**:

- Write the four types of event linking (causal, temporal, thematic, spatial).
- Make a short story (3–4 sentences) and label how each event connects (e.g., “This is causal because…”).
- Draw a diagram like the one above for your story.

---

## Section 3: Using Math to Plan Stories

Math can help you design NLG systems by giving you a clear way to organize stories. Let’s use **graph theory** (a math idea about connecting things) and **probabilities** (chances of things happening) to make stories logical.

### 3.1 Stories as a Graph

Imagine a story as a map:

- **Dots (Nodes)**: These are the events, like “Hero finds a treasure.”
- **Lines (Edges)**: These show how events connect, like “Finding a treasure leads to fighting a guard.”
- **Numbers (Weights)**: These show how likely one event leads to another (e.g., 0.9 means it’s very likely).

**Math Explanation**:

- A story is a graph \( G = (V, E) \), where:
  - \( V \) is the list of events (dots).
  - \( E \) is the list of connections (lines) between events.
- Each line has a weight (a number from 0 to 1) to show how strong the connection is.
- A story is a path through the map, like following a trail from dot to dot.

**How Good is the Story?**
You can check how well events connect by averaging the weights of the lines:
\[
\text{Story Quality} = \frac{\text{Sum of weights}}{\text{Number of lines}}
\]
A higher number means the story flows better.

### 3.2 Using Chances

The computer can use chances to pick the next event. For example, if the hero finds a map, there’s a 90% chance they’ll explore a forest.

**Example Story**:
Let’s plan a story with four events:

1. Hero finds a magic wand.
2. Hero learns a spell.
3. Hero fights a wizard.
4. Hero wins and saves the town.

**Graph**:

- Dots: \( V = \{\text{Find wand, Learn spell, Fight wizard, Win}\} \)
- Lines: \( E = \{(\text{Find wand, Learn spell, 0.9}), (\text{Learn spell, Fight wizard, 0.85}), (\text{Fight wizard, Win, 0.8})\} \)

**Chance of the Story**:
Multiply the weights:
\[
\text{Chance} = 0.9 \times 0.85 \times 0.8 = 0.612
\]
This means there’s a 61.2% chance this story makes sense.

**Story Quality**:
Add the weights and divide by the number of lines:
\[
\text{Quality} = \frac{0.9 + 0.85 + 0.8}{3} = 0.85
\]
A score of 0.85 means the story is very smooth!

**Picture This**:

```
Find wand --> Learn spell --> Fight wizard --> Win
   (0.9)         (0.85)         (0.8)
```

### 3.3 Advanced Math: Planning Stories

Scientists use **planning tools** to make complex stories:

- **Big-to-Small Planning**: Start with a big goal (e.g., “save the town”) and break it into smaller steps (e.g., “find a wand,” “fight the wizard”).
- **Rule-Based Planning**: List what needs to happen before and after each event (e.g., “you need a wand to learn a spell”).

**For Your Notes**:

- Write the graph math: \( G = (V, E) \), where \( V \) is events and \( E \) is connections.
- Try the example calculation for a three-event story you make up (pick events, give weights, calculate chance and quality).
- Draw the graph for your story, like the one above.

---

## Section 4: Building a Story-Writing AI

### 4.1 How NLG Works

An NLG system makes text in four steps:

1. **Get Data**: The computer gets facts or events, like a list of what happened in a game.
2. **Plan the Story**: Pick which events to use and put them in order (plot structure).
3. **Connect Events**: Make sure events link logically (event linking).
4. **Write the Text**: Turn the plan into sentences that sound human.

### 4.2 Python Program: Story Maker

Let’s create a Python program to write a science-themed story about a researcher named Mia. This uses plot structure and event linking.

**Code**:

```python
# Simple program to make a story
import random

# List of events for each part of the story
events = {
    "exposition": [
        "Mia, a young scientist, found a dusty book in the lab.",
        "Mia got an email about a secret experiment."
    ],
    "rising_action": [
        "Mia read the book and found a map to a hidden lab.",
        "She traveled to the lab through a stormy night.",
        "Mia met an old scientist who knew about the secret."
    ],
    "climax": [
        "Mia found a robot that could talk.",
        "She faced a rival trying to steal the robot."
    ],
    "falling_action": [
        "Mia hid the robot to keep it safe.",
        "She escaped as the lab started to fall apart."
    ],
    "resolution": [
        "Mia shared the robot with the world.",
        "She went back to her lab, ready for more discoveries."
    ]
}

# How events connect
links = {
    "exposition": {"next": "rising_action", "chance": 0.9},
    "rising_action": {"next": "climax", "chance": 0.85},
    "climax": {"next": "falling_action", "chance": 0.8},
    "falling_action": {"next": "resolution", "chance": 0.9}
}

# Make the story
def make_story():
    story = []
    current_part = "exposition"

    # Pick one event from each part
    while current_part:
        event = random.choice(events[current_part])
        story.append(event)
        if current_part in links:
            current_part = links[current_part]["next"]
        else:
            current_part = None

    # Add words to connect events
    connecting_words = [
        "",  # No word for the start
        "This led to",
        "All of a sudden,",
        "After that,",
        "Finally,"
    ]

    # Put the story together
    final_story = []
    for i, event in enumerate(story):
        final_story.append(f"{connecting_words[i]} {event}")

    return " ".join(final_story).strip()

# Check how smooth the story is
def check_quality():
    total_chance = sum(link["chance"] for link in links.values())
    return total_chance / len(links)

# Show the story
print("Generated Story:")
print(make_story())
print("\nStory Quality:", check_quality())
```

**Sample Output**:

```
Generated Story:
Mia, a young scientist, found a dusty book in the lab. This led to Mia read the book and found a map to a hidden lab. All of a sudden, Mia found a robot that could talk. After that, Mia hid the robot to keep it safe. Finally, Mia shared the robot with the world.

Story Quality: 0.8625
```

**What’s Happening**:

- **Plot Structure**: The program picks one event for each part of Freytag’s Pyramid.
- **Event Linking**: The `links` dictionary makes sure events follow a logical order.
- **Writing**: Connecting words like “This led to” make the story flow.
- **Quality**: The program checks how well events connect using the chances.

**For Your Notes**:

- Copy the code and run it in a Python program (like Jupyter Notebook).
- Change the `events` to make a new story (e.g., a space adventure).
- Try adding new connecting words or changing the chances.

### 4.3 Advanced AI: Using Smart Models

Modern NLG systems use **neural networks** (like GPT, which is like a super-smart writer) to make smoother text:

1. **Plan the Story**: Use a program like the one above to pick events.
2. **Write with AI**: Feed the events to a neural model to turn them into natural sentences.
3. **Link Events**: Use the model to add smart connecting words.

**Research Tip**: Look up “neural story generation” on Google Scholar to find papers like “Plan-and-Write: Towards Better Automatic Story Generation.”

---

## Section 5: Real-World Examples

### 5.1 News Writing

**Example: Heliograf for Sports**
The Washington Post’s Heliograf writes sports news:

- **Plot Structure**: Starts with the teams (exposition), describes key plays (rising action), highlights the big moment (climax), shows what happens next (falling action), and ends with the score (resolution).
- **Event Linking**: Uses time order (first quarter, second quarter) and cause-effect (a score changes the game).
- **Sample**: “The Lakers played the Heat on Tuesday (exposition). LeBron scored 25 points early (rising action). In the final minute, Davis dunked to win (climax). The Heat couldn’t recover (falling action). The Lakers won 112-108 (resolution).”

### 5.2 Video Game Stories

In games, NLG makes stories that change with player choices:

- **Plot Structure**: Often uses the Hero’s Journey (start in a normal world, face challenges, return as a hero).
- **Event Linking**: Links events based on what the player does (e.g., picking up a sword leads to a battle).
- **Sample**: “You found a magic ring in a cave (exposition). You trained to use its power (rising action). You fought a dark knight (climax). The knight was defeated (falling action). Your village celebrated you as a hero (resolution).”

### 5.3 Science Reports

NLG can write experiment reports:

- **Plot Structure**: Introduction (what’s the study), Methods (what you did), Results (what happened), Discussion (what it means).
- **Event Linking**: Uses cause-effect (e.g., “We tested a drug, which helped patients”) and time order (e.g., “Data collected on Day 1, analyzed on Day 2”).
- **Sample**: “We studied a new solar panel (exposition). We tested it in sunny and cloudy weather (rising action). It produced 20% more energy than others (climax). Companies started using it (falling action). The study was shared at a conference (resolution).”

**For Your Notes**:

- Pick one example (news, games, science). Write a short story or report using its plot structure and event linking.
- Look up one NLG system (like Heliograf) and note how it uses these ideas.

---

## Section 6: Tips for Becoming a Scientist

To grow as an NLG researcher, try these:

1. **Use Tools**:
   - Try **Hugging Face** (a website with AI tools) to write text with models like GPT.
   - Use **NLTK** or **spaCy** (Python tools) for simple text tasks.
2. **Read Research**:
   - Find papers on “storytelling AI” or “narrative generation” on arXiv or Google Scholar.
   - Example: Look for “Automated Narrative Generation” by Gatt and Krahmer.
3. **Make a Dataset**:
   - Create a list of events (e.g., in a spreadsheet) with their plot part and connections.
   - Example: Event: “Find a map,” Part: Exposition, Connects to: “Explore cave” (causal).
4. **Check Your Stories**:
   - Use tools like **BLEU** or **ROUGE** to see if your stories are like human-written ones.
   - Use quality scores (like in the Python code) to check how well events connect.

**For Your Notes**:

- Write one tool, one paper, and one dataset idea you want to try.
- Plan a small NLG project (e.g., a program to write weather reports).

---

## Section 7: Practice Exercises

1. **Exercise 1: Analyze a Story**Pick a short story, movie, or news article. List its plot structure (exposition, rising action, etc.) and how events connect (causal, temporal, etc.). Write a one-page summary.
2. **Exercise 2: Update the Python Code**Change the Python code to:

   - Use a new story theme (e.g., detective or fantasy).
   - Add spatial linking (e.g., events in the same castle).
   - Include two rising action events.
     Run it and check if the story is smooth.

3. **Exercise 3: Make a Story Graph**Write a five-event story. Draw a graph with dots (events) and lines (connections). Give each line a number (0 to 1) for how likely it is. Calculate the story’s chance and quality score.
4. **Exercise 4: Design an NLG System**Plan an NLG system for something you like (e.g., book reviews, space reports). Write:

   - The plot structure it will use.
   - How events will connect (e.g., causal or temporal).
   - A sample paragraph it might create.

5. **Exercise 5: Research a Paper**
   Find a paper on storytelling AI (e.g., on arXiv). Summarize how it uses plot structure or event linking. Note one idea you could use in your work.

**For Your Notes**:

- Do at least two exercises.
- Write clear answers and think about what you learned.
- Keep a “Research Ideas” section for ideas from the exercises.

---

## Section 8: Helpful Pictures

Here are drawings to make things clear:

1. **Plot Structure Timeline**:

   ```
   Exposition | Rising Action | Climax | Falling Action | Resolution
   -----------|--------------|--------|---------------|------------
   Meet chars | Get exciting | Big moment | Solve problems | End story
   ```

2. **Event Linking Map**:

   ```
   Event 1 --> Event 2 --> Event 3
   (Causal)    (Temporal)
   ```

3. **Story Flowchart**:

   ```
   Start
     |
   [Exposition]
     |
   [Rising Action]
     |
   [Climax]
     |
   [Falling Action]
     |
   [Resolution]
     |
    End
   ```

**For Your Notes**:

- Copy these drawings into your notebook.
- Use them to map out a story or report you create.

---

## Section 9: Problems and Fixes

You might run into these issues:

1. **Problem**: Stories sound boring or repetitive.
   - **Fix**: Add random event choices or use AI models for creative text.
2. **Problem**: Events don’t connect well.
   - **Fix**: Use graphs or chances to make sure events link logically.
3. **Problem**: The plot feels forced.
   - **Fix**: Try different structures (like Hero’s Journey) or ask friends to read your stories.
4. **Problem**: Big stories are hard to manage.
   - **Fix**: Use planning tools or maps to organize lots of events.

**For Your Notes**:

- List these problems and fixes.
- Think about how you’d solve one in a project.

---

## Section 10: Wrapping Up and Next Steps

Plot structure and event linking are the key to making great stories with NLG. You’ve learned:

- How to organize a story with Freytag’s Pyramid.
- How to connect events with causal, temporal, thematic, and spatial links.
- How to use math (graphs and chances) to plan stories.
- How to build a simple NLG system with Python.
- How real systems like Heliograf use these ideas.

**Next Steps for Your Scientist Journey**:

1. **Code Something**: Build a small NLG project, like a news or story writer.
2. **Read More**: Find papers on “storytelling AI” on arXiv or Google Scholar.
3. **Join Others**: Talk to AI researchers online or at conferences like ACL.
4. **Test Your Work**: Use tools like BLEU or quality scores to check your stories.

**For Your Notes**:

- Write a summary of the main ideas: plot structure, event linking types, math models, and NLG steps.
- Note one question you have (e.g., “How do I make stories more exciting?”) and look it up.
- Make a plan to keep learning NLG (e.g., code one project a month, read one paper a month).

---

This tutorial is your complete guide to plot structure and event linking in NLG, written in simple language to help you become a scientist. Copy it into your notes, run the Python code, do the exercises, and use the pictures to learn. If you need more help or want to explore something specific, let me know, and I’ll give you more examples or explanations!

**Current Date and Time**: 10:55 AM IST, Saturday, October 11, 2025.
