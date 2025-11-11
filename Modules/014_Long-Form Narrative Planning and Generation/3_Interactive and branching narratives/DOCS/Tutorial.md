# A Beginner-Friendly, Detailed Tutorial on Interactive and Branching Narratives in Natural Language Generation (NLG)

Welcome to this super detailed tutorial! As an aspiring scientist, you're starting an exciting journey to understand **Interactive and Branching Narratives** in **Natural Language Generation (NLG)**. Since you're relying only on this tutorial, I've made it as clear as possible, using simple words, fun analogies, and step-by-step explanations. Every term, concept, and idea is broken down so you can easily take notes and build a strong foundation for your research career. This tutorial is packed with theory, examples, math, visualizations, real-world cases, and hands-on exercises to make you a confident researcher. Let's get started!

---

## Table of Contents

1. What Are Interactive and Branching Narratives?
   - Understanding the Basics
   - Why They Matter for NLG
   - A Quick Look at Their History
   - Where They're Used in Real Life
2. Core Ideas Made Simple
   - What is NLG?
   - How Do Interactive Narratives Work?
   - What Are Branching Narratives?
   - Parts of an Interactive NLG System
3. Math and Computer Science Behind It
   - Using Graphs to Show Story Paths
   - Adding Probabilities to Choices
   - Markov Models for Story Flow
   - Example Math Problem with Solution
4. How to Design Your Own Interactive Story
   - Tips for Creating a Great Story
   - Planning Your Story Step-by-Step
   - Handling User Choices
5. Building an Interactive Story with Code
   - Simple Story Using Python
   - Advanced Story with AI Text Generation
6. Real-Life Examples
   - Video Games with Cool Stories
   - Chatbots That Talk Back
   - Learning Tools
   - Therapy Apps
7. Pictures and Analogies to Help You Understand
   - Drawing Story Maps
   - Fun Analogies to Make It Clear
8. Advanced Stuff to Know
   - Keeping Stories Logical
   - Making Big Stories Work
   - Creating New Story Parts on the Fly
   - Making Stories Personal for Users
9. Challenges and Doing the Right Thing
   - Technical Problems
   - Ethical Issues to Think About
   - Avoiding Unfair Stories
10. Hands-On Practice
    - Exercise 1: Make Your Own Simple Story
    - Exercise 2: Add Random Chances
    - Exercise 3: Try AI Text Generation
11. Ideas for Your Research Career
    - Big Questions Scientists Are Exploring
    - Questions You Could Study
12. Extra Resources to Keep Learning
13. Wrapping Up

---

## 1. What Are Interactive and Branching Narratives?

### Understanding the Basics

Imagine reading a story where _you_ decide what happens next. For example, if you're a knight, do you fight the dragon or run away? This is an **interactive narrative**—a story where your choices change how it unfolds. A **branching narrative** is a special kind where the story splits into different paths, like roads in a forest, leading to different endings.

**Natural Language Generation (NLG)** is when a computer writes text that sounds like a human wrote it. When we mix NLG with interactive narratives, the computer creates story text that changes based on what you choose, making it feel like a real, living story.

**Simple Example:**

- Start: "You're at a fork in the road."
- Choice 1: "Go left" → "You find a treasure chest!"
- Choice 2: "Go right" → "You meet a friendly elf."

### Why They Matter for NLG

Interactive and branching narratives make NLG more exciting because:

- They let users _feel_ part of the story by making choices.
- They create stories that fit what each user wants.
- As a scientist, you can study how computers understand choices, keep stories making sense, and create fun experiences.

**For Your Research Career:** You could study how to make computers write better stories or how users feel when making choices.

### A Quick Look at Their History

- **1970s-1980s:** Early computer games like _Zork_ let players type commands, and the game replied with text.
- **1990s:** Books like _Choose-Your-Own-Adventure_ let readers pick story paths by flipping to different pages.
- **2000s-Today:** Modern AI, like ChatGPT, makes stories that change dynamically based on what you say or do.

### Where They're Used in Real Life

- **Video Games:** Games like _The Witcher 3_ let you choose your path, like saving a village or fighting a monster.
- **Learning Tools:** Stories teach kids history by letting them decide what a character does (e.g., "What would you do as a pirate?").
- **Chatbots:** Customer service bots change their answers based on your questions.
- **Therapy Apps:** Apps help people practice handling tough situations, like staying calm in a stressful story.

---

## 2. Core Ideas Made Simple

### What is NLG?

NLG is when a computer turns data or ideas into sentences that sound natural. For example:

- **Data:** `{weather: sunny, temp: 75}`
- **NLG Output:** "It's a sunny day with a temperature of 75°F."

**In Interactive Stories:** NLG creates text that matches what you choose, like describing a castle if you decide to explore it.

**Three Steps of NLG:**

1. **Plan:** Decide what to say.
2. **Structure:** Put sentences in order.
3. **Write:** Make it sound natural.

### How Do Interactive Narratives Work?

An interactive narrative lets you shape the story. You might:

- Pick from options (e.g., "Go left or right?").
- Type what you want to do (e.g., "Talk to the wizard").
- Get a response from the computer that feels like it fits your choice.

**Example Flow:**

1. Computer shows: "You're in a dark cave."
2. You choose: "Light a torch."
3. Computer says: "The torch reveals a hidden door!"

### What Are Branching Narratives?

A branching narrative is like a tree with branches:

- **Nodes:** Each node is a part of the story, like "You're in a spooky forest."
- **Choices:** Options that lead to new nodes, like "Walk forward" or "Hide."
- **Paths:** The sequence of choices you make to reach an ending.

**Example:**

- Start: "You're at a river."
- Choices:
  - Swim across → "You find a treasure chest."
  - Build a raft → "The raft sinks, and you lose."

### Parts of an Interactive NLG System

1. **Story Structure:** The map of all possible story paths.
2. **Text Generator:** The part that writes the story text.
3. **User Interface:** How you choose options (buttons, typing, or voice).
4. **Memory Tracker:** Remembers your choices (e.g., if you picked up a key).
5. **Logic Keeper:** Makes sure the story stays sensible (e.g., you can't use a key you didn't find).

**Note for Your Notes:** Draw a diagram showing these 5 parts connected like a machine.

---

## 3. Math and Computer Science Behind It

Math helps us understand and build branching narratives like a scientist. Let's break it down.

### Using Graphs to Show Story Paths

Think of a story as a map where each stop (node) is a story moment, and roads (edges) are your choices. This is called a **directed graph**:

- **Nodes:** Story moments (e.g., "You're in a cave").
- **Edges:** Choices that take you to new moments (e.g., "Light a torch" → "See a treasure").

**Simple Example:**

- Node A: "You're in a forest."
- Choices:
  - Edge 1: "Go left" → Node B: "Find a river."
  - Edge 2: "Go right" → Node C: "Meet a wolf."

**For Your Notes:** Sketch this as circles (nodes) connected by arrows (edges).

### Adding Probabilities to Choices

Sometimes, choices have a chance of success or failure, like rolling a die. For example:

- Fighting a monster: 70% chance you win, 30% chance you lose.

**Math Formula (Write This Down):**
Probability of Outcome = (Number of Ways It Can Happen) ÷ (Total Possible Outcomes)

**Example:** If you roll a 6-sided die and want a 4 or 5:

- Ways to win: 2 (4 or 5)
- Total outcomes: 6
- Probability = 2 ÷ 6 = 0.33 (33% chance)

### Markov Models for Story Flow

A **Markov model** is a way to say, "What happens next depends only on where you are now, not what happened before." It's like choosing your next move in a board game based on your current spot.

**Example Transition Table:**
Places: Forest, Cave, Village.
From Forest:

- 50% chance to Cave.
- 50% chance to Village.
  From Cave or Village: Stay there (story ends).

**Math Representation (Copy This Table):**

| From \ To | Forest | Cave | Village |
| --------- | ------ | ---- | ------- |
| Forest    | 0      | 0.5  | 0.5     |
| Cave      | 0      | 1    | 0       |
| Village   | 0      | 0    | 1       |

### Example Math Problem with Solution

**Story:** You're a hero who can:

- Fight a troll (60% win, 40% lose).
- Talk to the troll (80% convince, 20% fail).
- Run away (100% escape).

**Question:** If you pick each option equally (1/3 chance), what's the chance you succeed (win, convince, or escape)?

**Solution (Step-by-Step):**

1. Success = Win fight OR Convince troll OR Escape.
2. Formula:
   P(Success) = (P(Win) × P(Choose Fight)) + (P(Convince) × P(Choose Talk)) + (P(Escape) × P(Choose Run))
3. Plug in numbers:
   P(Success) = (0.6 × 1/3) + (0.8 × 1/3) + (1.0 × 1/3)
4. Calculate:
   = 0.2 + 0.2667 + 0.3333 = 0.8
5. **Answer:** There's an 80% chance you succeed.

**Practice Problem for You:** If Fight = 50% win, Talk = 70% success, Run = 100%, what's P(Success)? (Answer: 0.733)

---

## 4. How to Design Your Own Interactive Story

### Tips for Creating a Great Story

- **Make It Fun:** Use exciting descriptions, like "The wind howls through the dark castle."
- **Give Real Choices:** Choices should change the story, not just lead to the same place.
- **Keep It Logical:** If a character dies, they can't come back later unless it makes sense.
- **Balance Choices:** Don't give too many options (confusing) or too few (boring).

**Quick Checklist (Write This):**
□ Exciting words □ Real choices □ Logical story □ 2-3 choices max

### Planning Your Story Step-by-Step

1. **Pick a Setting:** E.g., a pirate ship.
2. **List Key Moments:** Big story points, like "You find a treasure map."
3. **Add Choices:** What can the user do? (e.g., "Follow the map" or "Ignore it.")
4. **Plan Endings:** Have different endings, like finding treasure or getting caught.

**Example Plan (Copy This Format):**
Moment 1: You're on a pirate ship, and a storm hits.

- Choice 1: Steer through the storm → Moment 2: Reach an island.
- Choice 2: Drop anchor → Moment 3: Ship gets damaged.

Moment 2: On the island, find a treasure map.

- Choice 1: Follow map → Moment 4: Find treasure.
- Choice 2: Explore island → Moment 5: Meet pirates.

Moment 3: Ship is sinking.

- Choice 1: Repair ship → Moment 6: Escape.
- Choice 2: Abandon ship → Moment 7: Lost at sea.

### Handling User Choices

- **Button Choices:** Offer options like "1. Fight" or "2. Run" for easy picking.
- **Typing Choices:** Let users type actions, like "Search the room," but this needs smarter AI to understand.
- **Mix Both:** Use buttons for big choices and typing for small details.

**Pro Tip:** Start with button choices—they're easier to code and test.

---

## 5. Building an Interactive Story with Code

### Simple Story Using Python

Let's make a pirate adventure story. This Python code creates a story where you make choices, and the computer shows what happens next.

**Copy This Code Exactly:**

```python
# Pirate Adventure Interactive Story

# Story map as a dictionary
story = {
    "start": {
        "text": "You're on a pirate ship when a huge storm hits! Waves crash over the deck.",
        "choices": {
            "1": {"text": "Steer through the storm", "next": "island"},
            "2": {"text": "Drop anchor and wait", "next": "damage"}
        }
    },
    "island": {
        "text": "You reach a sunny island and find a treasure map in the sand.",
        "choices": {
            "1": {"text": "Follow the map", "next": "treasure"},
            "2": {"text": "Explore the island", "next": "pirates"}
        }
    },
    "damage": {
        "text": "The storm damages your ship, and it's sinking fast!",
        "choices": {
            "1": {"text": "Try to repair the ship", "next": "escape"},
            "2": {"text": "Abandon ship", "next": "lost"}
        }
    },
    "treasure": {
        "text": "You follow the map and dig up a chest full of gold! You're rich! The end.",
        "choices": {}
    },
    "pirates": {
        "text": "You meet enemy pirates who demand your map. Game over.",
        "choices": {}
    },
    "escape": {
        "text": "You fix the ship just in time and sail away safely. The end.",
        "choices": {}
    },
    "lost": {
        "text": "You swim to a lifeboat but get lost at sea. Game over.",
        "choices": {}
    }
}

def play_story():
    current = "start"
    while True:
        print("\n" + story[current]["text"])
        if not story[current]["choices"]:
            print("The end of your adventure!")
            break
        print("What do you do?")
        for choice_id, choice in story[current]["choices"].items():
            print(f"{choice_id}. {choice['text']}")
        user_choice = input("Type your choice (1, 2, etc.): ")
        if user_choice in story[current]["choices"]:
            current = story[current]["choices"][user_choice]["next"]
        else:
            print("That's not a valid choice. Try again!")

if __name__ == "__main__":
    print("Welcome to the Pirate Adventure!")
    play_story()
```

**How to Use It:**

1. Copy the code into a Python editor (like VS Code or an online tool like Replit).
2. Run the code.
3. Read the story and type numbers (like "1" or "2") to make choices.

**What’s Happening (Breakdown for Notes):**

- The `story` dictionary is like a map of the story, with each part (node) having text and choices.
- The `play_story` function shows the story, gets your choice, and moves to the next part.
- Empty `choices: {}` means "end of story."

**Test It:** Try all 4 possible endings!

### Advanced Story with AI Text Generation

For a fancier story, we can use an AI model like GPT-2 to create text on the fly. This lets users type anything, and the computer makes up what happens next. Here's a simple version.

**Copy This Code:**

```python
# Simple AI-Based Pirate Story (Pseudo-code)
# Note: Needs 'transformers' library for real use (pip install transformers)

# Pretend we have an AI that makes text
def make_text(story_so_far):
    # This is a fake version; real version uses GPT-2
    return f"{story_so_far} Something exciting happens next!"

# Track where we are in the story
story_state = {"place": "pirate ship", "has_map": False}

def play_ai_story():
    current_text = "You're on a pirate ship. A storm is coming. What do you do?"
    while True:
        print("\n" + current_text)
        user_action = input("What do you do? (Type 'quit' to end): ")
        if user_action.lower() == "quit":
            print("Thanks for playing!")
            break
        # Update the story with what you did
        current_text = f"You're on a pirate ship. You choose to {user_action}. What happens next?"
        new_text = make_text(current_text)
        print(new_text)
        current_text = new_text  # Keep going with the new story

if __name__ == "__main__":
    print("Welcome to the AI Pirate Adventure!")
    play_ai_story()
```

**Note:** To make this work for real, install the `transformers` library and use a model like GPT-2. This is a simplified version to show the idea.

**For Your Research:** Study how AI can understand free-text inputs better.

---

## 6. Real-Life Examples

### Video Games with Cool Stories

- **80 Days:** You travel the world in 80 days, choosing where to go, and the game writes descriptions of your adventures.
- **AI Dungeon:** You type what you want to do, and an AI makes up the story as you go.

**Research Idea:** Compare how players feel in button-choice vs. free-text games.

### Chatbots That Talk Back

- **Customer Service:** Bots for banks or stores answer your questions, like "Where's my package?" and change replies based on what you say.
- **Siri or Alexa:** They listen to you and give answers that fit your question.

**Example Flow:**
User: "Track my order"
Bot: "Sure! Order #123 is on its way. Want updates every hour?"

### Learning Tools

- **Duolingo Stories:** You read a story in a new language and pick what characters say, helping you learn.
- **History Sims:** You play as a historical figure, making choices to learn about events.

**Research Question:** Do interactive stories help students remember facts better?

### Therapy Apps

- **Woebot:** A chatbot that helps with mental health by guiding you through story-like exercises.
- **VR Therapy:** Virtual reality apps let you practice handling tough situations, like public speaking, with computer-generated feedback.

**Ethical Note:** Always get user consent for mental health apps.

---

## 7. Pictures and Analogies to Help You Understand

### Drawing Story Maps

A story map shows how choices connect. Here's the pirate story as a tree:

```
Start: Storm on Pirate Ship
├── Choice 1: Steer → Island
│   ├── Choice 1: Follow Map → Treasure (Win)
│   └── Choice 2: Explore → Pirates (Lose)
└── Choice 2: Anchor → Ship Damage
    ├── Choice 1: Repair → Escape (Neutral)
    └── Choice 2: Abandon → Lost at Sea (Lose)
```

**How to Draw (Step-by-Step):**

1. Write "Start" in a circle at the top.
2. Draw 2 lines down to 2 new circles (Island, Damage).
3. From each, draw 2 more lines to ending circles.
4. Label all arrows with choices.

**Tools:** Paper and pen, or free online tools like draw.io.

### Fun Analogies

- **Choose-Your-Own-Adventure Book:** Each page is a story moment, and you flip to different pages based on your choice.
- **Road Trip:** Choices are like picking which road to take at a fork, leading to different towns.
- **Cooking:** Your choices (ingredients) decide what dish (ending) you get.

**Memory Trick:** Think "Story Tree" = branches = choices.

---

## 8. Advanced Stuff to Know

### Keeping Stories Logical

The story needs to make sense. If you find a key, the computer should remember so you can use it later.

**Solution: Use Memory (Code Example):**

```python
memory = {"has_key": True, "met_pirate": False}

if memory["has_key"]:
    print("You unlock the chest!")
else:
    print("The chest is locked.")
```

### Making Big Stories Work

Big stories with lots of paths can get messy. Solutions:

- **Templates:** Pre-written text with blanks: "You find a {item} in the {place}."
- **AI Creation:** Let AI make new paths automatically.

**Research Topic:** How many paths before stories get too complex?

### Creating New Story Parts on the Fly

AI models like GPT-2 can write new story parts as you play, but they need training to stay on track.

**Training Tip:** Feed AI 100 adventure stories first.

### Making Stories Personal

Track what users like (e.g., adventure over talking) and give them more of those choices.

**Example:**
If you always fight → Next story offers more battles.
If you like talking → Next story offers more conversations.

**Code:**

```python
if user_fights > user_talks:
    offer_battle_choice()
```

---

## 9. Challenges and Doing the Right Thing

### Technical Problems

- **Story Logic:** The story might say something that doesn't match earlier events.
- **Big Stories:** Too many paths can slow down the computer.
- **Understanding Users:** If users type freely, the computer might not understand.

**Solutions:**
□ Test every path □ Use simple graphs □ Train AI better

### Ethical Issues

- **Unfair Stories:** AI might use stereotypes if trained on bad data.
- **Emotional Impact:** Scary or sad stories could upset users.
- **Privacy:** Save user choices safely so no one else sees them.

**Your Research Role:** Design fair AI stories.

### Avoiding Unfair Stories

1. Train AI on diverse stories (different cultures, genders).
2. Check text: "Does this sound fair?"
3. Let users report problems.

**Example Check:** Change "sneaky merchant" to "clever merchant."

---

## 10. Hands-On Practice

### Exercise 1: Make Your Own Simple Story

**Task:** Write a space adventure with 5 story moments and 2-3 choices each.
**Steps:**

1. Plan a story (e.g., "You crash on a planet").
2. Use the pirate code as a template to code it.
3. Test every path to make sure it works.

**Deadline:** Complete in 1 hour. **Success:** All paths work!

### Exercise 2: Add Random Chances

**Task:** Change the code so some choices have random outcomes.
**Example Code (Add This):**

```python
import random
if random.random() < 0.7:  # 70% chance
    print("You fix the spaceship!")
else:
    print("The spaceship explodes!")
```

**Test:** Run 10 times, count wins.

### Exercise 3: Try AI Text Generation

**Task:** If you have Python and `transformers`, try the AI code.
**Steps:**

1. `pip install transformers`
2. Replace `make_text()` with real GPT-2.
3. Test typing: "fight the alien"

**Research Note:** Write down what AI gets wrong.

---

## 11. Ideas for Your Research Career

### Big Questions Scientists Are Exploring

1. How do we make super long stories stay logical?
2. Can AI guess what users want to do next?
3. How can we mix text, pictures, and sound in stories?

### Questions You Could Study

1. How can AI learn to write better adventure stories?
2. What makes a story fun for users? (Survey 50 players)
3. How do we stop AI from writing unfair or harmful stories?

**Your First Paper Idea:** "Testing Choice Impact in Interactive Narratives"

**Steps to Publish:**

1. Do Exercise 1
2. Test with 10 friends
3. Write: "5 friends won, 5 lost—choices matter!"
4. Submit to student conference

---

## 12. Extra Resources to Keep Learning

**Books (Start Here):**

- _Interactive Storytelling_ by Andrew Glassner (easy to read).
- _Natural Language Processing with Python_ by Steven Bird (great for coding).

**Tools (Free):**

- **Twine:** Free tool to make interactive stories (no coding).
- **Hugging Face:** For trying AI text generation.
- **Graphviz:** For drawing story maps.

**Online Courses (Free Parts):**

- Coursera's NLP course (first 2 weeks free).
- Stanford's CS224N (free lectures on YouTube).

**Communities:**

- Join Reddit's r/gamedev or r/NLP.
- Check X for posts on #InteractiveStorytelling.

**Weekly Plan:**
Mon: Read 10 pages Wed: Code 1 hour Fri: Try new tool

---

## 13. Wrapping Up

Interactive and branching narratives in NLG are like building a magical story world where users are the heroes. By learning the ideas, math, coding, and real-world uses, you're ready to dive into research.

**Your Action Plan:**
Week 1: Do all 3 exercises
Week 2: Make your own 10-node story
Week 3: Write 1-page research idea
Week 4: Join 1 online community

**Final Reminder:** This is your first big step to becoming a scientist—keep exploring and creating! Every famous researcher started exactly where you are now.

**Quick Review Questions (Answer in Notes):**

1. What's a node? (Answer: Story moment)
2. What's NLG? (Answer: Computer writes human text)
3. P(Success) = ? (Answer: 0.8 from math example)

**Congratulations!** You're now ready to research interactive narratives. Start coding! 🚀
