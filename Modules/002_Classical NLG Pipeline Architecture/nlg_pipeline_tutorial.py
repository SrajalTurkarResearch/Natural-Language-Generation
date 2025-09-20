# nlg_pipeline_tutorial.py
# Comprehensive tutorial on Classical NLG Pipeline Architecture for beginners
# Includes theory, code guides, and visualizations

# Install required libraries: pip install nltk pandas matplotlib
import nltk
import pandas as pd
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# --- Theory Recap ---
"""
Classical NLG Pipeline:
1. Content Determination: Select relevant data.
2. Document Planning: Organize content into a structure.
3. Microplanning: Choose words, tone, sentence structure.
4. Surface Realization: Generate grammatically correct sentences.
5. Post-Processing: Polish text for errors and formatting.
6. Output: Deliver the final text.

Analogy: Like writing a research paper, you select key findings, outline sections,
choose precise words, write sentences, proofread, and publish.

Why It Matters for Scientists:
- Teaches data selection, structuring, and communication.
- Foundation for advanced NLG (e.g., transformers).
"""

# --- Practical Code Guides ---
# Sample weather data
weather_data = {
    "temperature": 75,
    "humidity": 60,
    "condition": "sunny",
    "wind_speed": 10,
}
relevance_scores = {
    "temperature": 0.8,
    "humidity": 0.3,
    "condition": 0.7,
    "wind_speed": 0.2,
}


# Stage 1: Content Determination
def content_determination(data, relevance, threshold=0.5):
    """Select relevant data based on a threshold."""
    selected = {
        key: value for key, value in data.items() if relevance[key] >= threshold
    }
    return selected


# Stage 2: Document Planning
def document_planning(selected_data):
    """Organize content into a logical structure."""
    plan = {
        "introduction": "Provide a general weather statement",
        "body": {
            "temperature": selected_data.get("temperature", "N/A"),
            "condition": selected_data.get("condition", "N/A"),
        },
        "conclusion": "Offer advice based on weather",
    }
    return plan


# Stage 3: Microplanning
def microplanning(doc_plan, tone="casual"):
    """Choose words, tone, and sentence structure."""
    temp = doc_plan["body"]["temperature"]
    cond = doc_plan["body"]["condition"]
    sentence = f"Today's weather is {cond} with a high of {temp}°F."
    conclusion = (
        "Great day for outdoor activities!"
        if tone == "casual"
        else "Suitable conditions for outdoor activities."
    )
    return {"sentence": sentence, "conclusion": conclusion}


# Stage 4: Surface Realization
def surface_realization(microplanned):
    """Generate grammatically correct sentences."""
    sentence = microplanned["sentence"]
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    has_verb = any(tag.startswith("VB") for word, tag in tagged)
    if has_verb:
        return sentence
    return "Error: Sentence lacks a verb."


# Stage 5: Post-Processing
def post_processing(realized, conclusion):
    """Polish text for errors and formatting."""
    polished = realized.replace("°F", " °F").capitalize()
    return f"{polished} {conclusion}"


# Stage 6: Output
def output(final_text):
    """Deliver the final text."""
    print("Generated Output:")
    print(final_text)


# Run the pipeline
selected_data = content_determination(weather_data, relevance_scores)
print("Selected Data:", selected_data)

doc_plan = document_planning(selected_data)
print("Document Plan:", doc_plan)

microplanned = microplanning(doc_plan)
print("Microplanned Content:", microplanned)

realized = surface_realization(microplanned)
print("Realized Sentence:", realized)

final_text = post_processing(realized, microplanned["conclusion"])
print("Final Text:", final_text)

output(final_text)


# --- Visualizations ---
def plot_pipeline():
    """Visualize the NLG pipeline as a flowchart."""
    stages = [
        "Raw Data",
        "Content Determination",
        "Document Planning",
        "Microplanning",
        "Surface Realization",
        "Post-Processing",
        "Output",
    ]
    plt.figure(figsize=(10, 2))
    plt.plot(range(len(stages)), [1] * len(stages), "o-")
    for i, stage in enumerate(stages):
        plt.text(i, 1.05, stage, rotation=45, ha="center")
    plt.ylim(0.5, 1.5)
    plt.axis("off")
    plt.title("Classical NLG Pipeline")
    plt.show()


def plot_document_plan():
    """Visualize the document plan as a tree."""
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.9, "Document", ha="center", fontweight="bold")
    plt.text(0.3, 0.7, "Intro", ha="center")
    plt.text(0.5, 0.7, "Body", ha="center")
    plt.text(0.7, 0.7, "Conclusion", ha="center")
    plt.text(0.4, 0.5, "Temp", ha="center")
    plt.text(0.6, 0.5, "Condition", ha="center")
    plt.plot([0.5, 0.3], [0.9, 0.7], "k-")
    plt.plot([0.5, 0.5], [0.9, 0.7], "k-")
    plt.plot([0.5, 0.7], [0.9, 0.7], "k-")
    plt.plot([0.5, 0.4], [0.7, 0.5], "k-")
    plt.plot([0.5, 0.6], [0.7, 0.5], "k-")
    plt.axis("off")
    plt.title("Document Plan Tree")
    plt.show()


# Run visualizations
plot_pipeline()
plot_document_plan()

# --- Research Directions ---
"""
1. Hybrid Pipelines: Combine classical pipelines with neural models (e.g., fine-tune transformers for microplanning).
2. Evaluation Metrics: Develop metrics beyond BLEU for semantic accuracy in scientific reports.
3. Ethical NLG: Study biases in generated text (e.g., gender bias in sports summaries).
4. Multimodal NLG: Generate text from images or videos (e.g., describe a graph for a research paper).
"""

# --- Rare Insights ---
"""
1. Post-Processing Importance: Often overlooked but critical for trust in scientific applications.
2. Template vs. Free-Form: Classical pipelines rely on templates for reliability but lack flexibility.
3. Human Feedback: Incorporating feedback in microplanning improves tone and readability.
"""

# --- Applications ---
"""
1. Automated Journalism: Generate sports or financial reports (e.g., The Washington Post’s Heliograf).
2. Chatbots: Customer service bots (e.g., Zendesk) use NLG for responses.
3. Medical Reports: Summarize patient data (e.g., Arria NLG).
4. Scientific Communication: Automate research summaries for journals or public outreach.
"""

if __name__ == "__main__":
    print("Classical NLG Pipeline Tutorial Completed!")
