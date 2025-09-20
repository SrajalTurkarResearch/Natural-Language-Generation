# projects.py
# Mini and major projects for tool-using LLMs in NLG
# Date: September 20, 2025

"""
Theory: Applying Tool-Using LLMs
-------------------------------
Projects demonstrate real-world NLG applications using tools like APIs and datasets.

Mini Project: Weather Report Generator
- Tool: Mock weather API
- Goal: Generate NLG report

Major Project: Scientific Paper Summarizer
- Dataset: arXiv papers
- Tool: Mock PDF parser
- Goal: Summarize abstracts

Analogy (Tesla-inspired): Projects are like inventing AC systems—practical, scalable solutions.
"""

# Import Libraries
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceHub
import os

# Configure Environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"


# Mini Project: Weather Report Generator
def weather_report_generator(city):
    """
    Generates NLG weather report using a mock API.
    Real-world: Replace with OpenWeatherMap API.
    """
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7})
    prompt = PromptTemplate(
        input_variables=["data"], template="Generate a weather report: {data}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Mock API data
    data = {"city": city, "temp": 68, "condition": "cloudy", "humidity": 60}
    report = chain.run(
        f"In {data['city']}, temperature is {data['temp']}°F, {data['condition']}, humidity {data['humidity']}%"
    )
    return report


# Major Project: Scientific Paper Summarizer
def paper_summarizer(abstract):
    """
    Summarizes a scientific abstract using LLM.
    Dataset: Mock arXiv abstract (replace with real API).
    """
    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.5})
    prompt = PromptTemplate(input_variables=["text"], template="Summarize: {text}")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(abstract)


# Visualization: Word Cloud for Summary
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def plot_word_cloud(text):
    """
    Generates a word cloud from summarized text.
    Insight: Highlights key terms in NLG output.
    """
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("word_cloud.png")
    plt.close()


# Main Execution
if __name__ == "__main__":
    # Mini Project
    city = "New York"
    report = weather_report_generator(city)
    print(f"Weather Report: {report}")

    # Major Project
    sample_abstract = "This paper explores quantum entanglement in high-energy physics, demonstrating novel measurement techniques."
    summary = paper_summarizer(sample_abstract)
    print(f"Paper Summary: {summary}")

    # Visualize
    plot_word_cloud(summary)

"""
Research Insight:
- Use real arXiv API (https://arxiv.org/help/api) for major project.
- Explore datasets like C-MTEB for multilingual NLG evaluation.
"""
