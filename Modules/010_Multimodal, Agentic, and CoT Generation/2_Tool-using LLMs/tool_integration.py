# tool_integration.py
# Module for tool-using LLMs in NLG
# Date: September 20, 2025

"""
Theory: Tool-Using LLMs
----------------------
Tool-using LLMs extend capabilities by calling external functions (e.g., calculators, APIs).
Logic: Parse prompt, select tool, fetch result, integrate into NLG.

Analogy (Tesla-inspired): Like an electrical grid distributing power, LLMs delegate tasks
to tools for precise outputs, ensuring efficiency and accuracy.

Goal: Implement a simple calculator tool and integrate with LLM for NLG.
"""

# Import Libraries
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.tools import Tool
import os

# Configure Environment
# Replace 'your_token_here' with your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"


# Define Tool: Calculator
def calculator(expression):
    """
    Simple calculator tool to evaluate mathematical expressions.
    Warning: Use eval() safely in production.
    """
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"


calc_tool = Tool(
    name="Calculator", func=calculator, description="Evaluates mathematical expressions"
)

# Setup LLM
llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7})
prompt = PromptTemplate(
    input_variables=["query"], template="Answer with tool if needed: {query}"
)


# Tool-Using NLG
def generate_with_tool(query):
    """
    Simulates tool-using LLM by parsing query, calling tool, and generating NLG.
    Example: "What is 5+3? Describe it." -> Compute 5+3, describe result.
    """
    chain = LLMChain(llm=llm, prompt=prompt)
    if "calc" in query.lower() or "+" in query:
        expr = query.split("?")[0].split()[-1]  # Extract expression
        result = calc_tool(expr)
        description = chain.run(f"Describe the number {result}")
        return f"Result: {result}. Description: {description}"
    return chain.run(query)


# Visualization: Tool Selection Flow
from graphviz import Digraph


def plot_tool_flow():
    """
    Visualizes the workflow of tool-using LLM.
    Insight: Clarifies decision-making process.
    """
    dot = Digraph(comment="Tool-Using LLM Workflow")
    dot.node("A", "User Prompt")
    dot.node("B", "LLM Parsing")
    dot.node("C", "Tool Selection")
    dot.node("D", "Tool Execution")
    dot.node("E", "NLG Output")
    dot.edges(["AB", "BC", "CD", "DE"])
    dot.render("tool_flow", format="png", cleanup=True)


# Exercise: Simple Tool Selector
def exercise_tool_selector(query):
    """
    Exercise: Design a basic tool selector.
    Returns tool name based on query content.
    """
    if any(op in query for op in ["+", "-", "*", "/"]):
        return "Calculator"
    return "None"


# Main Execution
if __name__ == "__main__":
    # Test Tool Integration
    query = "What is 5+3? Describe it."
    result = generate_with_tool(query)
    print(result)

    # Visualize
    plot_tool_flow()

    # Exercise
    test_query = "Calculate 10*2"
    tool = exercise_tool_selector(test_query)
    print(f"Exercise 2: Selected Tool for '{test_query}': {tool}")

"""
Research Insight:
- Explore agentic frameworks like ReAct for dynamic tool selection.
- Experiment with real APIs (e.g., OpenWeatherMap) for domain-specific NLG.
"""
