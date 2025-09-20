# Advanced Tool-Using QA Agent
# ReAct agent with tool calls, CoT, and mock tools (e.g., search, calc).
# Author: A hybrid intellect - Turing's logic, Einstein's intuition, Tesla's invention.

from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Load model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


# Mock tools (expand with real APIs; e.g., integrate web search)
def mock_search(query):
    return f"Mock result for {query}: Sample data."


def mock_calc(expr):
    return eval(expr)  # Caution: Unsafe for untrusted input; use sympy for safety


tools = {"search": mock_search, "calc": mock_calc}


def react_loop(query, max_steps=5):
    """
    ReAct loop: Thought → Action → Observation → Refine.
    Mathematical model: Akin to MDP where state includes query and observations.
    """
    thought = f"Query: {query}\nThought: "
    for _ in range(max_steps):
        inputs = tokenizer(thought, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        if "Action:" in response:
            action_match = re.search(r"Action: (\w+)\((.*)\)", response)
            if action_match:
                tool_name, args = action_match.groups()
                result = tools.get(tool_name, lambda x: "Unknown tool")(args)
                thought += f"{response}\nObservation: {result}\nThought: "
        if "Final Answer:" in response:
            return response.split("Final Answer:")[1].strip()
    return "Max steps reached."


# Example usage
if __name__ == "__main__":
    query = "What is 2+2?"
    answer = react_loop(query)
    print(answer)
