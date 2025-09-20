# Advanced Self-Refining CoT Agent
# Implements iterative CoT with self-critique and memory buffer.
# Author: A hybrid intellect - Turing's logic, Einstein's intuition, Tesla's invention.

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque

# Load model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Memory: Deque for short-term history (FIFO buffer; max length prevents overflow)
memory = deque(maxlen=5)


def cot_reason(query, iterations=3):
    """
    Perform Chain-of-Thought reasoning with self-refinement.
    Mathematical note: Iterations mimic gradient descent on reasoning quality.
    """
    current_reasoning = f"Query: {query}\nChain-of-Thought: Step 1: "
    for i in range(iterations):
        inputs = tokenizer(current_reasoning, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=100)
        step = tokenizer.decode(out[0], skip_special_tokens=True)
        current_reasoning += (
            step
            + f"\nRefine (iteration {i+1}): Is this logical? Missing facts? From memory: {list(memory)} \nRefined: "
        )
        memory.append(step)  # Update memory for persistence
    return current_reasoning


# Example usage
if __name__ == "__main__":
    query = "Solve: Integral of x^2 dx"  # Example mathematical query
    result = cot_reason(query)
    print(result)
