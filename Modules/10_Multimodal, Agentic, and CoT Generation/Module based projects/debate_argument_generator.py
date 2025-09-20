# Advanced Debate/Argument Generator
# Multi-agent system: Pro and Con agents debate with shared memory and CoT.
# Author: A hybrid intellect - Turing's logic, Einstein's intuition, Tesla's invention.

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Shared memory (list for episodic storage; advanced: use vector DB for retrieval)
shared_memory = []


def agent_debate(topic, rounds=3):
    """
    Multi-agent debate loop with CoT for each side.
    Mathematical note: Can model as game theory equilibrium search.
    """
    debate_log = f"Topic: {topic}\n"
    for r in range(rounds):
        # Pro agent
        pro_prompt = f"Debate pro: {topic}. CoT: Analyze counterpoints from memory: {shared_memory}. Argument:"
        inputs = tokenizer(pro_prompt, return_tensors="pt")
        pro_out = model.generate(**inputs, max_new_tokens=100)
        pro_arg = tokenizer.decode(pro_out[0], skip_special_tokens=True)
        shared_memory.append(pro_arg)
        debate_log += f"Round {r+1} Pro: {pro_arg}\n"

        # Con agent
        con_prompt = f"Debate con: {topic}. CoT: Refute pro from memory: {shared_memory}. Argument:"
        inputs = tokenizer(con_prompt, return_tensors="pt")
        con_out = model.generate(**inputs, max_new_tokens=100)
        con_arg = tokenizer.decode(con_out[0], skip_special_tokens=True)
        shared_memory.append(con_arg)
        debate_log += f"Round {r+1} Con: {con_arg}\n"
    return debate_log


# Example usage
if __name__ == "__main__":
    topic = "AI will replace humans"
    debate = agent_debate(topic)
    print(debate)
