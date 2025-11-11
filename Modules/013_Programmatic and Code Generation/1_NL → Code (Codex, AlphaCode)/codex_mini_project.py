# codex_mini_project.py: Simulating Codex for NL → Code
#
# Hey, scientist-in-training! This script is from Section 17 of the NL → Code tutorial,
# where we mimicked Codex by generating code from a simple natural language prompt.
# Here, we use a rule-based approach (not a full AI) to turn “Calculate sum of a list”
# into Python code. This shows how NL → Code starts with understanding instructions.
#
# Analogy: It’s like a robot chef writing a recipe from your request, “Make soup.”
# Why this matters: In research, Codex automates tasks like data analysis, saving you
# time for experiments (e.g., analyzing lab results).
#
# Run this to see code generation. Copy into your notebook and ask: “How can I improve
# this for my research?”

# Simple prompt
prompt = "Calculate sum of a list"

# Rule-based parsing: Check for keywords
if "sum" in prompt.lower() and "list" in prompt.lower():
    # Generate code template
    code = """
def sum_list(nums):
    return sum(nums)

nums = [1, 2, 3]
print(sum_list(nums))
"""
    # Print and execute
    print("Generated Code:")
    print(code)
    exec(code)
else:
    print("Prompt not recognized. Try 'Calculate sum of a list'.")

# Explanation: The script checks for “sum” and “list”, then generates a function.
# Real-World Use: In biology, you might prompt “Analyze gene counts” to get similar code.
# Notebook Tip: Try a new prompt like “Average a list.” Ask: “What keywords would I need?”
