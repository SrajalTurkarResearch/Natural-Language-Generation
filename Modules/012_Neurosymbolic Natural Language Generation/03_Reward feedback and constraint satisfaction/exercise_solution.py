# exercise_solution.py
# Purpose: Solve an exercise for constraint satisfaction
# Inspired by Newton’s problem-solving and McCarthy’s logic
# Use: Run to test a simple CSP check for NLG


def csp_check(value, max_val=50):
    # Check if value satisfies constraint: <= max_val
    return value <= max_val


# Test with a sample value
test_value = 40
result = csp_check(test_value)
print(f"Constraint check for {test_value}: {result}")

# Explanation for researchers:
# - Simple CSP function to validate text length
# - Can extend to multiple constraints (e.g., keywords, tone)
# - Try testing with value=60 to see failure
# - Next step: Integrate with reward_feedback.py for hybrid system
