# Mini Project: Rule-Based Chit-Chat Bot
# Simulates open-domain conversation with simple rules

import random
import matplotlib.pyplot as plt

# Define Response Dictionary
responses = {
    "hello": ["Hi!", "Hey there!", "Hello!"],
    "how are you": ["Good, you?", "Fine, thanks!", "Doing great!"],
}


# Chit-Chat Bot Function
def chit_bot(input_text):
    key = input_text.lower()
    return random.choice(responses.get(key, ["Let’s chat more!"]))


# Test
print("Mini Project Response:", chit_bot("hello"))

# Visualize Response Distribution
freq = [len(responses[key]) for key in responses]
plt.bar(responses.keys(), freq, color="green")
plt.title("Chit-Chat Response Distribution")
plt.xlabel("Input Triggers")
plt.ylabel("Number of Responses")
plt.show()
