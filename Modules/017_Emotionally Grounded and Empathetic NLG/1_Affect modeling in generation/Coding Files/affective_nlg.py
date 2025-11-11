import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt

# Download VADER lexicon for sentiment analysis
nltk.download("vader_lexicon")

# Initialize sentiment analyzer and text generator
sentiment_analyzer = SentimentIntensityAnalyzer()
text_generator = pipeline("text-generation", model="gpt2", device=-1)


def check_emotion(text):
    """
    Analyze the sentiment of input text using VADER.
    Returns emotion label (happy, sad, neutral) and sentiment scores.
    """
    scores = sentiment_analyzer.polarity_scores(text)
    main_score = scores["compound"]
    if main_score > 0.3:
        return "happy", scores
    elif main_score < -0.3:
        return "sad", scores
    else:
        return "neutral", scores


def write_emotional_text(user_text):
    """
    Generate emotionally appropriate text based on user input sentiment.
    Uses GPT-2 with a prompt tailored to the detected emotion.
    """
    emotion, scores = check_emotion(user_text)

    # Define prompts for different emotions
    emotion_instructions = {
        "happy": "Write in a cheerful and excited way: ",
        "sad": "Write in a kind and supportive way: ",
        "neutral": "Write in a clear and helpful way: ",
    }

    # Generate response
    instruction = emotion_instructions[emotion] + user_text
    response = text_generator(
        instruction, max_length=60, num_return_sequences=1, truncation=True
    )[0]["generated_text"]
    return emotion, scores, response


def show_emotion_graph(text, scores):
    """
    Visualize sentiment scores as a bar plot.
    """
    categories = ["Happy", "Neutral", "Sad", "Overall"]
    values = [scores["pos"], scores["neu"], scores["neg"], scores["compound"]]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=["green", "blue", "red", "purple"])
    plt.title(f"Emotions in: '{text}'")
    plt.ylabel("Score")
    plt.xlabel("Emotion Type")
    plt.show()


if __name__ == "__main__":
    # Test with example inputs
    user_texts = [
        "I’m so excited about my new job!",
        "I’m feeling really sad today.",
        "I need help with my homework.",
    ]

    for text in user_texts:
        emotion, scores, response = write_emotional_text(text)
        print(f"You said: {text}")
        print(f"Emotion: {emotion}")
        print(f"Computer says: {response}\n")
        show_emotion_graph(text, scores)
