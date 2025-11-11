# wordcloud_viz.py: Word cloud for qualitative theme visualization.
# Word clouds help represent themes from qual data (e.g., user feedback on NLG text),
# making abstract concepts tangible—like Tesla diagramming inventions.

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample text representing qual themes (e.g., from interviews or annotations).
text = "NLG AI research qualitative quantitative mixed methods ethics bias generation evaluation"

# Generate word cloud.
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

# Display the word cloud.
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Remove axes for clean visual.
plt.title("Qualitative Themes in NLG Research")
plt.show()

# Researcher Note: In mixed studies, use this to visualize qual data before quant correlation.
