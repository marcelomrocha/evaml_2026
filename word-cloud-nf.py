from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

negative_descriptors = [
    "mechanical appearance",
    "weak body expression",
    "emotionless voice",
    "boxy appearance",
    "poor emotional intonation",
    "exaggerated anger",
    "scary expression",
    "exaggerated anger",
    "mechanical facial expression",
    "unclear emotional expression",
    "unclear foot movements",
    "unclear sadness expression",
    "odd eye expression",
    "overly wide smile",
    "exaggerated happiness"
]

frequencies = Counter(negative_descriptors)

wordcloud = WordCloud(
    width=1400,
    height=800,
    background_color="white",
    prefer_horizontal=0.95,
    collocations=False,
    random_state=42
).generate_from_frequencies(frequencies)

plt.figure(figsize=(14, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()