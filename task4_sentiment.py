import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
data = pd.read_csv("twitter_training.csv", header=None)

# Rename columns
data.columns = ['ID', 'Entity', 'Sentiment', 'Text']

# Keep only needed columns
data = data[['Text']].dropna()

# Function to get sentiment
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
data['Predicted_Sentiment'] = data['Text'].apply(get_sentiment)

# Count sentiments
sentiment_counts = data['Predicted_Sentiment'].value_counts()

print(sentiment_counts)

# Plot Bar Chart
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()