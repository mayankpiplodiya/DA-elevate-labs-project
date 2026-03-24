import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download once
nltk.download('vader_lexicon')

# Load dataset
df = pd.read_csv('/Users/piplodiyam/Desktop/sentiment-project/twitter_training.csv')

# Rename columns if needed
df.columns = ['id', 'entity', 'sentiment', 'tweet']


# 1. Data Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'@\w+', '', text)     # remove mentions
    text = re.sub(r'#\w+', '', text)     # remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special chars
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)


# 2. Sentiment Analysis (NLTK)
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['predicted_sentiment'] = df['clean_tweet'].apply(get_sentiment)


# 3. Sentiment Distribution
sentiment_counts = df['predicted_sentiment'].value_counts()
print(sentiment_counts)

# 4. Visualization (Python)

plt.figure()
sns.countplot(x='predicted_sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 5. Save Processed Data

df.to_csv('processed_twitter_sentiment.csv', index=False)

# 6. Summary Table for Tableau
summary = df.groupby('predicted_sentiment').size().reset_index(name='count')
summary.to_csv('sentiment_summary.csv', index=False)

print("Files saved successfully!")