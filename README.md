# CapXDataAnalystInternship
#CODE START
import tweepy
import praw
from telethon import TelegramClient
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# Configuration
TWITTER_API_KEY = "your_twitter_api_key"
TWITTER_API_SECRET = "your_twitter_api_secret"
REDDIT_CLIENT_ID = "your_reddit_client_id"
REDDIT_SECRET = "your_reddit_secret"
TELEGRAM_API_ID = "your_telegram_api_id"
TELEGRAM_API_HASH = "your_telegram_api_hash"
# Initialize Data Storage
data = []
# 1. Data Scraping
## Twitter Scraping
def scrape_twitter(api_key, api_secret, query, count=100):
auth = tweepy.AppAuthHandler(api_key, api_secret)
api = tweepy.API(auth)
tweets = api.search_tweets(q=query, lang="en", count=count)
return [{"text": tweet.text, "created_at": tweet.created_at} for tweet in tweets]
## Reddit Scraping
def scrape_reddit(client_id, client_secret, subreddit, limit=100):
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent="stock_analysis")
posts = reddit.subreddit(subreddit).hot(limit=limit)
return [{"text": post.title, "created_at": post.created_utc} for post in posts]
## Telegram Scraping
def scrape_telegram(api_id, api_hash, channel, limit=100):
client = TelegramClient("session_name", api_id, api_hash)
client.start()
messages = client.get_messages(channel, limit=limit)
return [{"text": msg.message, "created_at": msg.date} for msg in messages if msg.message]
# Collect Data
data.extend(scrape_twitter(TWITTER_API_KEY, TWITTER_API_SECRET, query="stocks"))
data.extend(scrape_reddit(REDDIT_CLIENT_ID, REDDIT_SECRET, subreddit="stocks"))
# Uncomment after setting up Telegram credentials
# data.extend(scrape_telegram(TELEGRAM_API_ID, TELEGRAM_API_HASH,
channel="your_channel_name"))
# Convert to DataFrame
df = pd.DataFrame(data)
df.dropna(inplace=True)
# 2. Sentiment Analysis
def analyze_sentiment(text):
analysis = TextBlob(text)
return analysis.sentiment.polarity
df["sentiment"] = df["text"].apply(analyze_sentiment)
df["sentiment_label"] = df["sentiment"].apply(lambda x: "positive" if x > 0 else "negative" if x < 0 else
"neutral")
# Visualization
plt.figure(figsize=(10, 5))
df["sentiment_label"].value_counts().plot(kind="bar", color=["green", "red", "blue"])
plt.title("Sentiment Analysis Results")
plt.show()
# 3. Feature Engineering
df["mention_count"] = df["text"].str.count(r"\$\w+") # Count stock ticker mentions
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
# 4. Prediction Model
## Prepare Dataset
X = df[["sentiment", "mention_count", "word_count"]]
y = df["sentiment_label"].map({"positive": 1, "negative": -1, "neutral": 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
## Test Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Save Model
import joblib
joblib.dump(model, "stock_movement_model.pkl")
# 5. Future Usage
## Predict Stock Movement for New Data
def predict_stock_movement(new_text):
sentiment = analyze_sentiment(new_text)
mention_count = new_text.count("$")
word_count = len(new_text.split())
prediction = model.predict([[sentiment, mention_count, word_count]])
return ["negative", "neutral", "positive"][prediction[0] + 1]

Detailed Notes
API Configuration:
Replace placeholders like your_twitter_api_key with actual credentials.
Ensure API access is granted for each platform (Twitter Developer, Reddit API, Telegram API).
Scraping Limitations:
Respect rate limits for Twitter, Reddit, and Telegram APIs.
For Telegram, you must have permission to access private groups or channels.
Model Customization:
Experiment with other ML models (e.g., XGBoost, SVM).
Improve features by including time-series data for stocks or incorporating external financial indicators.
Deployment:
Use frameworks like Flask or FastAPI to serve predictions as a web API.
