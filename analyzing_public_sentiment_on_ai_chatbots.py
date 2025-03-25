## Analyzing Reddit Data

### Intallations
"""

!pip install praw
!pip install nltk
import nltk
nltk.download('vader_lexicon')

"""### Imports"""

import praw
import json
import time
import warnings
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# 👇 Add this to suppress the async warning
warnings.filterwarnings("ignore", category=UserWarning, module='praw')

"""### Initialization"""

# Reddit API credentials
CLIENT_ID = 'client_id'
CLIENT_SECRET = 'client_secret'
USER_AGENT = 'user _agent'

# Keywords to track
KEYWORDS = ['grok', 'chatgpt', 'deepseek', 'gemini']

file_map = {
    'grok': 'reddit_results_grok.jsonl',
    'chatgpt': 'reddit_results_chatgpt.jsonl',
    'deepseek': 'reddit_results_deepseek.jsonl',
    'gemini': 'reddit_results_gemini.jsonl'
}

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Number of posts to scrape
post_limit = 1000

"""### Scraping Reddit"""

# Scraper function
def scrape_reddit_data():
    for keyword in KEYWORDS:
        count = 0
        with open(file_map[keyword], 'w', encoding='utf-8') as f:
            for submission in reddit.subreddit('all').search(keyword, limit=post_limit, time_filter='week'):
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'author': str(submission.author),
                    'created_utc': submission.created_utc,
                    'score': submission.score,
                    'url': submission.url,
                    'comments': []
                }

                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    comment_data = {
                        'id': comment.id,
                        'author': str(comment.author),
                        'text': comment.body,
                        'created_utc': comment.created_utc,
                        'score': comment.score
                    }
                    post_data['comments'].append(comment_data)

                json.dump(post_data, f)
                f.write('\n')
                count += 1

                if count >= post_limit:
                    break

        print(f"Scraped {count} posts for '{keyword}'")

# Run it
scrape_reddit_data()

"""### Performing Sentimental Analysis"""

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize a dictionary to store sentiment counts for each keyword
sentiment_data = defaultdict(lambda: {'positive': defaultdict(int), 'negative': defaultdict(int)})

# Function to process a JSONL file and analyze sentiments for a specific keyword/topic
def process_file(file_path, keyword):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            post = json.loads(line)
            text = post.get('text', '')  # Adjust field name if necessary
            timestamp = post.get('created_utc', None)  # UTC timestamp for the post

            # Run sentiment analysis on the post's text
            sentiment = sia.polarity_scores(text)

            # Classify the sentiment based on compound score and store in the dictionary
            if sentiment['compound'] > 0.05:
                sentiment_data[keyword]['positive'][timestamp] += 1
            elif sentiment['compound'] < -0.05:
                sentiment_data[keyword]['negative'][timestamp] += 1

# Process each file for each keyword/topic
for keyword, file_path in file_map.items():
    process_file(file_path, keyword)

"""### Plotting Data"""

# Convert timestamps to date and sort sentiment data
def prepare_data_for_plotting():
    # Dictionary to store data in DataFrame format
    plot_data = {}

    for keyword, sentiment in sentiment_data.items():
        # Convert timestamp to date and aggregate sentiment counts
        sentiment_df = pd.DataFrame({
            'positive': sentiment['positive'],
            'negative': sentiment['negative']
        })

        # Convert Unix timestamp to datetime
        sentiment_df.index = pd.to_datetime(sentiment_df.index, unit='s')

        # Resample by day, summing the counts for each day
        sentiment_df = sentiment_df.resample('D').sum()

        plot_data[keyword] = sentiment_df

    return plot_data

# Prepare data for plotting
plot_data = prepare_data_for_plotting()

# Create subplots for each keyword
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# List of keywords for plotting
keywords = ['grok', 'chatgpt', 'deepseek', 'gemini']

# Plot the sentiment trends for each keyword
for i, keyword in enumerate(keywords):
    ax = axes[i // 2, i % 2]  # Get the correct subplot

    # Plot positive and negative sentiment trends
    plot_data[keyword].plot(ax=ax, kind='line', title=f'{keyword.capitalize()} Sentiment Trend', marker='o')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Posts')
    ax.legend(['Positive', 'Negative'])
    ax.grid(True)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

"""### Creating Word Cloud"""

from wordcloud import WordCloud

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize sentiment counters for each keyword/topic and store texts for wordcloud generation
sentiment_data = {
    'grok': {'positive': [], 'neutral': [], 'negative': []},
    'chatgpt': {'positive': [], 'neutral': [], 'negative': []},
    'deepseek': {'positive': [], 'neutral': [], 'negative': []},
    'gemini': {'positive': [], 'neutral': [], 'negative': []}
}

# Function to process a JSONL file and analyze sentiments for a specific keyword/topic
def process_file(file_path, keyword):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            post = json.loads(line)
            text = post.get('text', '')  # Adjust field name if necessary

            # Run sentiment analysis on the post's text
            sentiment = sia.polarity_scores(text)

            # Classify the sentiment based on compound score and store texts accordingly
            if sentiment['compound'] > 0.05:
                sentiment_data[keyword]['positive'].append(text)
            elif sentiment['compound'] < -0.05:
                sentiment_data[keyword]['negative'].append(text)
            else:
                sentiment_data[keyword]['neutral'].append(text)

# Process each file for each keyword/topic
for keyword, file_path in file_map.items():
    process_file(file_path, keyword)

# Function to generate word cloud from the given texts
def generate_wordcloud(texts, title, ax):
    # Join all texts into one large string
    text = ' '.join(texts)

    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot the word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14)

# Plot adjacent word clouds (positive and negative) for each keyword
fig, axes = plt.subplots(4, 2, figsize=(16, 18))

for i, (keyword, sentiment_texts) in enumerate(sentiment_data.items()):
    # Generate word cloud for positive sentiment
    generate_wordcloud(sentiment_texts['positive'], f"Positive Sentiment Word Cloud for {keyword.capitalize()}", axes[i][0])

    # Generate word cloud for negative sentiment
    generate_wordcloud(sentiment_texts['negative'], f"Negative Sentiment Word Cloud for {keyword.capitalize()}", axes[i][1])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

"""### Insights

1. ChatGPT had a consistent positive sentiment with small variations and minimal negative sentiment.

2. Grok’s positive sentiment fluctuated but had a spike around March 23 while negative sentiment saw some rises.

3. DeepSeek had low initial engagement but peaked strongly on March 24 in positive sentiment.

4. Gemini’s positive sentiment increased until March 24, then dropped, while negative sentiment stayed low.

## Analyzing Youtube Data

### Installations
"""

!pip install --upgrade google-api-python-client
!pip install google-api-python-client
!pip install nltk --quiet
!pip install pandas

"""### Imports"""

import json
import requests
import time
import os
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from collections import defaultdict
from googleapiclient.discovery import build
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

"""### Initialization"""

API_KEY = 'api_key'
SEARCH_KEYWORDS = ['grok', 'chatgpt', 'deepseek', 'gemini']
MAX_RESULTS = 5000  # Total target results per keyword
RESULTS_PER_PAGE = 50  # YouTube max per page

"""### Scraping Youtube"""

def fetch_youtube_results(keyword, max_results):
    all_results = []
    next_page_token = ''
    total_fetched = 0

    while total_fetched < max_results:
        url = (
            f'https://www.googleapis.com/youtube/v3/search'
            f'?part=snippet&type=video&q={keyword}'
            f'&maxResults={RESULTS_PER_PAGE}&key={API_KEY}'
            f'&pageToken={next_page_token}'
        )

        response = requests.get(url)
        data = response.json()

        if 'items' not in data:
            print(f"Error or limit hit for keyword: {keyword}")
            break

        all_results.extend(data['items'])
        total_fetched += len(data['items'])

        print(f"{keyword} → Fetched {total_fetched} videos")

        # Check for next page token
        next_page_token = data.get('nextPageToken', '')
        if not next_page_token:
            break

        # Delay to avoid quota burn
        time.sleep(0.5)

    # Save to JSONL
    with open(f'youtube_results_{keyword}.jsonl', 'w', encoding='utf-8') as f:
        for item in all_results:
            json.dump(item, f)
            f.write('\n')

    print(f"✅ Done saving {len(all_results)} results for {keyword}")

# Run for each keyword
for keyword in SEARCH_KEYWORDS:
    fetch_youtube_results(keyword, MAX_RESULTS)

"""### Processing youtube data"""

file_map = {
     'grok': 'youtube_results_grok.jsonl',
    'chatgpt': 'youtube_results_chatgpt.jsonl',
    'deepseek': 'youtube_results_deepseek.jsonl',
    'gemini': 'youtube_results_gemini.jsonl'
}

# Function to process each file and return a DataFrame with date-wise sentiment
def process_youtube_sentiment(file_path, keyword):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            video = json.loads(line)

            # Extract relevant information
            snippet = video.get('snippet', {})
            title = snippet.get('title', '')
            description = snippet.get('description', '')
            published = snippet.get('publishedAt', '')

            # If no title, description, or published date, skip the entry
            if not title and not description or not published:
                continue

            text = f"{title} {description}".strip()
            sentiment = sia.polarity_scores(text)
            date = published[:10]  # Extract 'YYYY-MM-DD'

            # Classify sentiment
            if sentiment['compound'] > 0.05:
                label = 'positive'
            elif sentiment['compound'] < -0.05:
                label = 'negative'
            else:
                continue  # skip neutral for trend

            # Append to data list
            data.append({'date': date, 'sentiment': label, 'keyword': keyword})

    # Return the data as a DataFrame
    return pd.DataFrame(data)

# Initialize an empty DataFrame to combine all data
all_data = pd.DataFrame()

# Loop through each keyword and process the corresponding file
for keyword, path in file_map.items():
    print(f"Processing data for keyword: {keyword}...")
    df = process_youtube_sentiment(path, keyword)

    # If df is not empty, concatenate it to the main all_data DataFrame
    if not df.empty:
        all_data = pd.concat([all_data, df], ignore_index=True)

# Convert 'date' column to datetime format
all_data['date'] = pd.to_datetime(all_data['date'])

# Filter for data from the last 7 days
last_week = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
filtered_data = all_data[all_data['date'] >= last_week]

# Check the structure of the combined DataFrame
print(filtered_data.head())

"""### Performing Sentimental Analysis"""

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to process each file and generate sentiment data for the last 7 days
def process_sentiment_last_7_days(file_path, keyword):
    sentiment_counts = {'positive': 0, 'negative': 0}
    seven_days_ago = datetime.now() - timedelta(days=7)

    # Convert seven_days_ago to a naive datetime (no timezone)
    seven_days_ago = seven_days_ago.replace(tzinfo=None)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            post = json.loads(line)
            text = post.get('snippet', {}).get('description', '')  # Adjust field name based on your JSON structure
            timestamp = pd.to_datetime(post.get('snippet', {}).get('publishedAt', ''), errors='coerce')

            # Ensure timestamp is naive (remove timezone info if present)
            if timestamp is not pd.NaT:
                timestamp = timestamp.replace(tzinfo=None)  # Remove timezone info if it's there

            # If the post was made in the last 7 days, analyze sentiment
            if timestamp and timestamp >= seven_days_ago:
                sentiment = sia.polarity_scores(text)

                # Classify sentiment
                if sentiment['compound'] > 0.05:
                    sentiment_counts['positive'] += 1
                elif sentiment['compound'] < -0.05:
                    sentiment_counts['negative'] += 1

    return sentiment_counts

# Create a dictionary to store sentiment counts for each keyword
all_sentiment_data = {}

# Process sentiment for each keyword
for keyword, file_path in file_map.items():
    all_sentiment_data[keyword] = process_sentiment_last_7_days(file_path, keyword)

# Convert sentiment data to DataFrame for easier plotting
df_sentiment = pd.DataFrame(all_sentiment_data).T

# Plot the overall sentiment for each keyword (positive vs negative)
df_sentiment.plot(kind='bar', stacked=False, color=['green', 'red'], figsize=(10, 6))

plt.title('Overall Sentiment in the Last 7 Days for Keywords (YouTube)')
plt.xlabel('Keyword')
plt.ylabel('Number of Posts')
plt.xticks(rotation=0)
plt.legend(['Positive', 'Negative'])
plt.tight_layout()
plt.show()

"""### Plotting Sentiments"""

# Plot setup
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Loop through keywords and plot sentiment
for idx, (keyword, ax) in enumerate(zip(file_map.keys(), axes)):
    subset = filtered_data[filtered_data['keyword'] == keyword]
    if subset.empty:
        ax.set_title(f"No data for {keyword}")
        continue

    # Group and count sentiment by date
    trend = subset.groupby(['date', 'sentiment']).size().unstack(fill_value=0).sort_index()

    # Plot lines
    if 'positive' in trend:
        ax.plot(trend.index, trend['positive'], label='Positive', marker='o', color='green')
    if 'negative' in trend:
        ax.plot(trend.index, trend['negative'], label='Negative', marker='x', color='red')

    ax.set_title(f"📊 Sentiment Trend - {keyword}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Posts")
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

"""### Creating Word Cloud"""

from wordcloud import WordCloud, STOPWORDS

sia = SentimentIntensityAnalyzer()

# File paths for each keyword
file_map = {
    'grok': 'youtube_results_grok.jsonl',
    'chatgpt': 'youtube_results_chatgpt.jsonl',
    'deepseek': 'youtube_results_deepseek.jsonl',
    'gemini': 'youtube_results_gemini.jsonl'
}

seven_days_ago = datetime.now() - timedelta(days=7)
stopwords = set(STOPWORDS)

# Function to collect text based on sentiment
def collect_sentiment_text(file_path):
    positive_text = ''
    negative_text = ''

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            post = json.loads(line)
            desc = post.get('snippet', {}).get('description', '')
            timestamp = pd.to_datetime(post.get('snippet', {}).get('publishedAt', ''), errors='coerce')

            if pd.isna(timestamp):
                continue

            timestamp = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp
            if timestamp < seven_days_ago:
                continue

            sentiment = sia.polarity_scores(desc)
            if sentiment['compound'] > 0.05:
                positive_text += ' ' + desc
            elif sentiment['compound'] < -0.05:
                negative_text += ' ' + desc

    return positive_text, negative_text

# Function to generate word cloud
def generate_wordcloud(text, title, ax):
    if not text.strip():
        ax.set_title(f"{title} (No Data)", fontsize=12)
        ax.axis('off')
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='viridis').generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

# Plot 4x2 grid of word clouds (positive & negative for each keyword)
fig, axes = plt.subplots(4, 2, figsize=(16, 18))

for i, (keyword, file_path) in enumerate(file_map.items()):
    pos_text, neg_text = collect_sentiment_text(file_path)
    generate_wordcloud(pos_text, f"{keyword.capitalize()} - Positive", axes[i][0])
    generate_wordcloud(neg_text, f"{keyword.capitalize()} - Negative", axes[i][1])

plt.tight_layout()
plt.show()

