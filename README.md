🔍 Real-Time Web Scraping & Sentiment Analysis: ChatGPT, Grok, DeepSeek, Gemini (Mar 18–25, 2025)

This project performs real-time web scraping and sentiment analysis of trending AI tools—ChatGPT, Grok, DeepSeek, and Gemini—from Reddit and YouTube over a 7-day period (March 18–25, 2025).

It visualizes insights with line charts and word clouds, helping understand public sentiment trends across platforms.
________________________________________

🚀 Features

•	🔄 Real-time scraping using PRAW & YouTube Data API
•	💬 Sentiment Analysis using VADER (NLTK)
•	📊 Daily trend line charts for positive & negative sentiment
•	☁️ Word clouds for positive and negative comments
•	🧠 Comparative insights across platforms
________________________________________

📦 Data Summary

Platform	Posts per Keyword	Total Keywords	Time Range
Reddit	250	4	Mar 18–25, 2025
YouTube	500	4	Mar 18–25, 2025
________________________________________

📈 Key Insights

YouTube Trends:

•	ChatGPT: High initial positivity (Mar 18), declined gradually.
•	Grok: Oscillating sentiment with spikes on Mar 19, 21, 24.
•	DeepSeek: Low until Mar 24, then sharp rise.
•	Gemini: Steady growth until Mar 24, then dropped.

Reddit Trends:

•	ChatGPT: Consistently positive with minimal negativity.
•	Grok: Volatile; positive spike on Mar 23.
•	DeepSeek: Low volume, strong peak on Mar 24.
•	Gemini: Growth until Mar 24, followed by a dip.

General Observations:

•	ChatGPT remains the most discussed.
•	DeepSeek & Gemini gained traction near end of window.
•	Reddit shows steadier sentiment; YouTube has more swings.
________________________________________

🛠 Tech Stack

•	Python (Google Colab)
•	PRAW – Reddit scraping
•	YouTube API / scraping
•	NLTK (VADER) – Sentiment Analysis
•	Matplotlib & WordCloud – Visualization
________________________________________

📉 Limitations & Challenges

•	YouTube scraping was limited by rate limits; avoided quota by not using full API.
•	Some Reddit posts lacked enough text for sentiment classification.
•	Time zone differences may slightly offset exact post date clustering.
•	Real-time dashboard not integrated—yet!
________________________________________

📊 Future Enhancements

•	📡 Deploy a real-time dashboard using:
o	Streamlit or Plotly Dash
o	Deploy on Render, Hugging Face Spaces, or Streamlit Cloud
•	📈 Add neutral sentiment to line charts
•	🧠 Explore topic modeling (LDA) on top of sentiment

________________________________________

📄 License

MIT License
________________________________________

👩‍💻 Author

Built by [Gargi Mishra] – [www.linkedin.com/gargi_510]
________________________________________

