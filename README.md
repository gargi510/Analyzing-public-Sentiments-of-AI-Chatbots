# 🔍 Real-Time Web Scraping & Sentiment Analysis: ChatGPT, Grok, DeepSeek, Gemini

### 📅 Duration: March 18–25, 2025

## 📌 Project Overview
This project performs **real-time web scraping and sentiment analysis** on trending AI tools—**ChatGPT, Grok, DeepSeek, and Gemini**—by collecting data from **Reddit** and **YouTube** over a 7-day period. The goal is to visualize sentiment trends and derive insights into how public perception fluctuates across different platforms.

## 🚀 Features
- **🔄 Real-time Scraping**: Extracts data using **PRAW** (Reddit API) and **YouTube Data API**.
- **💬 Sentiment Analysis**: Utilizes **VADER (NLTK)** to classify comments as positive, negative, or neutral.
- **📊 Daily Trend Charts**: Line charts showing sentiment shifts over time.
- **☁️ Word Clouds**: Highlights key positive and negative words for better sentiment understanding.
- **🧠 Comparative Insights**: Analyzes variations in sentiment across YouTube and Reddit.

---

## 📦 Data Summary
| Platform | Posts per Keyword | Total Keywords | Time Range        |
|----------|------------------|---------------|------------------|
| **Reddit**  | 250              | 4             | March 18–25, 2025 |
| **YouTube** | 500              | 4             | March 18–25, 2025 |

---

## 📈 Key Insights
### **YouTube Trends:**
- **ChatGPT**: High initial positivity (Mar 18), declined gradually.
- **Grok**: Oscillating sentiment with peaks on Mar 19, 21, 24.
- **DeepSeek**: Low sentiment until Mar 24, then a sharp rise.
- **Gemini**: Steady growth until Mar 24, followed by a drop.

### **Reddit Trends:**
- **ChatGPT**: Consistently positive with minimal negativity.
- **Grok**: Volatile; positive spike on Mar 23.
- **DeepSeek**: Low volume but a strong peak on Mar 24.
- **Gemini**: Growth until Mar 24, followed by a dip.

### **General Observations:**
- **ChatGPT** remains the most discussed AI tool.
- **DeepSeek & Gemini** gained traction towards the end of the analysis period.
- **Reddit** displays steadier sentiment trends, while **YouTube** exhibits more fluctuations.

---

## 🛠 Tech Stack
- **Python (Google Colab)**
- **PRAW** – Reddit Scraping
- **YouTube API / Scraping**
- **NLTK (VADER)** – Sentiment Analysis
- **Matplotlib & WordCloud** – Visualization

---

## 📉 Limitations & Challenges
- **YouTube API quota limits** constrained the amount of data collected.
- **Short Reddit posts** lacked sufficient text for accurate sentiment classification.
- **Time zone differences** may have slightly skewed post date clustering.
- **Real-time dashboard not yet integrated** but planned for future updates.

---

## 📊 Future Enhancements
- **📡 Deploy a real-time dashboard** using **Streamlit** or **Plotly Dash**.
- **📈 Improve visualization** by adding neutral sentiment to line charts.
- **🧠 Enhance analysis** with **topic modeling (LDA)** alongside sentiment analysis.

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 👩‍💻 Author
Built by **Gargi Mishra** – [LinkedIn](https://www.linkedin.com/in/gargi510)

---

## 📌 About
This project scrapes **YouTube** and **Reddit** to analyze public sentiment trends for **ChatGPT, Grok, DeepSeek, and Gemini** using NLP techniques and visualization tools.

---

