# Real-time-sentiment-analysis for
# Reddit Intelligence & Sentiment Engine (RISE)

An end-to-end **Data Engineering Pipeline** designed for deep linguistic analysis and unsupervised topic clustering of Reddit discussions.

## Backend Features
- **Recursive ETL:** Traverses nested JSON structures to capture deep-threaded replies.
- **Unsupervised Learning:** Implements **K-Means Clustering** via **TF-IDF Vectorization** for automated topic discovery.
- **Dynamic Sanitization:** Pre-extraction censorship layer using `better-profanity`.
- **Statistical Filtering:** High-impact data extraction using the **75th Percentile (Q3)** engagement threshold.

## Tech Stack
- **NLP:** VADER, NLTK
- **ML:** Scikit-Learn (K-Means)
- **Data:** Pandas, Requests
- **Visualization:** Plotly, Streamlit

## How to Run
1. Clone the repo: `git clone <your-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch: `streamlit run src/app.py`
