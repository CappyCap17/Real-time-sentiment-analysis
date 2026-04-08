import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from better_profanity import profanity

# Resource Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

profanity.load_censor_words()
analyzer = SentimentIntensityAnalyzer()

STOP_WORDS = set(stopwords.words('english')).union({
    'thats', 'dont', 'cant', 'im', 'ive', 'id', 'get', 'also', 'would', 
    'really', 'one', 'going', 'like', 'even', 'think', 'much', '####',
    'know', 'make', 'could', 'want', 'see', 'got', 'something', 'well', 'back'
})

def extract_comments_recursive(children, processed_list):
    for child in children:
        if child['kind'] == 't1':
            data = child['data']
            body = data.get('body', '')
            ups = data.get('ups', 0)

            if not body or "![gif]" in body or "http" in body or "www." in body:
                continue 
            
            censored_body = profanity.censor(body, "#")
            clean_text = re.sub(r'[^a-zA-Z\s]', '', censored_body).lower().strip()
            words = [w for w in clean_text.split() if w not in STOP_WORDS and len(w) > 3]
            final_text = " ".join(words)

            if final_text:
                score = analyzer.polarity_scores(body)['compound']
                if score >= 0.05: tag = "Happy"
                elif score <= -0.05: tag = "Bad"
                else: tag = "Neutral"

                processed_list.append({
                    'text': censored_body, 
                    'clean_text': final_text,
                    'ups': ups,
                    'sentiment': score,
                    'tag': tag
                })

            replies = data.get('replies')
            if replies and isinstance(replies, dict):
                inner_children = replies.get('data', {}).get('children', [])
                extract_comments_recursive(inner_children, processed_list)

def process_thread(json_data):
    if not json_data or len(json_data) < 2: return pd.DataFrame()
    processed_list = []
    extract_comments_recursive(json_data[1]['data']['children'], processed_list)
    return pd.DataFrame(processed_list)

def apply_kmeans(df, n=3):
    if df.empty or len(df) < n:
        if not df.empty: df['cluster'] = 0
        return df
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['clean_text'])
    df['cluster'] = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(X)
    return df