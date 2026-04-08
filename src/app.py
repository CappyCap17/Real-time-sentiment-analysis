import streamlit as st
import pandas as pd
import plotly.express as px
from scraper import fetch_reddit_json
from engine import process_thread, apply_kmeans

# Helper function to convert dataframe to CSV
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.set_page_config(page_title="Reddit Intelligence Dashboard", layout="wide")

# --- DYNAMIC HEADER ---
mode = st.sidebar.radio("Analysis Mode", ["Solo Analysis", "Dual Comparison"])
st.title(f"Reddit Intelligence Engine: {mode}")

# --- INPUT LOGIC ---
if mode == "Solo Analysis":
    urls = [st.text_input("Enter Reddit JSON Link")]
else:
    c1, c2 = st.columns(2)
    urls = [c1.text_input("Thread 1 JSON"), c2.text_input("Thread 2 JSON")]

if st.button("Run Analysis"):
    results = []
    for url in urls:
        if url:
            raw = fetch_reddit_json(url)
            processed = process_thread(raw)
            results.append(apply_kmeans(processed))
        else:
            results.append(pd.DataFrame())

    # --- MAIN ANALYTICS DISPLAY ---
    cols = st.columns(len(results)) if mode == "Dual Comparison" else [st.container()]
    
    for i, df in enumerate(results):
        with (cols[i] if mode == "Dual Comparison" else st.container()):
            if not df.empty:
                st.header(f"Thread {i+1} Analytics")
                
                # A. KEYWORDS
                st.subheader("Top 10 Dominant Keywords")
                all_words = df['clean_text'].str.split(expand=True).stack()
                if not all_words.empty:
                    top_10 = all_words.value_counts().head(10)
                    st.write(", ".join([f"**{w.upper()}**" for w in top_10.index]))
                
                # B. PIE CHART
                st.plotly_chart(px.pie(df, names='tag', hole=0.4, title="Sentiment Distribution",
                                       color='tag',
                                       color_discrete_map={'Happy':'#2ca02c', 'Bad':'#d62728', 'Neutral':'#7f7f7f', 'Warning/Threat':'#ff7f0e'}), use_container_width=True)
                
                # C. K-MEANS CLUSTERS
                df_v = df.copy()
                df_v['sz'] = df_v['ups'].clip(lower=1)
                st.plotly_chart(px.scatter(df_v, x='sentiment', y='ups', color='cluster', size='sz', 
                                           hover_data=['text'], title="Topic Clusters"), use_container_width=True)

                # D. RDA (RAW DATA ANALYSIS - TOP 25%)
                st.subheader("Raw Data Analysis (Upper Quartile)")
                q75 = df['ups'].quantile(0.75)
                rda_df = df[df['ups'] >= q75].sort_values('ups', ascending=False)
                
                st.info(f"Threshold: >={int(q75)} Upvotes. Showing top {len(rda_df)} comments.")
                st.dataframe(rda_df[['ups', 'tag', 'text']], use_container_width=True)
                
                # CSV EXPORT BUTTON
                csv_data = convert_df_to_csv(rda_df)
                st.download_button(
                    label=f"Export Thread {i+1} RDA to CSV",
                    data=csv_data,
                    file_name=f'thread_{i+1}_rda_cleaned.csv',
                    mime='text/csv',
                )
            else:
                st.error(f"No data found for Link {i+1}")

    # --- FINAL SUMMARY SECTION (SOLO & DUAL) ---
    st.divider()
    st.header("Final Intelligence Conclusion")
    
    if mode == "Solo Analysis" and not results[0].empty:
        df = results[0]
        avg_sent = df['sentiment'].mean()
        happy_perc = (len(df[df['tag'] == 'Happy']) / len(df)) * 100
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Thread Metrics")
            summary_table = {
                "Metric": ["Average Sentiment Score", "Happy Content Ratio", "Average Engagement"],
                "Value": [f"{avg_sent:.2f}", f"{happy_perc:.1f}%", f"{df['ups'].mean():.1f}"]
            }
            st.table(pd.DataFrame(summary_table))
        with c2:
            st.write("### Automated Summary")
            sentiment_label = "Positive" if avg_sent > 0.05 else "Negative" if avg_sent < -0.05 else "Neutral"
            st.info(f"Conclusion: This thread exhibits a predominantly **{sentiment_label}** tone. With a Happy Content Ratio of {happy_perc:.1f}%, the community interaction is generally constructive.")

    elif mode == "Dual Comparison" and not results[0].empty and not results[1].empty:
        sent1, sent2 = results[0]['sentiment'].mean(), results[1]['sentiment'].mean()
        health1 = len(results[0][results[0]['tag'] == 'Happy']) / len(results[0])
        health2 = len(results[1][results[1]['tag'] == 'Happy']) / len(results[1])
        
        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.write("### Performance Metrics Comparison")
            comparison_data = {
                "Metric": ["Avg Sentiment Score", "Happy Content %", "Avg Upvotes"],
                "Thread 1": [f"{sent1:.2f}", f"{health1*100:.1f}%", f"{results[0]['ups'].mean():.1f}"],
                "Thread 2": [f"{sent2:.2f}", f"{health2*100:.1f}%", f"{results[1]['ups'].mean():.1f}"]
            }
            st.table(pd.DataFrame(comparison_data))
        with res_c2:
            st.write("### Automated Summary")
            winner = "Thread 1" if health1 > health2 else "Thread 2"
            st.info(f"Conclusion: Based on linguistic analysis, **{winner}** represents a healthier community discussion with higher constructive engagement.")