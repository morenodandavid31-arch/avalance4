# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer once
analyzer = SentimentIntensityAnalyzer()

# Helper function to get dataset path
def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "customer_reviews.csv")
    return csv_path

# Function to get sentiment using VADER model
@st.cache_data
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"VADER model error: {e}")
        return "Neutral"

# App title
st.title("🔍 Local Sentiment Analysis Dashboard")
st.write("This is your *local model* powered sentiment analysis app using VADER.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df.head(10)   # remove .head(10) if you want all rows
            st.success("✅ Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Dataset not found. Please check the file path.")
        except Exception as e:
            st.error(f"⚠️ Unexpected error loading dataset: {e}")

with col2:
    if st.button("🔍 Analyze Sentiment"):
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment locally..."):
                    # Use .loc to avoid SettingWithCopyWarning
                    st.session_state["df"].loc[:, "Sentiment"] = st.session_state["df"]["SUMMARY"].apply(get_sentiment)
                    st.success("✅ Sentiment analysis completed!")
            except Exception as e:
                st.error(f"❌ Error during sentiment analysis: {e}")
        else:
            st.warning("⚠️ Please load the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    # Visualization using Plotly if sentiment analysis has been performed
    if "Sentiment" in st.session_state["df"].columns:
        st.subheader(f"📊 Sentiment Breakdown for {product}")

        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        sentiment_order = ['Negative', 'Neutral', 'Positive']
        sentiment_colors = {'Negative': 'red', 'Neutral': 'lightgray', 'Positive': 'green'}

        existing_sentiments = sentiment_counts['Sentiment'].unique()
        filtered_order = [s for s in sentiment_order if s in existing_sentiments]
        filtered_colors = {s: sentiment_colors[s] for s in filtered_order}

        sentiment_counts['Sentiment'] = pd.Categorical(
            sentiment_counts['Sentiment'],
            categories=filtered_order,
            ordered=True
        )
        sentiment_counts = sentiment_counts.sort_values('Sentiment')

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title=f"Distribution of Sentiment Classifications - {product}",
            labels={"Sentiment": "Sentiment Category", "Count": "Number of Reviews"},
            color="Sentiment",
            color_discrete_map=filtered_colors
        )
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)