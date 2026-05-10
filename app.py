import streamlit as st
import tweepy
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI X Sentiment Analyzer",
    page_icon="🚀",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #0f172a;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 35px;
}

.glass-card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.positive-card {
    border-left: 8px solid #00ff99;
}

.negative-card {
    border-left: 8px solid #ff4b4b;
}

.neutral-card {
    border-left: 8px solid #ffd700;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.metric-number {
    font-size: 35px;
    font-weight: bold;
    color: #38bdf8;
}

.metric-text {
    font-size: 18px;
    color: #cbd5e1;
}

.stButton>button {
    width: 100%;
    border-radius: 15px;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
    padding: 12px;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg,#0072ff,#00c6ff);
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown(
    '<div class="main-title">🚀 AI Powered X Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Analyze X/Twitter posts using Machine Learning & NLP</div>',
    unsafe_allow_html=True
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_resources():
    nltk.download("stopwords")
    stop_words = stopwords.words("english")

    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer, stop_words


model, vectorizer, stop_words = load_resources()

# ---------- X API ----------
try:
    BEARER_TOKEN = st.secrets["BEARER_TOKEN"]
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
except Exception:
    client = None

# ---------- SENTIMENT FUNCTION ----------
def predict_sentiment(text):
    text_clean = re.sub("[^a-zA-Z]", " ", text)
    text_clean = text_clean.lower().split()

    text_clean = [
        word for word in text_clean
        if word not in stop_words
    ]

    text_clean = " ".join(text_clean)
    text_vector = vectorizer.transform([text_clean])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]

        negative_prob = probabilities[0]
        positive_prob = probabilities[1]

        if abs(positive_prob - negative_prob) < 0.15:
            return "Neutral"
        elif positive_prob > negative_prob:
            return "Positive"
        else:
            return "Negative"

    prediction = model.predict(text_vector)[0]
    return "Positive" if prediction == 1 else "Negative"


# ---------- INPUT SECTION ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔍 Analyze X User")

    username = st.text_input(
        "Enter X Username",
        placeholder="Example: OpenAI"
    )

    tweet_count = st.slider(
        "Number of posts",
        5,
        10,
        5
    )

    analyze_user = st.button("Analyze User Posts")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("✍️ Analyze Manual Text")

    manual_text = st.text_area(
        "Enter text/post",
        height=170,
        placeholder="Example: I love this AI project. It is very useful!"
    )

    analyze_text = st.button("Analyze Text")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MANUAL TEXT ANALYSIS ----------
if analyze_text:
    if manual_text.strip() == "":
        st.warning("Please enter text.")
    else:
        sentiment = predict_sentiment(manual_text)

        if sentiment == "Positive":
            st.success(f"😊 Sentiment: {sentiment}")
        elif sentiment == "Neutral":
            st.warning(f"😐 Sentiment: {sentiment}")
        else:
            st.error(f"😡 Sentiment: {sentiment}")

# ---------- X USER ANALYSIS ----------
if analyze_user:
    if username.strip() == "":
        st.warning("Please enter username.")

    elif client is None:
        st.error("Bearer Token not found. Add BEARER_TOKEN in Streamlit Secrets.")

    else:
        try:
            with st.spinner("Fetching posts and analyzing sentiment..."):
                user = client.get_user(username=username)

                if user.data is None:
                    st.error("User not found.")
                else:
                    user_id = user.data.id

                    tweets = client.get_users_tweets(
                        id=user_id,
                        max_results=tweet_count,
                        tweet_fields=["created_at"]
                    )

                    if tweets.data is None:
                        st.warning("No posts found.")
                    else:
                        positive_count = 0
                        negative_count = 0
                        neutral_count = 0

                        st.subheader("📌 Post Analysis")

                        for tweet in tweets.data:
                            sentiment = predict_sentiment(tweet.text)

                            if sentiment == "Positive":
                                positive_count += 1
                                card_class = "glass-card positive-card"
                                emoji = "😊"

                            elif sentiment == "Neutral":
                                neutral_count += 1
                                card_class = "glass-card neutral-card"
                                emoji = "😐"

                            else:
                                negative_count += 1
                                card_class = "glass-card negative-card"
                                emoji = "😡"

                            st.markdown(f"""
                            <div class="{card_class}">
                                <h4>{emoji} {sentiment} Sentiment</h4>
                                <p>{tweet.text}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.subheader("📊 Sentiment Dashboard")

                        c1, c2, c3, c4 = st.columns(4)

                        with c1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-number">{len(tweets.data)}</div>
                                <div class="metric-text">Total Posts</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-number">{positive_count}</div>
                                <div class="metric-text">Positive</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-number">{neutral_count}</div>
                                <div class="metric-text">Neutral</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-number">{negative_count}</div>
                                <div class="metric-text">Negative</div>
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
Made with ❤️ by Deepanshu Gautam | MCA AI/ML Project
</div>
""", unsafe_allow_html=True)
