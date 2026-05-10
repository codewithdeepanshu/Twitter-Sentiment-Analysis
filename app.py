import streamlit as st
import tweepy
import pickle
import re
import nltk
from nltk.corpus import stopwords

st.set_page_config(
    page_title="X Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------- CSS FRONTEND ----------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00C2FF;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #CFCFCF;
}
.card {
    background-color: #1E1E2F;
    padding: 20px;
    border-radius: 15px;
    margin: 15px 0;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
.positive {
    border-left: 8px solid #00FF7F;
}
.negative {
    border-left: 8px solid #FF4B4B;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------- API ----------
BEARER_TOKEN = st.secrets["BEARER_TOKEN"]
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ---------- MODEL ----------
@st.cache_resource
def load_files():
    nltk.download("stopwords")
    stop_words = stopwords.words("english")

    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer, stop_words

model, vectorizer, stop_words = load_files()

def predict_sentiment(text):
    text_clean = re.sub("[^a-zA-Z]", " ", text)
    text_clean = text_clean.lower().split()
    text_clean = [word for word in text_clean if word not in stop_words]
    text_clean = " ".join(text_clean)

    text_vector = vectorizer.transform([text_clean])
    prediction = model.predict(text_vector)[0]

    return "Negative" if prediction == 0 else "Positive"

# ---------- FRONTEND ----------
st.markdown('<div class="title">🤖 AI Powered X Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze public opinion from X/Twitter posts using Machine Learning</div>', unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Analyze X User")
    username = st.text_input("Enter X Username", placeholder="Example: OpenAI")

    tweet_count = st.slider("Number of tweets", 5, 10, 5)

    analyze_user = st.button("Analyze User Tweets", use_container_width=True)

with col2:
    st.subheader("✍️ Analyze Manual Text")
    manual_text = st.text_area("Enter any text/post", height=150)

    analyze_text = st.button("Analyze Text", use_container_width=True)

st.divider()

# ---------- MANUAL TEXT ----------
if analyze_text:
    if manual_text.strip() == "":
        st.warning("Please enter text.")
    else:
        sentiment = predict_sentiment(manual_text)

        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")

# ---------- USER TWEETS ----------
if analyze_user:
    if username.strip() == "":
        st.warning("Please enter username.")
    else:
        try:
            with st.spinner("Fetching tweets and analyzing sentiment..."):
                user = client.get_user(username=username)
                user_id = user.data.id

                tweets = client.get_users_tweets(
                    id=user_id,
                    max_results=tweet_count
                )

                if tweets.data:
                    positive_count = 0
                    negative_count = 0

                    for tweet in tweets.data:
                        sentiment = predict_sentiment(tweet.text)

                        if sentiment == "Positive":
                            positive_count += 1
                            css_class = "card positive"
                            emoji = "😊"
                        else:
                            negative_count += 1
                            css_class = "card negative"
                            emoji = "😡"

                        st.markdown(f"""
                        <div class="{css_class}">
                            <h4>{emoji} {sentiment} Sentiment</h4>
                            <p>{tweet.text}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.subheader("📊 Sentiment Summary")
                    c1, c2, c3 = st.columns(3)

                    c1.metric("Total Tweets", len(tweets.data))
                    c2.metric("Positive", positive_count)
                    c3.metric("Negative", negative_count)

                else:
                    st.warning("No tweets found.")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown('<div class="footer">Developed by Deepanshu Gautam | MCA Project</div>', unsafe_allow_html=True)
