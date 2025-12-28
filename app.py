import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Mental Health Journal Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOGIN SYSTEM ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("🔐 User Login")
    username = st.text_input("Enter username")
    if st.button("Login"):
        if username.strip():
            st.session_state.user = username
            st.session_state.history = []
            st.rerun()
    st.stop()

# ---------------- MAIN APP ----------------
st.title("🧠 AI Mental Health Journal Analyzer")
st.caption(f"Logged in as **{st.session_state.user}**")

# ---------------- LOAD + TRAIN KNN MODEL ----------------
@st.cache_resource
def train_knn_model():
    dataset = load_dataset("dair-ai/emotion", split="train")
    df = pd.DataFrame(dataset)

    X = df["text"]
    y = df["label"]
    labels = dataset.features["label"].names

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    model.fit(X_vec, y)

    return model, vectorizer, labels

with st.spinner("Training KNN emotion model..."):
    knn_model, vectorizer, label_names = train_knn_model()

sentiment_analyzer = SentimentIntensityAnalyzer()
st.success("KNN Emotion Model Ready ✅")

# ---------------- EMOJI MAP ----------------
emoji_map = {
    "😢": "SADNESS", "😭": "SADNESS",
    "😡": "ANGER", "🤬": "ANGER",
    "😨": "FEAR", "😰": "FEAR",
    "😍": "LOVE", "❤️": "LOVE",
    "😂": "JOY", "😄": "JOY", "😁": "JOY"
}

# ---------------- INPUT ----------------
text = st.text_area(
    "✍️ Write your journal entry (any language + emojis supported):",
    placeholder="Example: आज बहुत उदास हूँ 😢"
)

# ---------------- ANALYSIS ----------------
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please write something.")
    else:
        # ---- Translate ----
        translated = GoogleTranslator(source="auto", target="en").translate(text)

        # ---- Emoji Detection ----
        emoji_emotion = None
        for ch in text:
            if ch in emoji_map:
                emoji_emotion = emoji_map[ch]
                break

        # ---- KNN Prediction ----
        vec = vectorizer.transform([translated])
        pred = knn_model.predict(vec)[0]
        predicted_emotion = label_names[pred].upper()

        final_emotion = emoji_emotion if emoji_emotion else predicted_emotion

        # ---- Sentiment + Severity ----
        sentiment = sentiment_analyzer.polarity_scores(translated)["compound"]

        if sentiment <= -0.5:
            severity = "🔴 High"
        elif sentiment <= -0.1:
            severity = "🟡 Medium"
        else:
            severity = "🟢 Low"

        # ---- Save History ----
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Text": text,
            "Translated": translated,
            "Emotion": final_emotion,
            "Severity": severity
        })

        # ---------------- OUTPUT ----------------
        st.subheader("🧠 Analysis Result")
        st.write(f"**Detected Emotion:** {final_emotion}")
        st.write(f"**Severity Level:** {severity}")
        st.write(f"**Translated Text:** {translated}")

        # ---------------- COPING SUGGESTIONS ----------------
        st.subheader("💡 Coping Suggestions")
        suggestions = {
            "FEAR": "Practice grounding exercises and slow breathing.",
            "SADNESS": "Talk to someone you trust or write your feelings.",
            "ANGER": "Pause, breathe deeply, and avoid reacting immediately.",
            "JOY": "Enjoy the moment and reflect on what made you happy.",
            "LOVE": "Express gratitude and nurture relationships."
        }
        st.info(suggestions.get(final_emotion, "Take care of yourself 💙"))

        # ---------------- DOWNLOAD REPORT ----------------
        report = f"""
AI Mental Health Journal Report
-----------------------------
User: {st.session_state.user}
Original Text:
{text}

Translated Text:
{translated}

Detected Emotion:
{final_emotion}

Severity Level:
{severity}
"""
        st.download_button(
            "📄 Download Report",
            report,
            file_name="mental_health_report.txt"
        )

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.subheader("📈 Journal History")
    dfh = pd.DataFrame(st.session_state.history)
    st.dataframe(dfh)

    st.subheader("📊 Emotion Distribution")
    counts = dfh["Emotion"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    st.pyplot(fig)

# ---------------- LOGOUT ----------------
if st.button("Logout"):
    st.session_state.clear()
    st.rerun()
