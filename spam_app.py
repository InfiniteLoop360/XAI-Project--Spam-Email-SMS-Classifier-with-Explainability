# spam_app.py
import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer

# ---------------------------
# 1. Load Model Pipeline
# ---------------------------
with open("spam_pipeline.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# ---------------------------
# 2. NLTK Preprocessing Setup
# ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
wnl = WordNetLemmatizer()

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)           # remove emails
    text = re.sub(r'\d+', ' ', text)               # remove numbers
    tokens = nltk.word_tokenize(text)
    tokens = [wnl.lemmatize(t) for t in tokens if t.isalpha() and t not in stop]
    return " ".join(tokens)

# ---------------------------
# 3. Explainability Function
# ---------------------------
def explain_with_highlight(message, top_n=5):
    clean_msg = clean_and_tokenize(message)
    num_chars = len(message)
    sample_df = pd.DataFrame({'clean_text': [clean_msg], 'num_characters': [num_chars]})

    # Prediction + probabilities
    proba = model_pipeline.predict_proba(sample_df)[0]
    pred = int(np.argmax(proba))
    pred_label = "spam" if pred == 1 else "ham"
    confidence = proba[pred]

    # LIME Explainer
    explainer = LimeTextExplainer(class_names=['ham', 'spam'])
    exp = explainer.explain_instance(
        clean_msg,
        classifier_fn=lambda x: model_pipeline.predict_proba(
            pd.DataFrame({'clean_text': x, 'num_characters': [len(m) for m in x]})),
        num_features=top_n
    )

    top_words = dict(exp.as_list()[:top_n])

    # Highlight words in message
    highlighted = []
    for word in message.split():
        clean_word = re.sub(r'\W+', '', word).lower()
        if clean_word in top_words:
            weight = top_words[clean_word]
            if (pred == 1 and weight > 0) or (pred == 0 and weight < 0):
                highlighted.append(f"<span style='color:red; font-weight:bold'>{word}</span>")
            else:
                highlighted.append(f"<span style='color:green; font-weight:bold'>{word}</span>")
        else:
            highlighted.append(word)

    # Build reasons list
    reasons = []
    for word, weight in top_words.items():
        if (pred == 1 and weight > 0):
            reasons.append(f"- The word '<b>{word}</b>' is strongly associated with SPAM. ❌")
        elif (pred == 0 and weight < 0):
            reasons.append(f"- The word '<b>{word}</b>' suggests HAM (safe). ✅")

    # Confidence bar
    bar_length = 20
    filled_length = int(bar_length * confidence)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    bar_display = f"[{bar}] {confidence*100:.1f}%"

    # Determine color and label text
    if pred_label == "spam":
        top_color = "red"
        top_text = "SPAM"
    else:
        top_color = "green"
        top_text = "NOT SPAM"

    # Final structured explanation
    explanation_html = f"""
    <hr>
    <h3>📢 The message is classified as: <span style='color:{top_color}; font-weight:bold'>{top_text}</span></h3>
    <p><b>Message:</b><br>"{' '.join(highlighted)}"</p>
    <p><b>Reasons for classification:</b><br>{"<br>".join(reasons) if reasons else "Based on overall content."}</p>
    <p><b>Confidence:</b> {bar_display}</p>
    <p>🤔 Based on these factors, this message is most likely <span style='color:{top_color}; font-weight:bold'>{top_text}</span>.</p>
    <hr>
    """

    return pred_label, explanation_html

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier with Explainability")

user_input = st.text_area("Enter your message:")

if st.button("Classify"):
    if user_input.strip():
        label, explanation_html = explain_with_highlight(user_input)
        st.markdown(explanation_html, unsafe_allow_html=True)
    else:
        st.warning("Please enter a message.")
