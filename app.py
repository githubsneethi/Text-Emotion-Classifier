import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Text Emotion Classifier", page_icon="🎭")
st.title("🎭 Text Emotion Classifier")
st.write("Enter text below to predict the emotion.")

# Label mapping for Nelgiriyewithana's emotions dataset
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Load model and vectorizer
@st.cache_resource
def load_pipeline():
    model = joblib.load('emotion_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_pipeline()
except Exception as e:
    st.error("Error loading model files. Ensure 'emotion_model.pkl' and 'tfidf_vectorizer.pkl' are in this folder.")
    st.stop()

# User Input
user_text = st.text_area("Input Text:", placeholder="Type a sentence here...")

if st.button("Predict Emotion"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # Transform input text
        text_vec = vectorizer.transform([user_text])

        # 1. Predict class
        pred = model.predict(text_vec)[0]

        # Handle label mapping
        try:
            pred_idx = int(pred)
            predicted_emotion = labels[pred_idx]
        except (ValueError, TypeError):
            predicted_emotion = str(pred)
            try:
                pred_idx = labels.index(predicted_emotion.lower())
            except ValueError:
                pred_idx = -1

        # Display Result
        st.success(f"Predicted Emotion: **{predicted_emotion.capitalize()}**")

        # 2. Calculate and Display Confidence
        st.subheader("Confidence Metrics")

        try:
            # If the model has standard predict_proba
            probabilities = model.predict_proba(text_vec)[0]
        except AttributeError:
            # If the model is LinearSVC, calculate probabilities using Softmax
            decision_scores = model.decision_function(text_vec)[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / exp_scores.sum()

        if pred_idx != -1:
            # Display highest confidence score
            confidence = probabilities[pred_idx]
            st.metric(label="Confidence Score", value=f"{confidence:.2%}")

            # 3. Create and display the Bar Chart
            st.write("### Emotion Probabilities")

            # Format data into a Pandas DataFrame for Streamlit's bar chart
            df_probs = pd.DataFrame({
                "Probability": probabilities
            }, index=[label.capitalize() for label in labels])

            # Sort the dataframe so the highest probability is at the top/left
            df_probs = df_probs.sort_values(by="Probability", ascending=False)

            # Render the bar chart
            st.bar_chart(df_probs)

        else:
            st.info("Confidence metrics are not available.")
