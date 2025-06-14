import streamlit as st
import joblib

st.title("Fake News Detector")
st.write("Enter a news article to check if it's fake or not")

text_input = st.text_area("Paste the news article here:")

if st.button("Check"):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorized.pkl")

    txt_vector = vectorizer.transform([text_input])
    prediction = model.predict(txt_vector)

    if prediction[0] == 1:
        st.success("Real News!")
    else:
        st.error("Fake News!")
