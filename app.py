import streamlit as st
import pickle

model_path = "model.pkl"
vectorizer_path = "tf_idf_vectorizer.pkl"
pca_path = "ml_pca.pkl"

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(pca_path, "rb") as file:
    pca = pickle.load(file)

st.title("Hate Speech Detection")

phrase = st.text_input("Enter a sentence or phrase:", "")

if st.button("Check"):
    if phrase.strip():
        # Preprocess and predict
        X_vectorized = vectorizer.transform([phrase])
        X_pca = pca.transform(X_vectorized.toarray())
        prediction = model.predict(X_pca)

        # Display result
        sentiment = "A Hate Speech" if prediction[0] == 1 else "Not A Hate Speech"
        st.success(f"The statement is: {sentiment}")
    else:
        st.error("Please enter a phrase.")
