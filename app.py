import streamlit as st

st.set_page_config(page_title="Resume Screening System", layout="centered")

import joblib
import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


@st.cache_resource
def load_models():
    vectorizer = joblib.load("vectorizer.pkl")
    lr_model = joblib.load("logistic_regression_model.pkl")
    nb_model = joblib.load("naive_bayes_model.pkl")
    knn_model = joblib.load("k-nearest_neighbors_model.pkl")
    return vectorizer, lr_model, nb_model, knn_model

vectorizer, lr_model, nb_model, knn_model = load_models()


def majority_vote(preds):
    count = Counter(preds)
    return count.most_common(1)[0][0]



st.title("📄 Resume Parser")
st.write("Upload your resume in PDF or DOCX format only.")

uploaded_file = st.file_uploader("Choose your resume file", type=["pdf", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_type == "pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            raw_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    cleaned_text = clean_text(raw_text)
    transformed_text = vectorizer.transform([cleaned_text])

    pred_lr = lr_model.predict(transformed_text)[0]
    pred_nb = nb_model.predict(transformed_text)[0]
    pred_knn = knn_model.predict(transformed_text)[0]

    all_preds = [pred_lr, pred_nb, pred_knn]
    final_prediction = majority_vote(all_preds)

    st.subheader("🔍 Model Predictions")
    st.write(f"**Logistic Regression**: {pred_lr}")
    st.write(f"**Naive Bayes**: {pred_nb}")
    st.write(f"**K-Nearest Neighbors**: {pred_knn}")

    st.success(f"🎯 Final Category Prediction :  {final_prediction}")
