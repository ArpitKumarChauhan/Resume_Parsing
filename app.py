import streamlit as st
import joblib
import PyPDF2
import docx
import re
import nltk

# NLTK Setup 
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Page Config 
st.set_page_config(page_title="Resume Screening System", layout="centered")

st.markdown("""
<div style="
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 30px;
    margin-top: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    border-left: 8px solid #4CAF50;
    text-align: center;
">
    <h1 style="font-size:45px; color:#2F4F4F; margin-bottom: 10px;">
         Resume Screening System
    </h1>
    <p style="font-size:20px; color:#555;">
        Upload multiple resumes (PDF or DOCX) and get job category predictions.</b>.
    </p>
</div>
<br>
""", unsafe_allow_html=True)


# Load Models 
@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer.pkl")
    knn_model = joblib.load("k-nearest_neighbors_model.pkl")
    return vectorizer, knn_model

vectorizer, knn_model = load_model()

#Helper Functions
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    except:
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    except:
        return ""

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    return " ".join([word for word in tokens if word not in stop_words])

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Not Found"

# File Upload 
uploaded_files = st.file_uploader(" Upload resumes here", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("<h2 style='color:#0066cc;'> Prediction Results</h2>", unsafe_allow_html=True)
    
    for file in uploaded_files:
        filename = file.name
        ext = filename.split(".")[-1].lower()

        # Extract text
        if ext == "pdf":
            text = extract_text_from_pdf(file)
        elif ext == "docx":
            text = extract_text_from_docx(file)
        else:
            st.error(f" Unsupported file format: {ext}")
            continue

        if not text.strip():
            st.error(f" No readable text found in: {filename}")
            continue

        # Clean & predict
        cleaned = clean_text(text)
        transformed = vectorizer.transform([cleaned])
        prediction = knn_model.predict(transformed)[0]
        email = extract_email(text)

        # Modern Card Design
        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-top: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border-left: 8px solid #4CAF50;
        ">
            <h3 style="color:#333; margin-bottom:10px;"> <u>{filename}</u></h3>
            <p style="font-size:17px;"><strong> Email:</strong> <code>{email}</code></p>
            <p style="font-size:17px;"><strong> Predicted Category:</strong> 
                <span style="background-color:#4CAF50; color:white; padding:5px 10px; border-radius:5px;">{prediction}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

#Footer 
st.markdown("""
<br><hr style='border:1px solid #ddd;'>
<div style='text-align: center; font-size: 15px; color: gray;'>
    &copy; 2025 Resume Screening System | Developed by Screenswift.
</div>
""", unsafe_allow_html=True)
