Resume Screening System
A web-based Resume Screening System built using Streamlit that classifies resumes into job categories using Machine Learning. 
It supports bulk PDF/DOCX uploads and provides real-time category predictions along with extracted email addresses.

Features
- Upload multiple resumes at once (PDF or DOCX)
- Automatically extract and clean resume text
- Extract email from each resume
- Predict job category using **K-Nearest Neighbors**
- Stylish, modern interface (glassmorphism cards)
- Built with `Streamlit`, `NLTK`, `scikit-learn`, and `joblib`

Project Structure
resume-screening/
│
├── app.py # Streamlit app GUI
├── model_training.py # Model training script
├── UpdatedResumeDataSet.csv # Dataset for model training
└── README.md

Installation & Setup

1. Clone the repository
git clone https://github.com/ArpitKumarChauhan/Resume_Parsing.git
cd resume-screening
2. Install dependencies
pip install streamlit scikit-learn nltk joblib PyPDF2 python-docx
3. Train the models
python model_training.py
4. Run the Streamlit app
streamlit run app.py


Preprocessing:
Lowercasing
Stopword removal using nltk
Removing punctuation and numbers

Dataset
Dataset: UpdatedResumeDataSet.csv

Contains resumes with labeled job categories

Used for supervised learning during model training

Future Improvements

Export results to downloadable CSV


Feel free to fork, use, or improve it for personal or commercial use.
