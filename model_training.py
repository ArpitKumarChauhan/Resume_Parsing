# model_training.py

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords_simple(text):
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

def plot_confusion(model_name, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

def main():
    print("Loading dataset and training models...")
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df = df[['Resume', 'Category']]
    df['Cleaned'] = df['Resume'].astype(str).apply(clean_text).apply(remove_stopwords_simple)

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['Cleaned'])
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Naive Bayes": MultinomialNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    accuracy_scores = {}

    labels = sorted(y.unique())  

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracy_scores[name] = acc

        print(f"Accuracy of {name}: {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        plot_confusion(name, y_test, y_pred, labels)


        filename = name.lower().replace(" ", "_") + "_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)


    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    plt.show()
    plt.close()

    print("\nAll models trained and saved successfully.")
    print("Evaluation graphs saved as PNG files.")

if __name__ == "__main__":
    main()
