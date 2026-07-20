import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score
from joblib import dump

def load_spam_data(csv_path: str) -> tuple[pd.Series, pd.Series]:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        raise

    text_column = 'text'
    label_column = 'spam'

    X = df[text_column]
    y = df[label_column]
    
    return X, y

def train_spam_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    model = make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X_train, y_train)
    return model

def evaluate_spam_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

def save_spam_model(model: Pipeline, filename: str = "spam_naive_bayes.joblib"):
    dump(model, filename)

if __name__ == '__main__':
    csv_file = "emails.csv"
    model_file = "spam_naive_bayes.joblib"
    
    # Data loading and preparation
    X, y = load_spam_data(csv_file)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a naive Bayes model
    model = train_spam_model(X_train, y_train)

    # Evaluate the model
    evaluate_spam_model(model, X_test, y_test)

    # Save the model
    save_spam_model(model, model_file)

    dump(X_test, "X_test_spam.joblib")
    dump(y_test, "y_test_spam.joblib")