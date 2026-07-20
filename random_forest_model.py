import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

def load_and_preprocess_cardio_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    df['age_years'] = (df['age'] / 365.25).astype(int)
    
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        
    columns_to_drop = ['cardio', 'age']
    X = df.drop(columns=columns_to_drop)
    y = df['cardio']
    
    return X, y

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_rf_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
if __name__ == '__main__':
    csv_file = "cardio.csv"
    model_file = "rf_cardio_model.joblib"
    
    X, y = load_and_preprocess_cardio_data(csv_file)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = train_random_forest(X_train, y_train)
    evaluate_rf_model(rf_model, X_test, y_test)
    
    dump(rf_model, model_file)
    dump(X_test, "X_test_cardio.joblib")
    dump(y_test, "y_test_cardio.joblib")