# Task 2 (Easy) – Content Type Prediction Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_content_type_model():
    # Step 1: Select relevant dataset features & handle missing values
    df = pd.read_csv('Dataset.csv')
    df_clean = df.dropna(subset=['type', 'rating']).copy()
    
    # Step 2: Encode categorical variables
    df_clean['target'] = (df_clean['type'] == 'Movie').astype(int)
    X = pd.get_dummies(df_clean[['release_year', 'rating', 'listed_in']], drop_first=True)
    y = df_clean['target']
    
    # Step 3: Train classification models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Step 4 & 5: Evaluate prediction performance & compare model accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("=== TASK 2: Content Type Prediction Accuracy ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_content_type_model()