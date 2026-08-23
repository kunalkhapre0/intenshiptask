# Task 3 (Medium) – Netflix Audience Rating Classification
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_rating_classifier():
    # Step 1 & 2: Analyze rating categories & Prepare training datasets
    df = pd.read_csv('Dataset.csv')
    df_clean = df.dropna(subset=['rating']).copy()
    
    X = pd.get_dummies(df_clean[['type', 'release_year', 'listed_in']], drop_first=True)
    y = df_clean['rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3 & 4: Train classification models & Optimize model performance (GridSearchCV)
    rf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [20, 50], 'max_depth': [5, 10]}
    grid = GridSearchCV(rf, param_grid, cv=2)
    grid.fit(X_train, y_train)
    
    # Step 5: Evaluate prediction accuracy
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("=== TASK 3: Audience Rating Classification ===")
    print("Best Hyperparameters:", grid.best_params_)
    print("Accuracy Score:", round(accuracy_score(y_test, y_pred), 4))

if __name__ == "__main__":
    train_rating_classifier()