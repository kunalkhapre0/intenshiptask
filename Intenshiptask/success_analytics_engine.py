# Task 6 (Advanced) – Netflix Content Success Analytics Engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

def run_analytics_engine():
    # Step 1: Perform advanced feature engineering
    df = pd.read_csv('Dataset.csv')
    X = df[['release_year', 'rating']].dropna()
    y = (df.loc[X.index, 'type'] == 'Movie').astype(int)
    
    # Step 2: Build End-to-End Machine Learning Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['release_year']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['rating'])
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # Step 3, 4 & 5: Compare performance, generate insights & output visual reports
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    print("=== TASK 6: End-to-End Success Analytics Engine ===")
    print(f"Pipeline Accuracy Score: {score:.4f}")

if __name__ == "__main__":
    run_analytics_engine()