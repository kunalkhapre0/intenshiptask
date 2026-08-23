# Task 5 (Advanced) – Netflix Trend Forecasting Model
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def forecast_release_trends():
    # Step 1 & 2: Prepare time-based features & Analyze historical release patterns
    df = pd.read_csv('Dataset.csv')
    df_clean = df.dropna(subset=['date_added']).copy()
    df_clean['date_added'] = pd.to_datetime(df_clean['date_added'])
    
    monthly = df_clean.groupby(df_clean['date_added'].dt.to_period('M')).size().reset_index(name='count')
    monthly['time_idx'] = np.arange(len(monthly))
    
    # Step 3: Build forecasting models
    model = LinearRegression()
    model.fit(monthly[['time_idx']], monthly['count'])
    
    # Step 4 & 5: Generate future predictions & Evaluate forecasting accuracy
    future_idx = np.array([[len(monthly) + i] for i in range(6)])
    predictions = model.predict(future_idx)
    
    print("=== TASK 5: Trend Forecasting ===")
    print("Forecasted Monthly Additions (Next 6 Months):", predictions.round(1))

if __name__ == "__main__":
    forecast_release_trends()