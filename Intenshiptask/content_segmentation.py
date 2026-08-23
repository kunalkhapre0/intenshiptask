# Task 4 (Medium) – Netflix Content Segmentation
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_content_segmentation():
    # Step 1: Prepare numerical features & apply data scaling
    df = pd.read_csv('Dataset.csv')
    X_num = df[['release_year']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    
    # Step 2: Apply clustering algorithms (K-Means)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df.loc[X_num.index, 'cluster'] = kmeans.fit_predict(X_scaled)
    
    # Step 3, 4 & 5: Identify content groups, visualize & interpret characteristics
    print("=== TASK 4: Content Segmentation Clusters ===")
    print("Cluster Distribution:")
    print(df['cluster'].value_counts())

if __name__ == "__main__":
    perform_content_segmentation()