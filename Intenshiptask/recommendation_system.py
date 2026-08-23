# Task 1 (Easy) – Netflix Content Recommendation System
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data():
    return pd.read_csv('Dataset.csv')

def build_recommendation_system(df):
    # Step 1: Prepare content-related features
    df['content_mix'] = df['listed_in'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['cast'].fillna('')
    
    # Step 2: Convert text data into machine-readable format (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_mix'])
    
    # Step 3: Calculate content similarity scores
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_title(df, cosine_sim, title, top_n=5):
    # Step 4 & 5: Generate recommendations & Evaluate quality
    matching = df[df['title'] == title]
    if matching.empty:
        return f"Title '{title}' not found in dataset."
    idx = matching.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_indices = [i[0] for i in sim_scores]
    return df.iloc[rec_indices][['title', 'type', 'listed_in']]

if __name__ == "__main__":
    df = load_data()
    sim_matrix = build_recommendation_system(df)
    sample_title = df['title'].iloc[0]
    print(f"=== TASK 1: Recommendations for '{sample_title}' ===")
    print(recommend_title(df, sim_matrix, sample_title))